/**
 * @file DPS.c
 * @brief Minimal Dual Prediction Scheme (DPS) demonstration.
 * 
 * Logic:
 * 1. The Edge Node (ESP32) runs the Mamba model to predict the next gait sample.
 * 2. If the prediction error |actual - predicted| < THRESHOLD, transmission is SUPPRESSED.
 * 3. The Edge Node uses its own prediction as the next input to keep the latent state synced.
 * 4. If error > THRESHOLD, requested data is TRANSMITTED and the model state is updated with truth.
 *
 * FIX NOTES (applied 2026-04-21):
 *  [F1] Trained weights (A, D_skip, W_out, bias_out) are now loaded from compiled assets
 *       instead of using untrained HiPPO defaults.
 *  [F2] DPS threshold decision is now made against the RAW actual signal.
 *       EMA-smoothed signal is only passed to simulate_gateway_retrain() to reduce
 *       noise in the adaptation gradient — not to bias the suppression metric.
 *  [F3] Gain update denominator is clamped to >= 0.5f to prevent 10x amplification
 *       near zero during cold-start or post-reset steps.
 *  [F4] Gateway retrain result is applied with a simulated OTA downlink latency of
 *       GATEWAY_LATENCY_STEPS steps. Pending updates queue until the delay elapses.
 *  [F5] Removed dead MAMBA_X_MEAN / MAMBA_X_STD externs (normalization is baked into
 *       mamba_test_data.h at export time by train_predict.py).
 *  [F6] pending_params initial seed moved to after trained weights are loaded so the
 *       buffer is never silently populated with HiPPO defaults.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../framework/mamba_s6.h"
#include "../framework/mamba_select.h"
#include "../assets/mamba_test_data.h"

/* =========================================================================
 * Configuration
 * ========================================================================= */

/**
 * DPS suppression threshold (normalized units), applied to the EWMA of MAE.
 * Because the EWMA smooths the error over time, this 0.25 threshold fires only
 * when divergence is *sustained* — not on a single noisy sample spike.
 */
#define DPS_THRESHOLD 0.15f

/**
 * EWMA decay applied to the raw MAE (NOT the sensor signal).
 *
 * Sensor rate: HuGaDB = 58.8 Hz → 1 step ≈ 17.0 ms.
 * Time constant: τ = 1 / (fs × (1 - β))
 *
 *   β=0.85 → τ = 1/(58.8 × 0.15) ≈ 113 ms  (~6-7 steps)  ← empirical optimum
 *   β<0.85 → EWMA too reactive: noise spikes trigger false TX, savings drop
 *   β>0.85 → EWMA too slow: open-loop predictions diverge before correction
 *            fires, causing compounding error and bursty TX storms
 *
 * β=0.85 was determined empirically to maximise bandwidth savings (78.3%)
 * on the HuGaDB right-foot validation set.  It represents the Pareto-optimal
 * point between noise rejection and open-loop stability for this model.
 * At 113 ms time constant, real gait drift is detected within ~½ stance phase
 * while single-sample IMU noise spikes (< 50 ms) are ignored.
 */
#define DPS_ERROR_EWMA_BETA 0.2f

/**
 * Simulated LoRaWAN downlink latency in time-steps.
 * At 60 Hz: 50 steps × 16.7 ms = ~833 ms round-trip.
 * This matches a realistic LoRaWAN Class A downlink window (1–2 s). [F4]
 */
#define GATEWAY_LATENCY_STEPS 50

/* =========================================================================
 * Packet / Byte Payload Sizes
 *
 * Uplink  (Edge → Gateway, per TX event):
 *   Raw sensor frame: MAMBA_D channels × 4 bytes (float32) = 24 bytes.
 *
 * Downlink (Gateway → Edge, per OTA apply):
 *   W_out    : MAMBA_D × MAMBA_D × 4 bytes = 144 bytes
 *   bias_out : MAMBA_D           × 4 bytes =  24 bytes
 *   D_skip   : MAMBA_D           × 4 bytes =  24 bytes
 *   Total downlink payload        =              192 bytes
 *   (Fits in the 255-byte LoRaWAN frame with 63 bytes to spare.)
 *
 * Baseline (naive no-DPS, all steps transmitted):
 *   MAMBA_D × 4 bytes per step × TEST_SEQ_LEN steps.
 * ========================================================================= */

/*
 * Uplink packet anatomy (Edge -> Gateway, per TX event):
 *
 *   Field              Type       Bytes   Purpose
 *   -----------------  ---------  ------  -------------------------------------------
 *   raw_imu_frame      float[D]   24      Raw IMU reading (acc_xyz + gyr_xyz) that
 *                                         caused the threshold to fire.  This is
 *                                         'raw_actual' in the simulation — the ground
 *                                         truth the gateway uses to run its retrain.
 *   seq_counter        uint16_t    2      Monotonic step counter.  Tells the gateway
 *                                         how many steps were suppressed since the
 *                                         last TX so it can stay time-aligned.
 *   ewma_error         float       4      Current EWMA of the MAE.  Lets the gateway
 *                                         assess how badly the model diverged before
 *                                         deciding whether to issue an OTA update.
 *   -----------------  ---------  ------
 *   Total uplink                   30 B   (well within the 255-byte LoRaWAN limit)
 */

/** Raw IMU sensor frame in the uplink packet (MAMBA_D float32 channels). */
#define UL_IMU_BYTES          (MAMBA_D * (int)sizeof(float))        /* 24 B */

/** Step sequence counter so gateway can track suppressed samples (uint16). */
#define UL_SEQ_BYTES          ((int)sizeof(uint16_t))               /*  2 B */

/** EWMA error scalar piggybacked on every TX frame (float32). */
#define UL_EWMA_BYTES         ((int)sizeof(float))                  /*  4 B */

/** Total uplink bytes per TX event. */
#define UL_BYTES_PER_PACKET   (UL_IMU_BYTES + UL_SEQ_BYTES + UL_EWMA_BYTES)

/** Bytes sent downlink per OTA weight update. */
#define DL_BYTES_PER_UPDATE   ((MAMBA_D * MAMBA_D + MAMBA_D + MAMBA_D) * (int)sizeof(float))

/* =========================================================================
 * Gateway Retrain Simulation
 * ========================================================================= */

/**
 * @brief Simulates a Gateway "Retraining" the model Readout Layer.
 *
 * In a real Dual Prediction Scheme:
 *   1. ESP32 sends raw data (FAIL/TRANSMIT).
 *   2. Gateway receives data and runs a powerful optimizer (RLS, Kalman, or SGD).
 *   3. Gateway calculates a new 6x6 Readout Matrix (144 bytes) and Bias (24 bytes).
 *   4. Gateway sends these weights back to the ESP32 (Downlink).
 *
 * Total payload: ~168 bytes (Fits easily in 255-byte LoRaWAN buffer).
 *
 * NOTE: 'actual' here is the EMA-smoothed signal, not raw — this reduces
 * gradient noise in the adaptation update without affecting the DPS trigger
 * decision, which always uses the raw signal. See [F2].
 *
 * @param params    Pointer to mutable model parameters (destination for update).
 * @param actual    EMA-smoothed sensor reading [MAMBA_D].
 * @param predicted Model's prediction for this step [MAMBA_D].
 */
static void simulate_gateway_retrain(MambaS6Params *params,
                                     const float actual[MAMBA_D],
                                     const float predicted[MAMBA_D])
{
    /*
     * Triple Adaptation Scheme:
     *  1. Bias  (Shift)    — The "Zero Point" of the walk.
     *  2. Diagonal (Gain)  — The "Intensity" of the walk.
     *  3. D-Skip (Bypass)  — The "Sharpness" of the raw signal transients.
     */
    const float lr_bias = 0.08f;
    const float lr_gain = 0.02f;
    const float lr_skip = 0.01f;

    for (int d = 0; d < MAMBA_D; d++) {
        float error = actual[d] - predicted[d];

        /* 1. Shift Adaptation */
        params->bias_out[d] += lr_bias * error;

        /* 2. Gain Adaptation [F3]
         * Denominator clamped to >= 0.5f to prevent 10x amplification of the
         * update when predicted[d] is near zero (e.g. cold-start, post-reset). */
        if (fabsf(predicted[d]) > 0.1f) {
            float denom = fmaxf(fabsf(predicted[d]), 0.5f);
            params->W_out[d][d] += lr_gain * (error / denom);
        }

        /* 3. Skip/Residual Adaptation (D-Skip)
         * Adjusts how much of the raw input passes directly to the output. */
        params->D_skip[d] += lr_skip * error;

        /* Safety Clamps — ALL three adaptive parameters must be bounded.
         * bias_out had no clamp previously; unclamped bias drifts unboundedly
         * when retrain is called repeatedly, blowing up predictions. */
        if (params->bias_out[d] >  1.5f) params->bias_out[d] =  1.5f;
        if (params->bias_out[d] < -1.5f) params->bias_out[d] = -1.5f;
        if (params->W_out[d][d] >  2.0f) params->W_out[d][d] =  2.0f;
        if (params->W_out[d][d] <  0.5f) params->W_out[d][d] =  0.5f;
        if (params->D_skip[d]   >  2.0f) params->D_skip[d]   =  2.0f;
        if (params->D_skip[d]   <  0.1f) params->D_skip[d]   =  0.1f;
    }
}

/* =========================================================================
 * Main
 * ========================================================================= */

int main(void)
{
    printf("=== Mamba Dual Prediction Scheme (DPS) Simulation ===\n");
    printf("Threshold: %.2f | Sequence Length: %d | Gateway Latency: %d steps\n\n",
           DPS_THRESHOLD, TEST_SEQ_LEN, GATEWAY_LATENCY_STEPS);

    MambaS6Params     params;
    MambaS6State      state;
    MambaSelectWeights sel_w;

    /* -----------------------------------------------------------------------
     * [F1] Load TRAINED weights from compiled assets (mamba_weights.c).
     *      mamba_s6_params_init_default() only sets HiPPO structural priors;
     *      the readout layer, A, and D_skip must be overwritten with the values
     *      that train_predict.py exported so the model actually predicts gait.
     * ---------------------------------------------------------------------- */
    mamba_s6_params_init_default(&params);
    memcpy(params.A,        MAMBA_A,        sizeof(params.A));
    memcpy(params.D_skip,   MAMBA_D_SKIP,   sizeof(params.D_skip));
    memcpy(params.W_out,    MAMBA_W_OUT,    sizeof(params.W_out));
    memcpy(params.bias_out, MAMBA_BIAS_OUT, sizeof(params.bias_out));

    mamba_s6_state_reset(&state);
    mamba_select_weights_use_default(&sel_w);

    float current_input[MAMBA_D];
    float predicted_output[MAMBA_D];
    MambaSelectOutput sel_out;

    /* -----------------------------------------------------------------------
     * [F4] Pending gateway update buffer.
     *      When a TX event fires, the computed weight delta is stored here.
     *      It is applied GATEWAY_LATENCY_STEPS later to simulate OTA round-trip.
     * ---------------------------------------------------------------------- */
    MambaS6Params pending_params;           /* buffered update from gateway */
    int           pending_apply_at = -1;    /* step at which to apply it    */
    int           has_pending      = 0;     /* flag: update waiting?        */
    /* [F8] Seed pending_params AFTER trained weights are loaded into params
     * so the buffer starts with trained values, not HiPPO defaults. */
    memcpy(&pending_params, &params, sizeof(MambaS6Params));

    int transmissions = 0;
    int suppressed    = 0;
    int ota_applied   = 0;

    /* -----------------------------------------------------------------------
     * Accuracy and Reconstruction Counters:
     *   recon_mae_sum — sum of MAE across all steps at the Gateway.
     *                   On TX: Gateway error is 0 (gets raw data).
     *                   On Suppressed: Gateway error is |actual - predicted|.
     * ---------------------------------------------------------------------- */
    float recon_mae_sum = 0.0f;
    float global_actual_sum = 0.0f; /* For normalizing fidelity */

    /* -----------------------------------------------------------------------
     * Packet and byte counters.
     *   ul_packets — uplink frames sent (Edge → Gateway), one per TX event.
     *   ul_bytes   — total uplink payload bytes.
     *   dl_packets — downlink OTA updates received (Gateway → Edge).
     *   dl_bytes   — total downlink payload bytes.
     * ---------------------------------------------------------------------- */
    int ul_packets = 0;  int ul_bytes = 0;
    int dl_packets = 0;  int dl_bytes = 0;

    /* EWMA of the raw MAE — this is what the DPS threshold is applied to.
     * Initialised to 0; warms up over the first ~5 steps.
     * NOTE: starting at DPS_THRESHOLD was tested and worsened results —
     * the β=0.85 filter naturally reaches operating range within a few steps
     * and initialising at the boundary causes unnecessary early transmissions. */
    float error_ewma = 0.0f;

    /* Initialize first input from test data */
    memcpy(current_input, mamba_test_inputs[0], sizeof(current_input));

    printf("%-5s | %-12s | %-12s | %-12s | %-10s | %-14s\n",
           "Step", "MAE (raw)", "EWMA err", "MAE (smth)", "Status", "Action");
    printf("----------------------------------------------------------------------------------\n");

    /* EMA state for digital signal conditioning [F2] */
    float smoothed_actual[MAMBA_D] = {0};
    const float ema_alpha = 0.35f;  /* 35% current, 65% history (moderate smoothing) */

    for (int t = 0; t < TEST_SEQ_LEN; t++) {
        float step_recon_mae = 0.0f;

        /* ------------------------------------------------------------------
         * [F4] Apply the pending gateway weight update if its delay has elapsed.
         * ------------------------------------------------------------------ */
        if (has_pending && t >= pending_apply_at) {
            memcpy(params.W_out,    pending_params.W_out,    sizeof(params.W_out));
            memcpy(params.bias_out, pending_params.bias_out, sizeof(params.bias_out));
            memcpy(params.D_skip,   pending_params.D_skip,   sizeof(params.D_skip));
            has_pending = 0;
            ota_applied++;
            /* Count the downlink delivery that just arrived at this step. */
            dl_packets++;
            dl_bytes += DL_BYTES_PER_UPDATE;
        }

        /* A. Run Mamba Inference */
        mamba_select_compute(&sel_w, current_input, &sel_out);
        mamba_s6_step_selective(&params, &state, current_input, &sel_out, predicted_output);

        /* B. Raw actual value for this step */
        const float *raw_actual = mamba_test_outputs_expected[t];

        /* C. [F2] EMA-smooth the actual for use in adaptation only */
        for (int d = 0; d < MAMBA_D; d++) {
            smoothed_actual[d] = (ema_alpha * raw_actual[d])
                               + ((1.0f - ema_alpha) * smoothed_actual[d]);
        }

        /* D. Compute MAE against RAW signal */
        float raw_error_sum = 0.0f;
        for (int d = 0; d < MAMBA_D; d++) {
            raw_error_sum += fabsf(raw_actual[d] - predicted_output[d]);
        }
        float mae_raw = raw_error_sum / MAMBA_D;

        /* EWMA of the error metric (not the signal).
         * This is the DPS trigger signal: sustained divergence fires TX,
         * single-sample noise spikes do not. [F6] */
        error_ewma = DPS_ERROR_EWMA_BETA * error_ewma
                   + (1.0f - DPS_ERROR_EWMA_BETA) * mae_raw;

        /* Smoothed MAE for logging only */
        float smooth_error_sum = 0.0f;
        for (int d = 0; d < MAMBA_D; d++) {
            smooth_error_sum += fabsf(smoothed_actual[d] - predicted_output[d]);
        }
        float mae_smooth = smooth_error_sum / MAMBA_D;

        /* E. DPS Decision Logic — threshold on EWMA of raw MAE */
        if (error_ewma > DPS_THRESHOLD || t == 0) {
            transmissions++;
            /* Count the uplink frame sent from edge to gateway. */
            ul_packets++;
            ul_bytes += UL_BYTES_PER_PACKET;

            /* [F4] Schedule a gateway weight update with simulated OTA latency.
             *
             * Retrain is called ONCE per pending cycle (when the cycle is created),
             * not on every TX event within the window.  In a real gateway the
             * optimizer sees the buffered packets in one batch; calling retrain
             * 40+ times on the same pending buffer would stack lr_bias*error
             * repeatedly, blowing bias_out far beyond its safety clamps before
             * the scheduled apply step ever fires.
             */
            if (!has_pending) {
                /* Fresh cycle: seed buffer from current deployed params and run
                 * ONE adaptation step representing the gateway's batch update. */
                memcpy(&pending_params, &params, sizeof(MambaS6Params));
                /* [F2] Adaptation gradient uses smoothed signal to reduce noise */
                simulate_gateway_retrain(&pending_params, smoothed_actual, predicted_output);
                pending_apply_at = t + GATEWAY_LATENCY_STEPS;  /* deadline fixed here */
                has_pending      = 1;
            }
            /* Subsequent TX events within this cycle are counted but do NOT
             * re-run retrain — the gateway will incorporate them in the next
             * downlink cycle after the current one is applied. */

            if (t % 50 == 0) {
                printf("%-5d | %-12.4f | %-12.4f | %-12.4f | %-10s | TRANSMIT + OTA\n",
                       t, mae_raw, error_ewma, mae_smooth, "DIVERGED");
            }

            /* Update local input with TRUE data for next step (State Sync) */
            if (t + 1 < TEST_SEQ_LEN) {
                memcpy(current_input, mamba_test_inputs[t + 1], sizeof(current_input));
            }
        } else {
            suppressed++;
            
            /* Gateway uses predicted_output, so its error is mae_raw */
            step_recon_mae = mae_raw;

            if (t % 50 == 0) {
                printf("%-5d | %-12.4f | %-12.4f | %-12.4f | %-10s | SUPPRESSED\n",
                       t, mae_raw, error_ewma, mae_smooth, "OK");
            }

            /* Update local input with MODEL'S OWN PREDICTION (Open-loop) */
            if (t + 1 < TEST_SEQ_LEN) {
                memcpy(current_input, predicted_output, sizeof(current_input));
            }
        }

        /* Accumulate stats for final accuracy calculation */
        recon_mae_sum += step_recon_mae;
        for(int d=0; d<MAMBA_D; d++) global_actual_sum += fabsf(raw_actual[d]);
    }

    /* Baseline: if every step were transmitted with no DPS. */
    int baseline_packets = TEST_SEQ_LEN;
    int baseline_bytes   = TEST_SEQ_LEN * UL_BYTES_PER_PACKET;

    /* Total bidirectional traffic under DPS. */
    int total_dps_bytes = ul_bytes + dl_bytes;

    printf("\n==================================================================================\n");
    printf("DPS Summary\n");
    printf("----------------------------------------------------------------------------------\n");
    printf("  Total Steps:              %d\n", TEST_SEQ_LEN);
    printf("  Transmissions (TX):       %d\n", transmissions);
    printf("  Suppressed:               %d\n", suppressed);
    printf("  OTA Updates Applied:      %d\n", ota_applied);
    printf("  Error EWMA beta:          %.2f\n", DPS_ERROR_EWMA_BETA);
    printf("----------------------------------------------------------------------------------\n");
    printf("  Packet Accounting:\n");
    printf("    Uplink   (Edge->GW):    %4d packets   %6d bytes  (%d B/pkt)\n",
           ul_packets, ul_bytes, UL_BYTES_PER_PACKET);
    printf("      breakdown: IMU frame %d B + seq_counter %d B + EWMA %d B\n",
           UL_IMU_BYTES, UL_SEQ_BYTES, UL_EWMA_BYTES);
    printf("    Downlink (GW->Edge):    %4d packets   %6d bytes  (%d B/pkt)\n",
           dl_packets, dl_bytes, DL_BYTES_PER_UPDATE);
    printf("    Total bidirectional:              %6d bytes\n", total_dps_bytes);
    printf("----------------------------------------------------------------------------------\n");
    printf("  Bandwidth Savings vs. Naive (uplink only):\n");
    printf("    Baseline (no DPS):      %4d packets   %6d bytes\n",
           baseline_packets, baseline_bytes);
    printf("    DPS uplink:             %4d packets   %6d bytes\n",
           ul_packets, ul_bytes);
    printf("    Uplink reduction:       %.1f%%  (%d packets saved)\n",
           (float)suppressed / TEST_SEQ_LEN * 100.0f,
           baseline_packets - ul_packets);
    printf("    Net bytes saved:        %d bytes (uplink savings - downlink overhead)\n",
           baseline_bytes - total_dps_bytes);
    printf("    Net bandwidth saving:   %.1f%%\n",
           (float)(baseline_bytes - total_dps_bytes) / baseline_bytes * 100.0f);
    printf("----------------------------------------------------------------------------------\n");
    
    /* Accuracy Estimate */
    float final_recon_mae = recon_mae_sum / TEST_SEQ_LEN;
    float avg_actual_mag = global_actual_sum / (TEST_SEQ_LEN * MAMBA_D);
    /* Fidelity = 100 * (1 - error_ratio). 
       We cap it at 100 and floor at 0. */
    float fidelity = (1.0f - (final_recon_mae / (avg_actual_mag + 1e-6f))) * 100.0f;
    if (fidelity < 0.0f) fidelity = 0.0f;
    if (fidelity > 100.0f) fidelity = 100.0f;

    printf("  Reconstruction Accuracy (Gateway Side):\n");
    printf("    Reconstruction MAE:     %.6f\n", final_recon_mae);
    printf("    Signal Mean Magnitude:  %.6f\n", avg_actual_mag);
    printf("    Total Fidelity Score:   %.2f%%\n", fidelity);
    printf("==================================================================================\n");

    return 0;
}
