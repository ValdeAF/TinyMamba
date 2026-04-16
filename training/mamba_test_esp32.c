/**
 * @file mamba_test_esp32.c
 * @brief Step 4 — ESP32 Test Framework for the Mamba Gait Inference Engine.
 *
 * ==========================================================================
 * OVERVIEW
 * ==========================================================================
 *
 * This file provides a self-contained test suite that can be compiled and
 * run in three environments:
 *
 *   1. ESP-IDF  (recommended target)  — uses esp_timer_get_time() for μs timing.
 *   2. Arduino-ESP32                  — uses micros() and Serial.print().
 *   3. Host PC (GCC/WSL)              — uses clock() from <time.h> for validation.
 *
 * ==========================================================================
 * TEST MODES  (select via compile-time define)
 * ==========================================================================
 *
 *   MAMBA_TEST_MODE_INFERENCE (default)
 *     - Initialises the model (params, state, scaler, weights).
 *     - Runs 10 hand-crafted GaitSample time-steps.
 *     - After each step prints to Serial/UART:
 *         • The normalised input vector x_t[D]
 *         • The selective parameters: Δ[D], B[N], C[N]
 *         • The output vector y_t[D]
 *         • The full hidden state h_t for channel 0: h[0][N]
 *         • Per-step assertion: all outputs finite, all Δ > 0
 *
 *   MAMBA_TEST_MODE_DRY_RUN
 *     - Suppresses all per-step output (avoids UART bottleneck skewing timing).
 *     - Runs the full 10-step sequence DRY_RUN_REPS times (default 200).
 *     - Reports per-step timing statistics:
 *         min / max / mean / total (in µs and ms)
 *     - Also reports the three sub-stage costs separately:
 *         Stage 1 — Serialisation + normalisation
 *         Stage 2 — Selection mechanism (GEMV × 3 + softplus)
 *         Stage 3 — SSM recurrence (ZOH + state update + output)
 *
 * Enable dry-run mode:
 *   ESP-IDF build flag:  -DMAMBA_TEST_MODE=MAMBA_TEST_MODE_DRY_RUN
 *   Arduino:             #define MAMBA_TEST_MODE MAMBA_TEST_MODE_DRY_RUN
 *                        (place before #include, or in build_flags in platformio.ini)
 *
 * ==========================================================================
 * BUILD COMMANDS
 * ==========================================================================
 *
 * Host (GCC) — inference mode:
 *   gcc -O2 -Wall \
 *       mamba_s6.c mamba_select.c mamba_weights.c mamba_gait.c \
 *       mamba_test_esp32.c -lm -o mamba_test && ./mamba_test
 *
 * Host (GCC) — dry-run mode:
 *   gcc -O2 -Wall -DMAMBA_TEST_MODE=MAMBA_TEST_MODE_DRY_RUN \
 *       mamba_s6.c mamba_select.c mamba_weights.c mamba_gait.c \
 *       mamba_test_esp32.c -lm -o mamba_dryrun && ./mamba_dryrun
 *
 * ESP-IDF (CMakeLists.txt):
 *   idf_component_register(
 *       SRCS "mamba_s6.c" "mamba_select.c" "mamba_weights.c"
 *            "mamba_gait.c" "mamba_test_esp32.c"
 *       INCLUDE_DIRS "."
 *   )
 *   target_compile_definitions(${COMPONENT_LIB} PRIVATE
 *       MAMBA_D=16  MAMBA_N=16
 *       # Uncomment for dry-run:
 *       # MAMBA_TEST_MODE=MAMBA_TEST_MODE_DRY_RUN
 *   )
 *   target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffast-math)
 *
 * Rename mamba_test_main() to app_main() for ESP-IDF, or keep as main()
 * for host testing.
 */

/* =========================================================================
 * Includes — must come before all other local includes
 * ========================================================================= */

#ifdef ARDUINO
    /* Arduino-ESP32: Serial object is global. */
    #include <Arduino.h>
#endif

#include "mamba_gait.h"      /* GaitSample, serialization, Min-Max scaler   */
#include "../framework/mamba_select.h"    /* MambaSelectWeights, MambaSelectOutput        */
#include "../framework/mamba_s6.h"        /* MambaS6Params, MambaS6State, SSM step        */

#include <math.h>            /* fabsf, sqrtf, isfinite                       */
#include <stdio.h>           /* printf                                        */
#include <string.h>          /* memset                                        */
#include <stdint.h>          /* int64_t, uint32_t                             */

/* =========================================================================
 * Platform-specific timer abstraction
 *
 * mamba_timer_now_us() returns a monotonic microsecond timestamp as int64_t.
 * Resolution on ESP32: 1 µs (hardware timer backed by APB clock).
 * Resolution on host:  CLOCKS_PER_SEC granularity (~1 µs on modern Linux).
 * ========================================================================= */
#if defined(IDF_VER) || defined(ESP_PLATFORM)
    /* ---- ESP-IDF -------------------------------------------------------- */
    #include "esp_timer.h"
    static inline int64_t mamba_timer_now_us(void)
    {
        return esp_timer_get_time();   /* µs, monotonic, 64-bit */
    }

#elif defined(ARDUINO)
    /* ---- Arduino-ESP32 -------------------------------------------------- */
    static inline int64_t mamba_timer_now_us(void)
    {
        return (int64_t)micros();      /* µs, wraps at ~70 min on 32-bit */
    }

#else
    /* ---- Host / POSIX fallback ------------------------------------------ */
    #include <time.h>
    static inline int64_t mamba_timer_now_us(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t)ts.tv_sec * 1000000LL + (int64_t)(ts.tv_nsec / 1000);
    }
#endif

/* =========================================================================
 * Serial output abstraction
 *
 * On Arduino the Serial object uses its own API.  On ESP-IDF and host,
 * printf() writes to UART0 (configured in sdkconfig) / stdout.
 * We define MAMBA_PRINT() so the body of the test function is platform-
 * agnostic.
 * ========================================================================= */
#ifdef ARDUINO
    #define MAMBA_PRINT(...)   Serial.printf(__VA_ARGS__)
#else
    #define MAMBA_PRINT(...)   printf(__VA_ARGS__)
#endif

/* =========================================================================
 * Test mode constants
 * ========================================================================= */
#define MAMBA_TEST_MODE_INFERENCE  1
#define MAMBA_TEST_MODE_DRY_RUN    2

#ifndef MAMBA_TEST_MODE
#define MAMBA_TEST_MODE  MAMBA_TEST_MODE_INFERENCE
#endif

/** Number of full 10-step sequence repetitions in dry-run mode. */
#ifndef DRY_RUN_REPS
#define DRY_RUN_REPS  200
#endif

/** Total dry-run steps (for computing per-step statistics). */
#define DRY_RUN_TOTAL_STEPS  (DRY_RUN_REPS * 10)

/* =========================================================================
 * Hardcoded 10-timestep GaitSample dataset
 *
 * Each row represents one stride of walking-speed gait measured by a
 * foot-mounted IMU + pressure insole.  Values are chosen to cover a range
 * of typical clinical measurements and to exercise all 16 channels of the
 * MAMBA_D=16 model with varied, non-trivial inputs.
 *
 * Field order matches the GaitSample struct (same as descriptor table):
 *   [0]  stride_length_mm      — uint16  [0, 2500]
 *   [1]  cadence_steps_per_min — float   [0, 200]
 *   [2]  stance_time_ms        — uint16  [0, 1500]
 *   [3]  swing_time_ms         — uint16  [0, 1000]
 *   [4]  double_support_ms     — uint16  [0, 600]
 *   [5]  accel_x_mg           — int16   [-4000, 4000]   (lateral)
 *   [6]  accel_y_mg           — int16   [-4000, 4000]   (ant-post)
 *   [7]  accel_z_mg           — int16   [-4000, 4000]   (vertical)
 *   [8]  gyro_x_mdps          — int16   [-35000, 35000]
 *   [9]  gyro_y_mdps          — int16   [-35000, 35000]
 *   [10] gyro_z_mdps          — int16   [-35000, 35000]
 *   [11] foot_pressure_kpa    — float   [0, 1000]
 *   [12] step_symmetry_pct    — float   [0, 100]
 *   [13] velocity_mm_s        — float   [0, 5000]
 *   [14] step_width_mm        — uint16  [0, 500]
 *   [15] step_count           — uint32  [0, 1e6]
 *
 * Simulated scenario: 10 strides of a healthy adult walking at ~100 steps/min,
 * with mild stride-to-stride variability (heel-strike micro-perturbations).
 * ========================================================================= */
static const GaitSample g_test_sequence[10] = {
    /* t=0  Normal comfortable walking, first stride */
    { 1380, 99.5f,  618, 382, 118,   48,  115,  812,   510,  7900,  195,  648.0f,  96.2f, 1380.0f, 102, 1 },
    /* t=1  Slight cadence increase, narrower step */
    { 1355, 101.0f, 610, 390, 122,   62,  130,  795,   480,  8100,  210,  655.0f,  95.8f, 1355.0f,  98, 2 },
    /* t=2  Longer stride, more vertical accel (heel-strike) */
    { 1415, 100.0f, 622, 378, 116,   35,  108,  870,   530,  7750,  180,  638.0f,  96.5f, 1415.0f, 105, 3 },
    /* t=3  Asymmetric stride — higher gyro y, lower symmetry */
    { 1340, 100.5f, 630, 370, 128,   80,  140,  790,   460,  8400,  230,  670.0f,  94.1f, 1340.0f, 110, 4 },
    /* t=4  Recovery stride — symmetry restored, moderate cadence */
    { 1375, 100.2f, 615, 385, 120,   52,  118,  808,   500,  7980,  200,  650.0f,  96.0f, 1375.0f, 103, 5 },
    /* t=5  Slight velocity increase — preparing for faster walk */
    { 1430, 103.0f, 605, 395, 114,   40,  105,  845,   550,  8300,  185,  635.0f,  96.8f, 1430.0f, 100, 6 },
    /* t=6  Faster walk — shorter stance, longer swing, higher cadence */
    { 1490, 108.0f, 590, 410, 108,   30,   95,  880,   600,  9200,  170,  620.0f,  97.1f, 1490.0f,  96, 7 },
    /* t=7  Peak speed — narrowest double support, highest velocity */
    { 1530, 112.0f, 578, 422, 102,   18,   82,  910,   650,  9600,  155,  608.0f,  97.5f, 1530.0f,  93, 8 },
    /* t=8  Deceleration — cadence drops, symmetry dips slightly */
    { 1410, 104.0f, 620, 380, 120,   58,  122,  820,   510,  8050,  205,  652.0f,  95.5f, 1410.0f, 104, 9 },
    /* t=9  Return to comfortable pace — values close to t=0 */
    { 1385, 100.0f, 616, 384, 119,   50,  116,  815,   505,  7950,  198,  646.0f,  96.3f, 1385.0f, 102, 10},
};

#define N_TIMESTEPS  ((int)(sizeof(g_test_sequence) / sizeof(g_test_sequence[0])))

/* =========================================================================
 * Model state (static — avoids large stack frames on the ESP32 call stack)
 * ========================================================================= */
static MambaS6Params    g_params;
static MambaS6State     g_state;
static MambaSelectWeights g_sel_weights;
static GaitFeatureMap   g_feat_map;

/* =========================================================================
 * Internal print utilities
 * ========================================================================= */

/**
 * @brief Print a float row with label, up to `len` elements.
 *        Rows longer than 8 are split onto a continuation line.
 */
static void print_float_row(const char *label, const float *v, int len)
{
    MAMBA_PRINT("  %-10s [", label);
    for (int i = 0; i < len; ++i) {
        MAMBA_PRINT("%7.4f", v[i]);
        if (i < len - 1) MAMBA_PRINT(",");
        if ((i + 1) % 8 == 0 && i < len - 1) MAMBA_PRINT("\n             ");
    }
    MAMBA_PRINT(" ]\n");
}

/**
 * @brief Print the L2-norm of a float vector.
 */
static float vec_norm(const float *v, int len)
{
    float s = 0.0f;
    for (int i = 0; i < len; ++i) s += v[i] * v[i];
    return sqrtf(s);
}

/* =========================================================================
 * Model lifecycle
 * ========================================================================= */

static void model_init(void)
{
    mamba_s6_params_init_default(&g_params);
    mamba_s6_state_reset(&g_state);
    mamba_select_weights_use_default(&g_sel_weights);
    mamba_gait_feature_map_default(&g_feat_map);
}

static void model_state_reset(void)
{
    mamba_s6_state_reset(&g_state);
}

/* =========================================================================
 * Single inference step (all three stages)
 *
 * Returns 0 on success, non-zero if any output is non-finite.
 * ========================================================================= */
static int run_one_step(const GaitSample     *sample,
                         MambaMinMaxScaler    *scaler,
                         float                 x_out[MAMBA_D],
                         MambaSelectOutput    *sel_out,
                         float                 y_out[MAMBA_D])
{
    /* --- Stage 1: Serialization & static normalization ------------------- */
    mamba_gait_extract_normalize(&g_feat_map, sample, x_out);

    /* --- Stage 2: Selection mechanism ------------------------------------ */
    mamba_select_compute(&g_sel_weights, x_out, sel_out);

    /* --- Stage 3: SSM recurrent step ------------------------------------- */
    mamba_s6_step_selective(&g_params, &g_state, x_out, sel_out, y_out);

    /* Sanity check: all outputs must be finite. */
    for (int d = 0; d < MAMBA_D; ++d) {
        if (!isfinite(y_out[d])) return -1;
    }
    for (int d = 0; d < MAMBA_D; ++d) {
        if (sel_out->delta[d] <= 0.0f) return -2;   /* Δ must be positive */
    }
    return 0;
}

/* =========================================================================
 * INFERENCE MODE
 *
 * Runs the 10-step sequence once and prints detailed per-step diagnostics
 * to the Serial console / UART.
 * ========================================================================= */
#if (MAMBA_TEST_MODE == MAMBA_TEST_MODE_INFERENCE)

static void run_inference_mode(void)
{
    MambaMinMaxScaler scaler;
    mamba_minmax_scaler_reset(&scaler);   /* unused in static-norm path */

    float           x[MAMBA_D];
    MambaSelectOutput sel;
    float           y[MAMBA_D];

    MAMBA_PRINT("\n");
    MAMBA_PRINT("╔══════════════════════════════════════════════════════════╗\n");
    MAMBA_PRINT("║     Mamba Gait Inference Engine — Step-by-Step Log      ║\n");
    MAMBA_PRINT("╠══════════════════════════════════════════════════════════╣\n");
    MAMBA_PRINT("║  MAMBA_D = %-3d   MAMBA_N = %-3d   Timesteps = %-3d        ║\n",
                MAMBA_D, MAMBA_N, N_TIMESTEPS);
    MAMBA_PRINT("╚══════════════════════════════════════════════════════════╝\n\n");

    int total_pass = 0, total_fail = 0;

    for (int t = 0; t < N_TIMESTEPS; ++t) {
        int64_t t_start = mamba_timer_now_us();
        int ret = run_one_step(&g_test_sequence[t], &scaler, x, &sel, y);
        int64_t t_end   = mamba_timer_now_us();
        int64_t step_us = t_end - t_start;

        MAMBA_PRINT("┌─── Timestep t = %d  [%s]  (%lld µs / %.3f ms) ─────────────\n",
                    t,
                    ret == 0 ? " OK " : "FAIL",
                    (long long)step_us,
                    (float)step_us / 1000.0f);

        /* Input vector x_t */
        MAMBA_PRINT("│ x_t  (normalised input, all channels ∈ [0,1]):\n");
        MAMBA_PRINT("│"); print_float_row("", x, MAMBA_D);

        /* Selective parameters — delta */
        MAMBA_PRINT("│ Δ_t  (per-channel timescale, all > 0):\n");
        MAMBA_PRINT("│"); print_float_row("", sel.delta, MAMBA_D);

        /* Selective parameters — B_t and C_t (shared, first 8 shown) */
        MAMBA_PRINT("│ B_t  (shared input projection, N=%d, first 8):\n", MAMBA_N);
        MAMBA_PRINT("│"); print_float_row("", sel.B, (MAMBA_N < 8 ? MAMBA_N : 8));

        MAMBA_PRINT("│ C_t  (shared output projection, N=%d, first 8):\n", MAMBA_N);
        MAMBA_PRINT("│"); print_float_row("", sel.C, (MAMBA_N < 8 ? MAMBA_N : 8));

        /* Output vector y_t */
        MAMBA_PRINT("│ y_t  (SSM output):\n");
        MAMBA_PRINT("│"); print_float_row("", y, MAMBA_D);
        MAMBA_PRINT("│   ||y_t|| = %.6f\n", vec_norm(y, MAMBA_D));

        /* Hidden state h_t — channel 0, all N states */
        MAMBA_PRINT("│ h_t[ch=0]  (hidden state, channel 0, all N=%d elements):\n", MAMBA_N);
        MAMBA_PRINT("│"); print_float_row("", g_state.h[0], MAMBA_N);
        MAMBA_PRINT("│   ||h[0]|| = %.6f\n", vec_norm(g_state.h[0], MAMBA_N));

        /* Assert summary */
        if (ret == 0) {
            MAMBA_PRINT("│ ✓ Assertions: outputs finite, all Δ > 0\n");
            ++total_pass;
        } else {
            MAMBA_PRINT("│ ✗ ASSERTION FAILED — code %d\n", ret);
            ++total_fail;
        }

        MAMBA_PRINT("└──────────────────────────────────────────────────────────\n\n");
    }

    MAMBA_PRINT("══════════════════════════════════════════════════════════\n");
    MAMBA_PRINT("  Final hidden state norms per channel:\n");
    for (int d = 0; d < MAMBA_D; ++d) {
        MAMBA_PRINT("    h[ch=%2d] ||h|| = %.6f\n", d, vec_norm(g_state.h[d], MAMBA_N));
    }
    MAMBA_PRINT("\n");
    MAMBA_PRINT("  Summary: %d / %d steps PASSED\n", total_pass, N_TIMESTEPS);
    MAMBA_PRINT("══════════════════════════════════════════════════════════\n");
}

#endif /* MAMBA_TEST_MODE_INFERENCE */

/* =========================================================================
 * DRY RUN MODE
 *
 * Repeats the full 10-step sequence DRY_RUN_REPS times.  Per-step Serial
 * output is suppressed so UART latency does not contaminate the timing.
 *
 * Reports:
 *   • Full-step timing: min / max / mean / std-dev (µs and ms)
 *   • Sub-stage breakdown: serialization vs. selection vs. SSM step
 * ========================================================================= */
#if (MAMBA_TEST_MODE == MAMBA_TEST_MODE_DRY_RUN)

/* Welford online algorithm state for streaming mean + variance. */
typedef struct {
    int64_t count;
    double  mean;
    double  M2;           /* sum of squared deviations from mean */
    int64_t min_val;
    int64_t max_val;
} Welford;

static void welford_init(Welford *w)
{
    w->count   = 0;
    w->mean    = 0.0;
    w->M2      = 0.0;
    w->min_val = INT64_MAX;
    w->max_val = INT64_MIN;
}

static void welford_update(Welford *w, int64_t x)
{
    if (x < w->min_val) w->min_val = x;
    if (x > w->max_val) w->max_val = x;
    w->count++;
    double delta  = (double)x - w->mean;
    w->mean      += delta / (double)w->count;
    double delta2 = (double)x - w->mean;
    w->M2        += delta * delta2;
}

static double welford_stddev(const Welford *w)
{
    if (w->count < 2) return 0.0;
    return sqrt(w->M2 / (double)(w->count - 1));
}

static void run_dry_run_mode(void)
{
    Welford  w_total, w_s1, w_s2, w_s3;
    welford_init(&w_total);
    welford_init(&w_s1);
    welford_init(&w_s2);
    welford_init(&w_s3);

    float            x[MAMBA_D];
    MambaSelectOutput sel;
    float            y[MAMBA_D];
    MambaMinMaxScaler scaler;
    mamba_minmax_scaler_reset(&scaler);

    MAMBA_PRINT("\n");
    MAMBA_PRINT("╔══════════════════════════════════════════════════════════╗\n");
    MAMBA_PRINT("║          Mamba Dry-Run Timing Benchmark                 ║\n");
    MAMBA_PRINT("╠══════════════════════════════════════════════════════════╣\n");
    MAMBA_PRINT("║  MAMBA_D = %-3d  MAMBA_N = %-3d                           ║\n",
                MAMBA_D, MAMBA_N);
    MAMBA_PRINT("║  Repetitions: %-5d  Total steps: %-6d               ║\n",
                DRY_RUN_REPS, DRY_RUN_TOTAL_STEPS);
    MAMBA_PRINT("╠══════════════════════════════════════════════════════════╣\n");
    MAMBA_PRINT("║  Sub-stages timed separately:                           ║\n");
    MAMBA_PRINT("║    S1  Serialization + static Min-Max normalize         ║\n");
    MAMBA_PRINT("║    S2  Selection mechanism  (3 × GEMV + softplus)       ║\n");
    MAMBA_PRINT("║    S3  SSM recurrence  (ZOH + state update + output)    ║\n");
    MAMBA_PRINT("╚══════════════════════════════════════════════════════════╝\n\n");
    MAMBA_PRINT("  Running... (output suppressed during benchmark)\n\n");

    for (int rep = 0; rep < DRY_RUN_REPS; ++rep) {
        /* Reset state for each new sequence so each rep is independent. */
        model_state_reset();

        for (int t = 0; t < N_TIMESTEPS; ++t) {
            const GaitSample *s = &g_test_sequence[t];

            /* ---- Stage 1: Serialization ------------------------------ */
            int64_t t0 = mamba_timer_now_us();
            mamba_gait_extract_normalize(&g_feat_map, s, x);
            int64_t t1 = mamba_timer_now_us();

            /* ---- Stage 2: Selection ---------------------------------- */
            mamba_select_compute(&g_sel_weights, x, &sel);
            int64_t t2 = mamba_timer_now_us();

            /* ---- Stage 3: SSM step ----------------------------------- */
            mamba_s6_step_selective(&g_params, &g_state, x, &sel, y);
            int64_t t3 = mamba_timer_now_us();

            int64_t us_s1    = t1 - t0;
            int64_t us_s2    = t2 - t1;
            int64_t us_s3    = t3 - t2;
            int64_t us_total = t3 - t0;

            welford_update(&w_s1,    us_s1);
            welford_update(&w_s2,    us_s2);
            welford_update(&w_s3,    us_s3);
            welford_update(&w_total, us_total);

            /* Volatile sink: prevent the compiler from eliminating the  */
            /* computation as dead code during optimisation.              */
            volatile float sink = y[0] + y[MAMBA_D - 1];
            (void)sink;
        }
    }

    /* ---- Report ------------------------------------------------------ */
    MAMBA_PRINT("╔══════════════════════════════════════════════════════════╗\n");
    MAMBA_PRINT("║                  Timing Results                         ║\n");
    MAMBA_PRINT("╠═══════════╦══════════╦══════════╦══════════╦════════════╣\n");
    MAMBA_PRINT("║ Stage     ║ Min (µs) ║ Max (µs) ║ Mean(µs) ║ StdDev(µs) ║\n");
    MAMBA_PRINT("╠═══════════╬══════════╬══════════╬══════════╬════════════╣\n");

    /* Sub-stage rows */
    const char *labels[] = { "S1-Serial ", "S2-Select ", "S3-SSM    ", "TOTAL     " };
    const Welford *stats[] = { &w_s1, &w_s2, &w_s3, &w_total };

    for (int i = 0; i < 4; ++i) {
        const Welford *w = stats[i];
        if (i == 3) {
            MAMBA_PRINT("╠═══════════╬══════════╬══════════╬══════════╬════════════╣\n");
        }
        MAMBA_PRINT("║ %-9s ║ %8lld ║ %8lld ║ %8.1f ║ %10.1f ║\n",
                    labels[i],
                    (long long)w->min_val,
                    (long long)w->max_val,
                    w->mean,
                    welford_stddev(w));
    }

    MAMBA_PRINT("╠═══════════╩══════════╩══════════╩══════════╩════════════╣\n");
    MAMBA_PRINT("║  Mean per-step:  %.3f µs  =  %.4f ms                  ║\n",
                w_total.mean, w_total.mean / 1000.0);
    MAMBA_PRINT("║  Estimated max throughput:  %.1f steps/sec              ║\n",
                1e6 / w_total.mean);
    MAMBA_PRINT("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Stage breakdown as percentage of total */
    MAMBA_PRINT("  Stage breakdown (% of mean total):\n");
    MAMBA_PRINT("    S1 Serialize:  %5.1f%%\n", 100.0 * w_s1.mean    / w_total.mean);
    MAMBA_PRINT("    S2 Select:     %5.1f%%\n", 100.0 * w_s2.mean    / w_total.mean);
    MAMBA_PRINT("    S3 SSM step:   %5.1f%%\n", 100.0 * w_s3.mean    / w_total.mean);
    MAMBA_PRINT("\n");
}

#endif /* MAMBA_TEST_MODE_DRY_RUN */

/* =========================================================================
 * Entry point
 *
 * On ESP-IDF: rename to app_main().
 * On Arduino: call from setup() after Serial.begin(115200).
 * On host:    compile and run directly.
 * ========================================================================= */
void mamba_test_main(void)
{
    /* ---- Initialise all model components -------------------------------- */
    model_init();

    MAMBA_PRINT("\n[Mamba ESP32] Model initialised.\n");
    MAMBA_PRINT("  MAMBA_D=%d  MAMBA_N=%d  state_size=%d floats (%d bytes)\n",
                MAMBA_D, MAMBA_N, MAMBA_STATE_SIZE,
                (int)(MAMBA_STATE_SIZE * sizeof(float)));

#if (MAMBA_TEST_MODE == MAMBA_TEST_MODE_INFERENCE)
    MAMBA_PRINT("  Mode: INFERENCE (detailed per-step output)\n");
    run_inference_mode();

#elif (MAMBA_TEST_MODE == MAMBA_TEST_MODE_DRY_RUN)
    MAMBA_PRINT("  Mode: DRY RUN (timing benchmark, %d reps × 10 steps)\n",
                DRY_RUN_REPS);
    run_dry_run_mode();

#else
    #error "Unknown MAMBA_TEST_MODE. Use MAMBA_TEST_MODE_INFERENCE or MAMBA_TEST_MODE_DRY_RUN."
#endif
}

/* =========================================================================
 * Platform entry points
 * ========================================================================= */

#ifdef ARDUINO
/* ---- Arduino-ESP32 ------------------------------------------------------ */
void setup(void)
{
    Serial.begin(115200);
    while (!Serial) { delay(10); }    /* Wait for USB-Serial to connect     */
    delay(500);
    mamba_test_main();
}

void loop(void)
{
    /* Nothing — the test runs once in setup(). */
    delay(10000);
}

#elif defined(IDF_VER) || defined(ESP_PLATFORM)
/* ---- ESP-IDF ------------------------------------------------------------ */
void app_main(void)
{
    mamba_test_main();
}

#else
/* ---- Host / POSIX ------------------------------------------------------- */
int main(void)
{
    mamba_test_main();
    return 0;
}
#endif
