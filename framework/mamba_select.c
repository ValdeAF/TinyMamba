/**
 * @file mamba_select.c
 * @brief Mamba S6 selection mechanism — dense linear projections.
 *
 * See mamba_select.h for full documentation and design rationale.
 *
 * Implementation notes
 * --------------------
 * 1. All three projections (W_delta, W_B, W_C) are standard GEMV operations
 *    (matrix × vector).  Each reads weight rows from Flash via the cache and
 *    accumulates into a float register — exactly the pattern the Xtensa LX6/LX7
 *    FPU is designed for.
 *
 * 2. The inner loop iterates over the input dimension D (columns of each
 *    weight matrix).  When -DMAMBA_D is a power of two and -O2/-O3 is used,
 *    the compiler will unroll and pipeline these loops automatically.
 *
 * 3. softplus() is applied only to the D scalars of delta_raw, not to B or C.
 *    It is implemented as log1p(exp(x)) with a clamp at x > 20 to avoid
 *    float overflow (exp(20) ≈ 4.85e8, safely within float32 range to log,
 *    but we skip the log for clarity since log(1+4.85e8) ≈ x anyway).
 *
 * 4. No heap allocation: all temporaries (delta_raw) are on the stack.
 *    The only stack frame cost per call is MAMBA_D floats (~64 B for D=16).
 */

#include "mamba_select.h"

#include <math.h>       /* expf, log1pf */
#ifdef __esp32__
#include "esp_dsp.h"    /* dsps_dotprod_f32 — ESP-DSP acceleration */
#endif

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/**
 * @brief Numerically stable softplus: f(x) = log(1 + exp(x)).
 *
 * For x > 20 the exponential dominates and f(x) ≈ x, so we return x
 * directly to avoid a needless (and potentially lossy) exp/log round-trip.
 *
 * Expected Δ range after softplus given typical pre-activations:
 *   x = -2 → Δ ≈ 0.127   (fast decay, ~8 steps memory)
 *   x =  0 → Δ ≈ 0.693   (moderate, ~1.4 steps memory)
 *   x =  2 → Δ ≈ 2.127   (slow decay, almost integration)
 */
static inline float softplus(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* =========================================================================
 * Public API — implementation
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * mamba_select_weights_use_default
 * ---------------------------------------------------------------------- */
void mamba_select_weights_use_default(MambaSelectWeights *w)
{
    w->W_delta = MAMBA_W_DELTA;
    w->b_delta = MAMBA_B_DELTA;
    w->W_B     = MAMBA_W_B;
    w->W_C     = MAMBA_W_C;

    w->scale_W_delta = MAMBA_SCALE_W_DELTA;
    w->scale_W_B     = MAMBA_SCALE_W_B;
    w->scale_W_C     = MAMBA_SCALE_W_C;
}

/* -------------------------------------------------------------------------
 * mamba_select_compute
 *
 * Three independent GEMV operations on Flash-resident weight matrices,
 * followed by softplus on the delta output.
 *
 * Memory access pattern:
 *   Each weight row (length D) is read sequentially — exactly the access
 *   pattern that maximises burst-read efficiency from the SPI Flash cache.
 * ---------------------------------------------------------------------- */
void mamba_select_compute(const MambaSelectWeights *w,
                          const float               u[MAMBA_D],
                          MambaSelectOutput        *out)
{
    /* Dynamic quantization of the input u into INT8 */
    int8_t u_q[MAMBA_D];
    float u_max = 0.0f;
    for (int d = 0; d < MAMBA_D; ++d) {
        float abs_u = fabsf(u[d]);
        if (abs_u > u_max) {
            u_max = abs_u;
        }
    }
    
    float u_scale = 0.0f;
    float u_scale_inv = 0.0f;
    if (u_max > 0.0f) {
        u_scale = u_max / 127.0f;
        u_scale_inv = 127.0f / u_max;
    }
    
    for (int d = 0; d < MAMBA_D; ++d) {
        float val = u[d] * u_scale_inv;
        int val_i = (int)roundf(val);
        if (val_i > 127) val_i = 127;
        if (val_i < -127) val_i = -127;
        u_q[d] = (int8_t)val_i;
    }

    /* ------------------------------------------------------------------
     * 1. Delta projection:  W_delta[D][D] @ u_q[D]  +  b_delta[D]
     *    INT8 dot product loop, scaled back to float, then bias and softplus.
     * ------------------------------------------------------------------ */
    float delta_mult = w->scale_W_delta * u_scale;
    for (int d = 0; d < MAMBA_D; ++d) {
        int32_t acc_q = 0;
        for (int k = 0; k < MAMBA_D; ++k) {
            acc_q += (int32_t)w->W_delta[d][k] * u_q[k];
        }
        float acc = w->b_delta[d] + (float)acc_q * delta_mult;
        out->delta[d] = softplus(acc);
    }

    /* ------------------------------------------------------------------
     * 2. B projection:  W_B[N][D] @ u_q[D]  →  B[N]
     * ------------------------------------------------------------------ */
    float B_mult = w->scale_W_B * u_scale;
    for (int n = 0; n < MAMBA_N; ++n) {
        int32_t acc_q = 0;
        for (int k = 0; k < MAMBA_D; ++k) {
            acc_q += (int32_t)w->W_B[n][k] * u_q[k];
        }
        out->B[n] = (float)acc_q * B_mult;
    }

    /* ------------------------------------------------------------------
     * 3. C projection:  W_C[N][D] @ u_q[D]  →  C[N]
     * ------------------------------------------------------------------ */
    float C_mult = w->scale_W_C * u_scale;
    for (int n = 0; n < MAMBA_N; ++n) {
        int32_t acc_q = 0;
        for (int k = 0; k < MAMBA_D; ++k) {
            acc_q += (int32_t)w->W_C[n][k] * u_q[k];
        }
        out->C[n] = (float)acc_q * C_mult;
    }
}
