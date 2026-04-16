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

#include <math.h>   /* expf, log1pf */

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
    /* ------------------------------------------------------------------
     * 1. Delta projection:  W_delta[D][D] @ u[D]  +  b_delta[D]
     *    then apply softplus element-wise.
     *
     * W_delta[d][:] is the d-th row of the weight matrix.
     * The dot product with u gives the raw (pre-activation) delta for
     * channel d.  Adding b_delta[d] (the bias) shifts the operating point.
     * ------------------------------------------------------------------ */
    for (int d = 0; d < MAMBA_D; ++d) {
        /* Read the bias from Flash — single scalar, always hot in cache. */
        float acc = w->b_delta[d];

        /* Sequential read of row d of W_delta from Flash (cache-friendly). */
        const float *row_d = w->W_delta[d];   /* pointer to W_delta[d][0] */
        for (int k = 0; k < MAMBA_D; ++k) {
            acc += row_d[k] * u[k];
        }

        /* softplus enforces Δ > 0, which is required for ZOH stability.   */
        out->delta[d] = softplus(acc);
    }

    /* ------------------------------------------------------------------
     * 2. B projection:  W_B[N][D] @ u[D]  →  B[N]
     *
     * B is SHARED across all D channels (see mamba_select.h for rationale).
     * Row n of W_B projects the full D-dim input to the n-th state element.
     * ------------------------------------------------------------------ */
    for (int n = 0; n < MAMBA_N; ++n) {
        float acc = 0.0f;
        const float *row_n = w->W_B[n];       /* pointer to W_B[n][0]     */
        for (int k = 0; k < MAMBA_D; ++k) {
            acc += row_n[k] * u[k];
        }
        out->B[n] = acc;
    }

    /* ------------------------------------------------------------------
     * 3. C projection:  W_C[N][D] @ u[D]  →  C[N]
     *
     * C is also SHARED across all D channels.
     * ------------------------------------------------------------------ */
    for (int n = 0; n < MAMBA_N; ++n) {
        float acc = 0.0f;
        const float *row_n = w->W_C[n];       /* pointer to W_C[n][0]     */
        for (int k = 0; k < MAMBA_D; ++k) {
            acc += row_n[k] * u[k];
        }
        out->C[n] = acc;
    }
}
