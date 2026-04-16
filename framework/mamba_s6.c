/**
 * @file mamba_s6.c
 * @brief Implementation of a single Mamba (S6) block — recurrent view.
 *
 * See mamba_s6.h for full documentation, notation, and design rationale.
 *
 * Compilation notes for ESP32 (ESP-IDF / Arduino-ESP32):
 *   - The Xtensa LX7 (ESP32-S3) and LX6 (ESP32) both have a hardware FPU
 *     for single-precision floats; ensure CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ
 *     and floating-point ABI flags are set correctly in sdkconfig.
 *   - Build with -O2 or -O3 and -ffast-math for best throughput.
 *   - Link against libm (standard in ESP-IDF).
 */

#include "mamba_s6.h"
#include "mamba_select.h"   /* MambaSelectOutput — Step 2 */

#include <math.h>    /* expf, logf */
#include <string.h>  /* memset    */

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/**
 * @brief Numerically stable softplus: softplus(x) = log(1 + exp(x)).
 *
 * Clamped to avoid overflow for large positive x:
 *   if x > 20  → returns x (exp(x) dominates)
 *   otherwise  → returns log1p(expf(x))
 *
 * The threshold of 20 keeps the error < 2e-9 relative to the true value
 * while staying well within float32 range.
 */
static inline float softplus(float x)
{
    if (x > 20.0f) return x;
    /* log1p(y) = log(1 + y), more accurate than logf(1.0f + expf(x))
     * when expf(x) is small.                                           */
    return log1pf(expf(x));
}

/**
 * @brief Safe division: (numerator) / (denominator), returning fallback
 *        when |denominator| < epsilon.
 *
 * Used in ZOH to handle the degenerate case A_n → 0.
 */
static inline float safe_div(float num, float den, float fallback)
{
    if (fabsf(den) < 1e-8f) return fallback;
    return num / den;
}

/* =========================================================================
 * Public API — implementation
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * mamba_s6_state_reset
 * ---------------------------------------------------------------------- */
void mamba_s6_state_reset(MambaS6State *state)
{
    memset(state->h, 0, sizeof(state->h));
}

/* -------------------------------------------------------------------------
 * mamba_s6_params_init_default
 * ---------------------------------------------------------------------- */
void mamba_s6_params_init_default(MambaS6Params *params)
{
    /* Copy A and D_skip from the trained/mock weight constants in Flash */
    memcpy(params->A, MAMBA_A, sizeof(params->A));
    memcpy(params->D_skip, MAMBA_D_SKIP, sizeof(params->D_skip));

    /* Initialize deprecated per-channel projections to neutral values.
     * Note: mamba_s6_step_selective() ignores these in favor of
     * the dense matrices in mamba_select.c. */
    const float inv_N = 1.0f / (float)MAMBA_N;
    for (int d = 0; d < MAMBA_D; ++d) {
        params->W_delta[d] = 1.0f;
        for (int n = 0; n < MAMBA_N; ++n) {
            params->W_B[d][n] = inv_N;
            params->W_C[d][n] = inv_N;
        }
    }
}

/* -------------------------------------------------------------------------
 * mamba_s6_zoh_discretize
 *
 * Exact ZOH formula for a diagonal continuous-time system:
 *
 *   Ā_n  = exp( Δ · A_n )
 *
 *   B̄_n  = ( Ā_n − 1 ) / A_n  ·  B_n
 *
 * Derivation (scalar channel, diagonal A):
 *   The matrix exponential exp(Δ A) collapses to a scalar exp(Δ A_n) for
 *   each diagonal entry.  The ZOH input gain is then the integral:
 *
 *     B̄_n = ∫₀^Δ  exp(τ A_n) dτ  · B_n
 *           = [ exp(Δ A_n) − 1 ] / A_n  ·  B_n
 *
 *   When A_n → 0 the limit gives B̄_n = Δ · B_n (Euler / first-order ZOH).
 *   safe_div() handles this with the fallback = Δ * B_n.
 * ---------------------------------------------------------------------- */
void mamba_s6_zoh_discretize(float        delta,
                              const float *A,
                              const float *B,
                              float       *A_bar,
                              float       *B_bar,
                              int          N)
{
    for (int n = 0; n < N; ++n) {
        /* Ā_n = exp(Δ · A_n)
         * Because A_n < 0 this is always in (0, 1], giving stable decay. */
        const float eDA = expf(delta * A[n]);
        A_bar[n] = eDA;

        /* B̄_n = (Ā_n − 1) / A_n · B_n
         * Fallback for A_n ≈ 0: limit is Δ · B_n  (l'Hôpital / Taylor). */
        const float zoh_gain = safe_div(eDA - 1.0f, A[n], delta);
        B_bar[n] = zoh_gain * B[n];
    }
}

/* -------------------------------------------------------------------------
 * mamba_s6_step_precomputed
 *
 * Core recurrent update.  Caller supplies pre-projected Δ, B, C so this
 * function focuses purely on:
 *   1. ZOH discretisation  →  Ā, B̄
 *   2. State update         →  h_t  =  Ā ⊙ h_{t-1}  +  B̄ · x_d
 *   3. Output computation   →  y_d  =  C · h_t  +  D_skip · x_d
 *
 * All inner loops are over N (state dimension), which the compiler can
 * unroll when MAMBA_N is a power-of-two.
 * ---------------------------------------------------------------------- */
void mamba_s6_step_precomputed(const MambaS6Params *params,
                                MambaS6State        *state,
                                const float          x[MAMBA_D],
                                const float          delta[MAMBA_D],
                                const float          B_in[MAMBA_D][MAMBA_N],
                                const float          C_in[MAMBA_D][MAMBA_N],
                                float                y_out[MAMBA_D])
{
    /* Temporary buffers for discretised parameters.
     * Allocated on the stack; size is MAMBA_N floats each (~64 bytes for N=16).
     * If MAMBA_N is very large (>128) consider moving to static locals with
     * a mutex, or heap-allocating in an init function.                    */
    float A_bar[MAMBA_N];
    float B_bar[MAMBA_N];

    for (int d = 0; d < MAMBA_D; ++d) {

        /* ---- Step 1: ZOH discretisation -------------------------------- */
        mamba_s6_zoh_discretize(
            delta[d],
            params->A[d],   /* const float* — pointer to A[d][0]  */
            B_in[d],        /* const float* — pointer to B_in[d][0] */
            A_bar,
            B_bar,
            MAMBA_N
        );

        /* ---- Step 2: Hidden-state update --------------------------------
         *   h_t[n] = Ā_n · h_{t-1}[n]  +  B̄_n · x_d
         *
         * Note: in S6 the input B is already a vector (projection of x),
         * so the contribution is simply B̄_n * x_d (scalar multiply).    */
        const float x_d = x[d];

        for (int n = 0; n < MAMBA_N; ++n) {
            state->h[d][n] = A_bar[n] * state->h[d][n]
                           + B_bar[n] * x_d;
        }

        /* ---- Step 3: Output projection ----------------------------------
         *   y_d = Σ_n  C_in[d][n] · h_t[d][n]   +   D_skip[d] · x_d    */
        float y_d = params->D_skip[d] * x_d;

        for (int n = 0; n < MAMBA_N; ++n) {
            y_d += C_in[d][n] * state->h[d][n];
        }

        y_out[d] = y_d;
    }
}

/* -------------------------------------------------------------------------
 * mamba_s6_step
 *
 * Convenience wrapper that applies the linear projections to obtain Δ, B, C
 * from the raw input x before calling mamba_s6_step_precomputed().
 *
 * Projection rules (bias-free, weight-per-channel):
 *   delta[d]   = softplus( W_delta[d] * x[d] )
 *   B_proj[d][n] = W_B[d][n]  * x[d]
 *   C_proj[d][n] = W_C[d][n]  * x[d]
 *
 * In a production Mamba implementation W_delta, W_B, W_C would be dense
 * matrices operating on the full model dimension.  The scalar-per-channel
 * version here keeps the arithmetic count at O(D·N) per step, which is
 * well-suited to resource-constrained MCUs.
 * ---------------------------------------------------------------------- */
void mamba_s6_step(const MambaS6Params *params,
                   MambaS6State        *state,
                   const float          x[MAMBA_D],
                   float                y_out[MAMBA_D])
{
    /* Projected selective parameters — stack-allocated. */
    float delta[MAMBA_D];
    float B_proj[MAMBA_D][MAMBA_N];
    float C_proj[MAMBA_D][MAMBA_N];

    for (int d = 0; d < MAMBA_D; ++d) {
        const float x_d = x[d];

        /* Δ: softplus ensures positivity (required for stable ZOH). */
        delta[d] = softplus(params->W_delta[d] * x_d);

        for (int n = 0; n < MAMBA_N; ++n) {
            B_proj[d][n] = params->W_B[d][n] * x_d;
            C_proj[d][n] = params->W_C[d][n] * x_d;
        }
    }

    mamba_s6_step_precomputed(params, state, x,
                               delta,
                               (const float(*)[MAMBA_N]) B_proj,
                               (const float(*)[MAMBA_N]) C_proj,
                               y_out);
}

/* -------------------------------------------------------------------------
 * mamba_s6_step_selective   (Step 2 - recommended path)
 *
 * Implements the architecturally-correct Mamba S6 recurrent step where:
 *   - B_t in R^N  is SHARED across all D channels
 *   - C_t in R^N  is SHARED across all D channels
 *   - Delta_t in R^D  is per-channel (one timescale per SSM channel)
 *
 * The shared B is broadcast through ZOH using each channel's own Delta[d]:
 *
 *   A_bar[d][n]  = exp( Delta[d] * A[d][n] )
 *   B_bar[d][n]  = ( A_bar[d][n] - 1 ) / A[d][n]  *  B_t[n]
 *
 * State update (per channel d, per state n):
 *   h_t[d][n] = A_bar[d][n] * h_{t-1}[d][n]  +  B_bar[d][n] * x_d
 *
 * Output (per channel d):
 *   y[d] = sum_n  C_t[n] * h_t[d][n]  +  D_skip[d] * x_d
 *
 * Stack usage: 2 x MAMBA_N floats (A_bar + B_bar) -- ~128 B for N=16.
 * No per-channel [D][N] temporary is needed because B and C are shared.
 * ---------------------------------------------------------------------- */
void mamba_s6_step_selective(const MambaS6Params     *params,
                              MambaS6State            *state,
                              const float              x[MAMBA_D],
                              const MambaSelectOutput *sel,
                              float                    y_out[MAMBA_D])
{
    float A_bar[MAMBA_N];
    float B_bar[MAMBA_N];

    for (int d = 0; d < MAMBA_D; ++d) {

        /* ---- Step 1: ZOH -- uses shared B_t but per-channel Delta[d] --- */
        mamba_s6_zoh_discretize(
            sel->delta[d],
            params->A[d],    /* continuous A diagonal for channel d   */
            sel->B,          /* shared B_t in R^N                     */
            A_bar,
            B_bar,
            MAMBA_N
        );

        /* ---- Step 2: Hidden-state update -------------------------------- */
        const float x_d = x[d];

        for (int n = 0; n < MAMBA_N; ++n) {
            state->h[d][n] = A_bar[n] * state->h[d][n]
                           + B_bar[n] * x_d;
        }

        /* ---- Step 3: Output -- shared C_t dotted with h_t[d] ----------- */
        float y_d = params->D_skip[d] * x_d;

        for (int n = 0; n < MAMBA_N; ++n) {
            y_d += sel->C[n] * state->h[d][n];
        }

        y_out[d] = y_d;
    }
}
