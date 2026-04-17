/**
 * @file mamba_s6.h
 * @brief Single Mamba (S6) block for inference on ESP32 (recurrent view).
 *
 * Architecture reference:
 *   Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
 *   https://arxiv.org/abs/2312.00752
 *
 * Design decisions:
 *   - Recurrent (step-by-step) view — no convolution, O(N) memory per channel.
 *   - Diagonal A (stored as a 1-D vector per channel, HiPPO-init style).
 *   - Zero-Order Hold (ZOH) discretisation for Ā and B̄.
 *   - All arithmetic uses `float` to exploit the ESP32's hardware FPU.
 *   - D (model dimension) and N (state dimension) are compile-time constants
 *     so the compiler can unroll / vectorise inner loops.
 *
 * Step 2 addition — Selection Mechanism:
 *   The proper S6 selection mechanism is now in mamba_select.h / mamba_select.c.
 *   Use mamba_s6_step_selective() for the fully-correct Mamba inference path.
 *   The older mamba_s6_step() remains for backward compatibility but its
 *   internal projections are simplified (scalar-per-channel, not dense matrix).
 *
 * Notation used throughout:
 *   D      — model (feature) dimension  (number of SSM channels)
 *   N      — state dimension per channel (latent state size)
 *   x_d    — scalar input for channel d  ∈ ℝ
 *   h[d]   — hidden state vector         ∈ ℝ^N
 *   Δ[d]   — timescale per channel, > 0 after softplus
 *   A[d]   — continuous-time diagonal    ∈ ℝ^N  (negative, per channel)
 *   B_t    — shared input vector         ∈ ℝ^N  (NOT per-channel in S6)
 *   C_t    — shared output vector        ∈ ℝ^N  (NOT per-channel in S6)
 *   D_skip — direct feedthrough scalar   ∈ ℝ    (residual / skip)
 *
 * ZOH discretisation (diagonal A, shared B):
 *   Ā[d][n]  = exp( Δ[d] · A[d][n] )
 *   B̄[d][n]  = (Ā[d][n] − 1) / A[d][n]  ·  B_t[n]   ← B_t shared, Δ[d] per-channel
 *   This file uses the EXACT formula.
 *
 * SSM recurrence (per channel d, per time-step):
 *   h_t[d][n] = Ā[d][n] · h_{t-1}[d][n]  +  B̄[d][n] · x_d
 *   y[d]      = Σ_n  C_t[n] · h_t[d][n]  +  D_skip[d] · x_d
 */

#ifndef MAMBA_S6_H
#define MAMBA_S6_H

#include <stdint.h>

/* =========================================================================
 * Compile-time configuration
 * =========================================================================
 * Override these from your build system or before including this header:
 *   -DMAMBA_D=16  -DMAMBA_N=16
 * ========================================================================= */

/** Model dimension: number of parallel SSM channels. */
#ifndef MAMBA_D
#define MAMBA_D  6
#endif

/** State dimension: size of the hidden state vector per channel. */
#ifndef MAMBA_N
#define MAMBA_N  64
#endif

/* =========================================================================
 * Derived constants (do not modify)
 * ========================================================================= */

/** Total hidden-state elements across all channels. */
#define MAMBA_STATE_SIZE   (MAMBA_D * MAMBA_N)

/* =========================================================================
 * Parameter structure
 *
 * All arrays are in ROW-MAJOR order: index [d][n] maps to [d * MAMBA_N + n].
 *
 * In a full Mamba model these weights are trained offline and loaded from
 * flash.  The selective (input-dependent) parameters Δ, B, C are computed
 * at runtime by linear projections of the input; their *projection weights*
 * are therefore stored here alongside the fixed parameters.
 * ========================================================================= */
typedef struct {
    /* -----------------------------------------------------------------
     * Fixed learned parameters (loaded from flash / DRAM once)
     * ----------------------------------------------------------------- */

    /**
     * @brief Continuous-time diagonal state-transition matrix A.
     * Shape: [D][N].  All entries must be negative (stability).
     * Typically initialised with the HiPPO spectrum:
     *   A[n] = -(n+1)  for n = 0 … N-1   (then repeated across D).
     */
    float A[MAMBA_D][MAMBA_N];

    /**
     * @brief Direct feedthrough (skip-connection) scale per channel.
     * Shape: [D].  Applied as:  y += D_skip[d] * x[d].
     */
    float D_skip[MAMBA_D];

    /**
     * @brief DPS 6x6 Readout Layer weights.
     * Shape: [D][D]. Multiplied against the inner SSM outputs.
     */
    float W_out[MAMBA_D][MAMBA_D];

    /**
     * @brief Output bias added after the Readout Layer computation.
     * Shape: [D].
     */
    float bias_out[MAMBA_D];

    /* -----------------------------------------------------------------
     * Simplified projection weights — used by mamba_s6_step() only.
     *
     * DEPRECATED in favour of MambaSelectWeights (mamba_select.h).
     * These are scalar-per-channel weights that do NOT implement the
     * full S6 dense projection.  They are retained so that mamba_s6_step()
     * remains compilable without pulling in mamba_select.h.
     *
     * For the architecturally-correct implementation use:
     *   mamba_select_compute()   — builds MambaSelectOutput from dense matrices
     *   mamba_s6_step_selective() — runs the SSM with shared B_t and C_t
     * ----------------------------------------------------------------- */

    /**
     * @brief [DEPRECATED] Scalar delta weight per channel.
     * Shape: [D].  delta[d] = softplus(W_delta[d] * x[d]).
     * Replaced by MambaSelectWeights.W_delta [D][D].
     */
    float W_delta[MAMBA_D];

    /**
     * @brief [DEPRECATED] Per-channel B weight.
     * Shape: [D][N].  B[d][n] = W_B[d][n] * x[d].
     * Replaced by MambaSelectWeights.W_B [N][D] (shared across channels).
     */
    float W_B[MAMBA_D][MAMBA_N];

    /**
     * @brief [DEPRECATED] Per-channel C weight.
     * Shape: [D][N].  C[d][n] = W_C[d][n] * x[d].
     * Replaced by MambaSelectWeights.W_C [N][D] (shared across channels).
     */
    float W_C[MAMBA_D][MAMBA_N];

} MambaS6Params;

/* =========================================================================
 * State structure
 *
 * Represents the recurrent hidden state h ∈ ℝ^{D × N}.
 * Must be zeroed before the first token (use mamba_s6_state_reset()).
 * ========================================================================= */
typedef struct {
    /** Hidden state h[d][n] stored row-major. */
    float h[MAMBA_D][MAMBA_N];
} MambaS6State;

/* Trained weights for the fixed parameters (usually stored in Flash) */
extern const float MAMBA_A[MAMBA_D][MAMBA_N];
extern const float MAMBA_D_SKIP[MAMBA_D];
extern const float MAMBA_W_OUT[MAMBA_D][MAMBA_D];
extern const float MAMBA_BIAS_OUT[MAMBA_D];

/* Forward declaration — breaks the circular dependency between mamba_s6.h
 * and mamba_select.h.  The full definition (and typedef) lives in mamba_select.h.
 * All .c files must include both headers to use the full type. */
struct MambaSelectOutput;
/* Typedef alias so callers can write MambaSelectOutput* instead of struct MambaSelectOutput* */
typedef struct MambaSelectOutput MambaSelectOutput;

/* =========================================================================
 * Public API
 * ========================================================================= */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Zero-initialise a MambaS6State (call before first token).
 * @param state  Pointer to the state to reset.
 */
void mamba_s6_state_reset(MambaS6State *state);

/**
 * @brief Initialise parameter struct with sensible HiPPO defaults.
 *
 * Sets:
 *   - A[d][n] = -(n + 1)   for all d, n   (HiPPO-LegS diagonal)
 *   - D_skip[d] = 1.0f     for all d
 *   - W_delta[d] = 1.0f    for all d
 *   - W_B[d][n]  = 1.0f / N for all d, n
 *   - W_C[d][n]  = 1.0f / N for all d, n
 *
 * In production, replace with values trained offline and stored in flash.
 *
 * @param params  Pointer to the parameter struct to initialise.
 */
void mamba_s6_params_init_default(MambaS6Params *params);

/**
 * @brief Run one recurrent step of the S6 block (input-projected path).
 *
 * Internally:
 *   1. Projects x to delta, B, C via the stored weight vectors.
 *   2. Applies ZOH discretisation to obtain Ā, B̄.
 *   3. Updates hidden state h.
 *   4. Computes output y.
 *
 * @param params  Pointer to (const) model parameters.
 * @param state   Pointer to mutable hidden state (updated in-place).
 * @param x       Input vector of length MAMBA_D (one scalar per channel).
 * @param y_out   Output vector of length MAMBA_D (written by this function).
 */
void mamba_s6_step(const MambaS6Params *params,
                   MambaS6State        *state,
                   const float          x[MAMBA_D],
                   float                y_out[MAMBA_D]);

/**
 * @brief Run one recurrent step with caller-supplied Δ, B, C (no projection).
 *
 * Use this variant when the projection has already been applied upstream
 * (e.g. by a dedicated Linear layer or a DSP pre-processing stage).
 *
 * @param params   Pointer to (const) model parameters (only A, D_skip used).
 * @param state    Pointer to mutable hidden state.
 * @param x        Input vector [D].
 * @param delta    Timescale vector [D], must be > 0 (softplus applied upstream).
 * @param B_in     Input matrix [D][N], row-major.
 * @param C_in     Output matrix [D][N], row-major.
 * @param y_out    Output vector [D].
 */
void mamba_s6_step_precomputed(const MambaS6Params *params,
                                MambaS6State        *state,
                                const float          x[MAMBA_D],
                                const float          delta[MAMBA_D],
                                const float          B_in[MAMBA_D][MAMBA_N],
                                const float          C_in[MAMBA_D][MAMBA_N],
                                float                y_out[MAMBA_D]);

/**
 * @brief Apply ZOH discretisation for a single channel.
 *
 * Computes:
 *   A_bar[n] = exp(delta * A[n])
 *   B_bar[n] = (A_bar[n] - 1.0f) / A[n] * B[n]
 *
 * This function is exposed so callers can cache discretised parameters
 * across successive time-steps when Δ is constant (non-selective mode).
 *
 * @param delta   Scalar timescale Δ > 0.
 * @param A       Continuous-time diagonal A, length N (negative values).
 * @param B       Continuous-time B vector, length N.
 * @param A_bar   Output: discretised Ā, length N (caller-allocated).
 * @param B_bar   Output: discretised B̄, length N (caller-allocated).
 * @param N       State dimension (pass MAMBA_N for the configured size).
 */
void mamba_s6_zoh_discretize(float        delta,
                              const float *A,
                              const float *B,
                              float       *A_bar,
                              float       *B_bar,
                              int          N);

/**
 * @brief Run one recurrent step using the full S6 selection mechanism output.
 *
 * This is the RECOMMENDED entry point after Step 2.  It consumes the output
 * of mamba_select_compute() which provides:
 *   - delta[D]  — per-channel timescales (dense projection + softplus)
 *   - B[N]      — shared input vector    (dense D→N projection)
 *   - C[N]      — shared output vector   (dense D→N projection)
 *
 * The key architectural difference from mamba_s6_step_precomputed() is that
 * B and C are SHARED across all D channels (ℝ^N, not ℝ^{D×N}).  During ZOH
 * discretisation B is broadcast per-channel using each channel's own Δ:
 *
 *   B̄[d][n] = (exp(Δ[d]·A[d][n]) − 1) / A[d][n]  ·  B[n]
 *
 * so the effective B̄ is still per-channel even though raw B is shared.
 *
 * @param params  Pointer to (const) model parameters (only A, D_skip used).
 * @param state   Pointer to mutable hidden state (updated in-place).
 * @param x       Input vector [D], one scalar per channel.
 * @param sel     Output of mamba_select_compute(): delta[D], B[N], C[N].
 * @param y_out   Output vector [D], written by this function.
 */
void mamba_s6_step_selective(const MambaS6Params    *params,
                              MambaS6State           *state,
                              const float             x[MAMBA_D],
                              const MambaSelectOutput *sel,
                              float                   y_out[MAMBA_D]);

/**
 * @brief Over-The-Air (OTA) Readout Updater for Dual Prediction Scheme.
 * Retrieves updated weights computed by the powerful Gateway and injects them
 * into the edge module's active parameters.
 * @param params Pointer to mutable model parameters.
 * @param new_W  Pointer to the new 6x6 Float Readout Weights (144 bytes).
 * @param new_bias Pointer to the new 6-Float array (24 bytes).
 */
void mamba_update_readout_weights(MambaS6Params *params, 
                                  const float new_W[MAMBA_D][MAMBA_D], 
                                  const float new_bias[MAMBA_D]);

#ifdef __cplusplus
}
#endif

#endif /* MAMBA_S6_H */
