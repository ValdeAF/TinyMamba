/**
 * @file mamba_select.h
 * @brief Mamba S6 Selection Mechanism — dense linear projections for Δ, B, C.
 *
 * Background
 * ----------
 * In the S6 ("selective SSM") model the parameters that govern the state
 * transition are NOT fixed — they are functions of the current input token.
 * This is the core innovation of Mamba over prior SSMs.
 *
 * For each time-step t, given the SSM input vector u_t ∈ ℝ^D:
 *
 *   Δ_t  ∈ ℝ^D — per-channel timescale, computed by a D→D linear layer
 *                 followed by softplus to enforce positivity.
 *   B_t  ∈ ℝ^N — input vector, computed by a D→N linear layer.
 *                 **Shared across all D channels** (not per-channel).
 *   C_t  ∈ ℝ^N — output vector, computed by a D→N linear layer.
 *                 **Shared across all D channels** (not per-channel).
 *
 * This matches the S6 formulation from Gu & Dao (2023), §3.2:
 *   s_B(x) = Linear_N(x),  s_C(x) = Linear_N(x),  s_Δ(x) = softplus(Linear_D(x))
 *
 * Key difference from Step-1 simplified projections
 * --------------------------------------------------
 * Step 1 used scalar weights (one weight per channel):
 *   delta[d] = softplus( w_d * x[d] )          — only sees its own channel
 *   B[d][n]  = W_B[d][n] * x[d]                — per-channel, not shared
 *
 * This module uses proper dense matrix-vector products:
 *   delta[d] = softplus( Σ_k  W_delta[d][k] * x[k]  +  b_delta[d] )  — sees all D inputs
 *   B[n]     = Σ_k  W_B[n][k] * x[k]           — shared ∈ ℝ^N
 *   C[n]     = Σ_k  W_C[n][k] * x[k]           — shared ∈ ℝ^N
 *
 * Flash storage on ESP32
 * ----------------------
 * Weight arrays are declared as `const` globals in mamba_weights.c and
 * referenced here via pointer-to-const.  The ESP32 linker places `const`
 * globals in the `.rodata` section, which is mapped to Flash (SPI).
 * Access is transparently cached through the 32 KB I/D-cache on ESP32-S3.
 *
 * For ESP-IDF, do NOT add `DRAM_ATTR` to weight arrays — that would copy
 * them into precious SRAM.  The `const` keyword alone is sufficient.
 *
 * To use PROGMEM (Arduino-ESP32 style) instead, prepend:
 *   const float MAMBA_W_DELTA[...][...] PROGMEM = { ... };
 * and read via pgm_read_float_near() — though ESP32 does not require this
 * (unlike AVR), it is harmless.
 *
 * Arithmetic complexity per call to mamba_select_compute()
 * ---------------------------------------------------------
 *   Δ  projection: D × D  multiplies + D adds  + D softplus
 *   B  projection: N × D  multiplies + N adds
 *   C  projection: N × D  multiplies + N adds
 *   Total MACs:  D² + 2·N·D  (e.g. 768 MACs for D=N=16)
 */

#ifndef MAMBA_SELECT_H
#define MAMBA_SELECT_H

/* Pull in MAMBA_D and MAMBA_N defines (and forward-decl of MambaS6State). */
#include "mamba_s6.h"

/* =========================================================================
 * Weight structure
 *
 * Holds POINTERS to const weight arrays so the struct itself can live in
 * cheap SRAM while the actual weight data stays in Flash.
 *
 * Assign using mamba_select_weights_use_default() or manually point to
 * your own trained weight arrays.
 * ========================================================================= */

/**
 * @brief Projection weights for the S6 selection mechanism.
 *
 * All pointer fields must point to const-qualified arrays so the linker
 * knows it can place the targets in Flash (.rodata).
 *
 * Shape conventions (row-major, C multi-dim array layout):
 *   W_delta[MAMBA_D][MAMBA_D]  —  row d, column k  → W_delta[d][k]
 *   b_delta[MAMBA_D]           —  bias for channel d
 *   W_B    [MAMBA_N][MAMBA_D]  —  row n, column k  → W_B[n][k]
 *   W_C    [MAMBA_N][MAMBA_D]  —  row n, column k  → W_C[n][k]
 */
typedef struct {
    /**
     * W_delta: [D][D] projection, maps the full D-dim input to D-dim
     * pre-softplus timescale values.  One row per output channel.
     * Pointer to a 2-D const array in Flash.
     */
    const float (*W_delta)[MAMBA_D];    /* points to float[MAMBA_D][MAMBA_D] */

    /**
     * b_delta: [D] bias added before softplus.
     * Often initialised to a small positive value so the initial Δ after
     * softplus is meaningful (e.g., ≈ log(exp(1)-1) ≈ 0.541 → Δ ≈ 1).
     */
    const float *b_delta;               /* points to float[MAMBA_D]          */

    /**
     * W_B: [N][D] projection, maps D-dim input to N-dim shared B vector.
     * One row per state dimension.
     */
    const float (*W_B)[MAMBA_D];        /* points to float[MAMBA_N][MAMBA_D] */

    /**
     * W_C: [N][D] projection, maps D-dim input to N-dim shared C vector.
     */
    const float (*W_C)[MAMBA_D];        /* points to float[MAMBA_N][MAMBA_D] */

} MambaSelectWeights;

/* =========================================================================
 * Output structure
 *
 * Holds the computed selective parameters for one time-step.
 * Stack-allocate this in the calling function; it is small (~(D+2N) floats).
 * ========================================================================= */

/**
 * @brief Selective parameters computed from one input token u_t.
 *
 * delta[D] — per-channel timescales, always > 0 (softplus guarantees this).
 * B[N]     — shared input projection vector.
 * C[N]     — shared output projection vector.
 *
 * These feed directly into mamba_s6_step_selective() via ZOH discretisation.
 */
typedef struct MambaSelectOutput {
    float delta[MAMBA_D];   /**< Δ_t ∈ ℝ^D, positive, one per SSM channel. */
    float B[MAMBA_N];       /**< B_t ∈ ℝ^N, shared across all D channels.  */
    float C[MAMBA_N];       /**< C_t ∈ ℝ^N, shared across all D channels.  */
} MambaSelectOutput;

/* =========================================================================
 * Default Flash-resident weight arrays
 *
 * Defined as `const` in mamba_weights.c — the linker places them in .rodata
 * (Flash) automatically on ESP32.
 *
 * IMPORTANT: mamba_weights.c must be compiled with the SAME -DMAMBA_D and
 * -DMAMBA_N values as every other translation unit.  Add them to your
 * CMakeLists.txt target_compile_definitions() to ensure consistency.
 * ========================================================================= */
extern const float MAMBA_W_DELTA[MAMBA_D][MAMBA_D];
extern const float MAMBA_B_DELTA[MAMBA_D];
extern const float MAMBA_W_B[MAMBA_N][MAMBA_D];
extern const float MAMBA_W_C[MAMBA_N][MAMBA_D];

/* =========================================================================
 * Public API
 * ========================================================================= */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Wire a MambaSelectWeights struct to the default Flash weight arrays.
 *
 * After this call, w->W_delta, w->b_delta, w->W_B, w->W_C all point to the
 * `const` arrays defined in mamba_weights.c.  The struct itself is tiny
 * (four pointers) and should live in SRAM.
 *
 * @param w  Pointer to a MambaSelectWeights struct to initialise.
 */
void mamba_select_weights_use_default(MambaSelectWeights *w);

/**
 * @brief Compute Δ_t, B_t, C_t from the input vector u_t.
 *
 * Implements the three linear projections of the S6 selection mechanism:
 *
 *   delta_raw[d] = Σ_{k=0}^{D-1}  W_delta[d][k] * u[k]  +  b_delta[d]
 *   delta[d]     = softplus( delta_raw[d] )
 *
 *   B[n]         = Σ_{k=0}^{D-1}  W_B[n][k] * u[k]
 *
 *   C[n]         = Σ_{k=0}^{D-1}  W_C[n][k] * u[k]
 *
 * All three projections read weight data from const Flash arrays via the
 * pointers in `w`.  The results are written to `out` (SRAM).
 *
 * @param w    Pointer to weight struct (pointers to Flash-resident matrices).
 * @param u    Input vector u_t, length MAMBA_D.
 * @param out  Output struct: delta[D], B[N], C[N] — written by this function.
 */
void mamba_select_compute(const MambaSelectWeights *w,
                          const float               u[MAMBA_D],
                          MambaSelectOutput        *out);

#ifdef __cplusplus
}
#endif

#endif /* MAMBA_SELECT_H */
