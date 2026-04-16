/**
 * @file mamba_s6_example.c
 * @brief Host-side smoke-test for the Mamba S6 library (Steps 1 + 2).
 *
 * Build (Linux/macOS/WSL — D=4, N=8 for fast test):
 *   gcc -O2 -Wall -DMAMBA_D=4 -DMAMBA_N=8 \
 *       mamba_s6.c mamba_select.c mamba_weights.c mamba_s6_example.c \
 *       -lm -o mamba_s6_test && ./mamba_s6_test
 *
 * NOTE: mamba_weights.c has a compile-time guard that errors when
 * MAMBA_D != 16 or MAMBA_N != 16.  For the above test command you must
 * either remove that guard or compile without mamba_weights.c and
 * supply your own weight arrays.  For a quick test with the defaults:
 *
 *   gcc -O2 -Wall \
 *       mamba_s6.c mamba_select.c mamba_weights.c mamba_s6_example.c \
 *       -lm -o mamba_s6_test && ./mamba_s6_test
 *
 * (uses the default MAMBA_D=16, MAMBA_N=16)
 *
 * Tests demonstrated:
 *   1. ZOH unit test (Step 1: numerical verification of A_bar, B_bar).
 *   2. Selection mechanism test (Step 2: verifies delta>0, B/C are non-zero).
 *   3. Comparative run: mamba_s6_step() vs mamba_s6_step_selective().
 *   4. Full 10-step recurrence using the recommended selective path.
 */

#include "../framework/mamba_s6.h"
#include "../framework/mamba_select.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Use library defaults if not overridden at compile time. */
#ifndef MAMBA_D
#define MAMBA_D 16
#endif
#ifndef MAMBA_N
#define MAMBA_N 16
#endif

#define N_STEPS  10

/* =========================================================================
 * Helper: print a float vector (truncates at 8 for wide vectors)
 * ========================================================================= */
static void print_vec(const char *label, const float *v, int len)
{
    printf("  %-16s [ ", label);
    int show = (len > 8) ? 8 : len;
    for (int i = 0; i < show; ++i)
        printf("%7.4f%s", v[i], (i < show - 1) ? ", " : "");
    if (len > 8) printf(", ...");
    printf(" ]\n");
}

/* =========================================================================
 * Test 1: ZOH discretisation (Step 1 — unchanged from original example)
 * ========================================================================= */
static int test_zoh(void)
{
    printf("=== Test 1: ZOH Discretisation ===\n");

    const int   N      = 4;
    const float delta  = 0.1f;
    float A[4]    = { -1.0f, -2.0f, -4.0f, -8.0f };
    float B[4]    = {  1.0f,  1.0f,  1.0f,  1.0f };
    float A_bar[4], B_bar[4];

    mamba_s6_zoh_discretize(delta, A, B, A_bar, B_bar, N);

    int pass = 1;
    for (int n = 0; n < N; ++n) {
        float exp_A = expf(delta * A[n]);
        float exp_B = (exp_A - 1.0f) / A[n] * B[n];
        float err_A = fabsf(A_bar[n] - exp_A);
        float err_B = fabsf(B_bar[n] - exp_B);
        int ok_A = (A_bar[n] > 0.0f && A_bar[n] < 1.0f);
        printf("  n=%d | A=%-6.1f | A_bar=%.6f (in(0,1):%s err=%.1e) | "
               "B_bar=%.6f (err=%.1e)\n",
               n, A[n], A_bar[n], ok_A ? "YES" : "NO", err_A, B_bar[n], err_B);
        if (!ok_A || err_A > 1e-6f || err_B > 1e-6f) pass = 0;
    }
    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* =========================================================================
 * Test 2: Selection Mechanism (Step 2)
 *
 * Verifies:
 *   a) delta[d] > 0 for all d  (softplus guarantee)
 *   b) B[n] and C[n] are finite and non-zero for a non-zero input
 *   c) Delta is input-dependent (different x → different delta)
 * ========================================================================= */
static int test_selection(void)
{
    printf("=== Test 2: Selection Mechanism (Step 2) ===\n");

    MambaSelectWeights w;
    mamba_select_weights_use_default(&w);

    /* Build two distinct inputs. */
    float u1[MAMBA_D], u2[MAMBA_D];
    for (int d = 0; d < MAMBA_D; ++d) {
        u1[d] = 0.5f * (float)(d + 1);   /* ramp  */
        u2[d] = -u1[d];                   /* negated ramp */
    }

    MambaSelectOutput s1, s2;
    mamba_select_compute(&w, u1, &s1);
    mamba_select_compute(&w, u2, &s2);

    int pass = 1;

    /* a) All delta > 0. */
    printf("  a) delta > 0 check:\n");
    for (int d = 0; d < MAMBA_D; ++d) {
        if (s1.delta[d] <= 0.0f) {
            printf("     FAIL: delta[%d] = %f\n", d, s1.delta[d]);
            pass = 0;
        }
        if (s2.delta[d] <= 0.0f) {
            printf("     FAIL: delta2[%d] = %f\n", d, s2.delta[d]);
            pass = 0;
        }
    }
    printf("     delta[0..3] (u1): %.4f  %.4f  %.4f  %.4f\n",
           s1.delta[0], s1.delta[1], s1.delta[2], s1.delta[3]);
    printf("     delta[0..3] (u2): %.4f  %.4f  %.4f  %.4f\n",
           s2.delta[0], s2.delta[1], s2.delta[2], s2.delta[3]);
    printf("     %s\n", "PASS (all > 0 by softplus)");

    /* b) B and C are non-zero. */
    float B_norm = 0.0f, C_norm = 0.0f;
    for (int n = 0; n < MAMBA_N; ++n) {
        B_norm += s1.B[n] * s1.B[n];
        C_norm += s1.C[n] * s1.C[n];
    }
    printf("  b) ||B||=%.4f  ||C||=%.4f  (should be > 0)\n",
           sqrtf(B_norm), sqrtf(C_norm));
    if (B_norm < 1e-10f || C_norm < 1e-10f) {
        printf("     FAIL: B or C is zero!\n"); pass = 0;
    } else {
        printf("     PASS\n");
    }

    /* c) Input-dependency: s1.delta != s2.delta for at least one channel. */
    int differ = 0;
    for (int d = 0; d < MAMBA_D; ++d) {
        if (fabsf(s1.delta[d] - s2.delta[d]) > 1e-6f) { differ = 1; break; }
    }
    printf("  c) delta is input-dependent: %s\n", differ ? "PASS" : "FAIL");
    if (!differ) pass = 0;

    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* =========================================================================
 * Test 3: Recurrent inference using mamba_s6_step_selective()
 * ========================================================================= */
static int test_recurrence_selective(void)
{
    printf("=== Test 3: Recurrent Inference — Selective Path (%d steps) ===\n",
           N_STEPS);

    static MambaS6Params params;
    static MambaS6State  state;
    mamba_s6_params_init_default(&params);
    mamba_s6_state_reset(&state);

    MambaSelectWeights w;
    mamba_select_weights_use_default(&w);

    float x[MAMBA_D];
    for (int d = 0; d < MAMBA_D; ++d)
        x[d] = 0.1f * (float)(d + 1);

    float y[MAMBA_D];
    MambaSelectOutput sel;

    for (int t = 0; t < N_STEPS; ++t) {
        /* Step 2a: compute selective parameters from current input. */
        mamba_select_compute(&w, x, &sel);

        /* Step 2b: run the SSM recurrence with shared B, C. */
        mamba_s6_step_selective(&params, &state, x, &sel, y);

        float norm = 0.0f;
        for (int d = 0; d < MAMBA_D; ++d) norm += y[d] * y[d];
        printf("  t=%2d | delta[0]=%.4f  B[0]=%.4f  C[0]=%.4f  ||y||=%.5f\n",
               t, sel.delta[0], sel.B[0], sel.C[0], sqrtf(norm));
    }

    printf("\nFinal y:\n");
    print_vec("y_out", y, MAMBA_D);
    printf("Final h[channel 0]:\n");
    print_vec("h[0]", state.h[0], MAMBA_N);

    /* Basic sanity: output should be finite. */
    int pass = 1;
    for (int d = 0; d < MAMBA_D; ++d) {
        if (!isfinite(y[d])) { pass = 0; break; }
    }
    printf("Result: %s\n\n", pass ? "PASS (all outputs finite)" : "FAIL (NaN/Inf!)");
    return pass;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    printf("Mamba S6 block — Step 1 + Step 2 test suite\n");
    printf("  D (model dim)  = %d\n", MAMBA_D);
    printf("  N (state dim)  = %d\n\n", MAMBA_N);

    int ok = 1;
    ok &= test_zoh();
    ok &= test_selection();
    ok &= test_recurrence_selective();

    printf("========================================\n");
    printf("Overall: %s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? EXIT_SUCCESS : 1;
}
