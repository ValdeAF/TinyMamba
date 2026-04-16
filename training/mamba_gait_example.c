/**
 * @file mamba_gait_example.c
 * @brief Integration demo: GaitSample → normalize → S6 selection → SSM step.
 *
 * Build:
 *   gcc -O2 -Wall \
 *       mamba_s6.c mamba_select.c mamba_weights.c \
 *       mamba_gait.c mamba_gait_example.c \
 *       -lm -o mamba_gait_test && ./mamba_gait_test
 *
 * Tests:
 *   1. Static normalization: fixed physical bounds from descriptor table.
 *   2. Adaptive normalization: running min/max learned from a stream.
 *   3. Debug print: tabular view of each feature before/after scaling.
 *   4. End-to-end: [GaitSample] → normalize → select → SSM step → y[D].
 *   5. Custom struct: demonstrates using the macro with a non-GaitSample type.
 */

#include "mamba_gait.h"
#include "../framework/mamba_select.h"
#include "../framework/mamba_s6.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Helper: build a synthetic GaitSample (simulates one stride of brisk walk)
 * ========================================================================= */
static GaitSample make_sample(int step_index)
{
    GaitSample s;
    memset(&s, 0, sizeof(s));

    /* Temporal: brisk walking ≈ 100 steps/min, stride ~1400 mm */
    s.stride_length_mm      = (uint16_t)(1350 + (step_index % 5) * 10);
    s.cadence_steps_per_min = 100.0f + (float)(step_index % 3) * 2.0f;
    s.stance_time_ms        = 620;
    s.swing_time_ms         = 380;
    s.double_support_ms     = 120;

    /* Accelerometer: heel-strike dominant vertical ~800 mg, slight lateral */
    s.accel_x_mg  = (int16_t)(50  * (step_index % 2 ? 1 : -1));
    s.accel_y_mg  = (int16_t)(120 + step_index * 3);
    s.accel_z_mg  = (int16_t)(800 + (step_index % 4) * 50);

    /* Gyroscope: sagittal plane dominant during walking */
    s.gyro_x_mdps = (int16_t)( 500 + step_index * 10);
    s.gyro_y_mdps = (int16_t)(8000 - step_index * 20);
    s.gyro_z_mdps = (int16_t)( 200);

    /* Kinetics & spatial */
    s.foot_pressure_kpa   = 650.0f + (float)(step_index % 6) * 15.0f;
    s.step_symmetry_pct   = 95.0f  - (float)(step_index % 4) * 0.5f;
    s.velocity_mm_s       = 1400.0f + (float)(step_index % 5) * 30.0f;
    s.step_width_mm       = (uint16_t)(100 + step_index);
    s.step_count          = (uint32_t)(step_index + 1);

    return s;
}

/* =========================================================================
 * Test 1: Static (pre-fitted) Min-Max normalization
 * ========================================================================= */
static int test_static_normalization(void)
{
    printf("=== Test 1: Static Normalization (pre-fitted physical bounds) ===\n");

    GaitFeatureMap map;
    mamba_gait_feature_map_default(&map);

    GaitSample sample = make_sample(0);
    float x[MAMBA_D];
    int n = mamba_gait_extract_normalize(&map, &sample, x);

    printf("  Features extracted: %d / %d (MAMBA_D = %d)\n", n, map.num_fields, MAMBA_D);

    int pass = 1;
    for (int d = 0; d < n; ++d) {
        if (x[d] < 0.0f || x[d] > 1.0f) {
            printf("  FAIL: x[%d] = %.4f out of [0, 1]\n", d, x[d]);
            pass = 0;
        }
    }

    printf("  x[0..7]: ");
    for (int d = 0; d < 8; ++d) printf("%.3f ", x[d]);
    printf("\n");
    printf("  x[8..15]: ");
    for (int d = 8; d < 16; ++d) printf("%.3f ", x[d]);
    printf("\n");

    printf("  All values in [0,1]: %s\n", pass ? "YES" : "NO");
    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* =========================================================================
 * Test 2: Adaptive (online) Min-Max normalization
 * ========================================================================= */
static int test_adaptive_normalization(void)
{
    printf("=== Test 2: Adaptive Normalization (running min/max) ===\n");

    GaitFeatureMap    map;
    MambaMinMaxScaler scaler;
    mamba_gait_feature_map_default(&map);
    mamba_minmax_scaler_reset(&scaler);

    const int N_OBS = 8;
    printf("  Streaming %d synthetic observations:\n", N_OBS);
    printf("  %-4s  %-8s  %-8s  %-8s\n", "Step", "x[0]", "x[5]", "x[14]");

    float x[MAMBA_D];
    int   all_bounded = 1;

    for (int i = 0; i < N_OBS; ++i) {
        GaitSample s = make_sample(i);
        mamba_gait_extract_scale_adaptive(&map, &s, &scaler, x);

        /* Check bounds */
        for (int d = 0; d < MAMBA_D; ++d) {
            if (x[d] < 0.0f || x[d] > 1.0f) { all_bounded = 0; }
        }

        printf("  %-4d  %-8.4f  %-8.4f  %-8.4f\n", i, x[0], x[5], x[14]);
    }

    /* After seeing multiple samples, x[0] for the first observation should
     * be 0.0 (the minimum seen) when re-normalised — verify the scaler
     * has expanded its range. */
    float range_stride = scaler.max_observed[0] - scaler.min_observed[0];
    printf("  Scaler learned range for stride_length_mm: [%.0f, %.0f] = %.0f mm\n",
           scaler.min_observed[0], scaler.max_observed[0], range_stride);

    int pass = all_bounded && (range_stride > 0.0f);
    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* =========================================================================
 * Test 3: Debug print
 * ========================================================================= */
static void test_debug_print(void)
{
    printf("=== Test 3: Debug Feature Table ===\n");

    GaitFeatureMap map;
    mamba_gait_feature_map_default(&map);
    GaitSample sample = make_sample(5);

    mamba_gait_debug_print(&map, &sample, NULL);
}

/* =========================================================================
 * Test 4: End-to-end — GaitSample → normalize → select → SSM → y[D]
 * ========================================================================= */
static int test_end_to_end(void)
{
    printf("=== Test 4: End-to-End Pipeline ===\n");

    /* --- Serialization layer setup ---------------------------------------- */
    GaitFeatureMap    map;
    MambaMinMaxScaler scaler;
    mamba_gait_feature_map_default(&map);
    mamba_minmax_scaler_reset(&scaler);

    /* --- SSM layer setup --------------------------------------------------- */
    static MambaS6Params    params;
    static MambaS6State     state;
    MambaSelectWeights      sel_w;

    mamba_s6_params_init_default(&params);
    mamba_s6_state_reset(&state);
    mamba_select_weights_use_default(&sel_w);

    printf("  Running 10-step gait sequence through full pipeline:\n");
    printf("  %-4s  %-10s  %-8s\n", "Step", "||y_out||", "delta[0]");

    float x[MAMBA_D];
    float y[MAMBA_D];
    MambaSelectOutput sel_out;

    int pass = 1;
    for (int t = 0; t < 10; ++t) {
        GaitSample s = make_sample(t);

        /* Stage 1: serialize & normalize → x[D] */
        mamba_gait_extract_scale_adaptive(&map, &s, &scaler, x);

        /* Stage 2: compute selective parameters Δ, B_t, C_t */
        mamba_select_compute(&sel_w, x, &sel_out);

        /* Stage 3: one recurrent SSM step */
        mamba_s6_step_selective(&params, &state, x, &sel_out, y);

        float norm = 0.0f;
        for (int d = 0; d < MAMBA_D; ++d) norm += y[d] * y[d];
        norm = sqrtf(norm);

        if (!isfinite(norm)) { pass = 0; }
        printf("  %-4d  %-10.5f  %-8.4f\n", t, norm, sel_out.delta[0]);
    }

    printf("Result: %s\n\n", pass ? "PASS (all outputs finite)" : "FAIL (NaN/Inf)");
    return pass;
}

/* =========================================================================
 * Test 5: Custom struct — demonstrate GAIT_FIELD with a non-GaitSample type
 *
 * Any struct can be used. Here a simplified 4-field IMU reading is wired up
 * and padded to MAMBA_D with zeros.
 * ========================================================================= */
typedef struct {
    float    ax_g;      /* acceleration in g          */
    float    ay_g;
    int16_t  gyr_dps;   /* angular rate in degrees/s  */
    uint8_t  heartrate; /* BPM from optical sensor    */
} SimpleIMU;

static int test_custom_struct(void)
{
    printf("=== Test 5: Custom Struct (SimpleIMU, 4 fields → padded to D) ===\n");

    static const GaitFieldDescriptor imu_fields[] = {
        GAIT_FIELD(SimpleIMU, ax_g,      FIELD_F32,  -4.0f,  4.0f),
        GAIT_FIELD(SimpleIMU, ay_g,      FIELD_F32,  -4.0f,  4.0f),
        GAIT_FIELD(SimpleIMU, gyr_dps,   FIELD_I16, -2000.f, 2000.f),
        GAIT_FIELD(SimpleIMU, heartrate, FIELD_U8,   40.f,   200.f),
    };

    GaitFeatureMap map = {
        .fields     = imu_fields,
        .num_fields = (int)(sizeof(imu_fields) / sizeof(imu_fields[0])),
    };

    SimpleIMU imu = { .ax_g = 1.5f, .ay_g = -0.3f, .gyr_dps = 450, .heartrate = 78 };

    float x[MAMBA_D];
    int n = mamba_gait_extract_normalize(&map, &imu, x);

    printf("  Fields used: %d, MAMBA_D: %d\n", n, MAMBA_D);
    printf("  x: [");
    for (int d = 0; d < MAMBA_D; ++d) {
        printf("%.3f%s", x[d], d < MAMBA_D - 1 ? ", " : "");
        if (d == 7) { printf("\n      "); }
    }
    printf("]\n");

    printf("  Expecting x[2] (gyr=450, range[-2000,2000]) ≈ 0.6125: %.4f\n", x[2]);

    float expected = (450.0f - (-2000.0f)) / (2000.0f - (-2000.0f));  /* = 0.6125 */
    int pass = (fabsf(x[2] - expected) < 1e-5f);
    printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void)
{
    printf("Mamba Gait Serialization Layer — Step 3 Test Suite\n");
    printf("  MAMBA_D = %d\n\n", MAMBA_D);

    int ok = 1;
    ok &= test_static_normalization();
    ok &= test_adaptive_normalization();
    test_debug_print();                           /* informational, no pass/fail */
    ok &= test_end_to_end();
    ok &= test_custom_struct();

    printf("==============================================\n");
    printf("Overall: %s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? EXIT_SUCCESS : 1;
}
