/**
 * @file mamba_gait.c
 * @brief Serialization & Normalization Layer — implementation.
 *
 * See mamba_gait.h for full documentation, design rationale, and usage.
 *
 * Implementation notes
 * --------------------
 * 1. Generic struct access is done by casting the opaque gait_struct pointer
 *    to `const uint8_t *` and adding the byte offset stored in the descriptor.
 *    This is well-defined C for accessing struct members by offset as long as
 *    the correct type is then cast back.  We use `memcpy` (not pointer-cast)
 *    to read multi-byte fields to avoid potential alignment faults on the
 *    Xtensa LX6/LX7, which has alignment requirements for 16/32-bit loads.
 *
 * 2. The Min-Max clamp [0, 1] protects downstream layers from out-of-range
 *    inputs caused by sensor glitches or an incorrectly set physical range.
 *
 * 3. The descriptor array `s_default_fields[]` is declared `const` and
 *    `static` at file scope so the linker places it in .rodata (Flash) on
 *    ESP32 and uses read-only data memory on host.  The `static` qualifier
 *    here gives it internal linkage (not the same as restricting to SRAM).
 *
 * 4. All arithmetic is in float32 to utilise the ESP32 FPU.
 */

#include "mamba_gait.h"

#include <string.h>   /* memcpy   */
#include <stdint.h>   /* uintN_t  */
#include <stdio.h>    /* printf   */

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/**
 * @brief Clamp `v` to [lo, hi].
 */
static inline float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/**
 * @brief Safe Min-Max normalisation: (x - min) / (max - min).
 *
 * Returns 0.5f when max == min to avoid division by zero and stay neutral.
 */
static inline float minmax_norm(float x, float mn, float mx)
{
    float range = mx - mn;
    if (range < 1e-9f) return 0.5f;          /* degenerate range → midpoint */
    return clampf((x - mn) / range, 0.0f, 1.0f);
}

/**
 * @brief Read one field from an opaque struct byte-buffer and return as float.
 *
 * Uses memcpy to avoid alignment faults (safe on ESP32 Xtensa).
 * The base pointer is treated as `const uint8_t *` (byte-addressable).
 *
 * @param base    Pointer to the start of the struct.
 * @param desc    Field descriptor (offset + type).
 * @return        Field value cast to float.
 */
static float read_field_as_float(const void *base, const GaitFieldDescriptor *desc)
{
    const uint8_t *ptr = (const uint8_t *)base + desc->offset;

    switch (desc->type) {
        case FIELD_F32: {
            float v;
            memcpy(&v, ptr, sizeof(float));
            return v;
        }
        case FIELD_I8: {
            int8_t v;
            memcpy(&v, ptr, sizeof(int8_t));
            return (float)v;
        }
        case FIELD_U8: {
            uint8_t v;
            memcpy(&v, ptr, sizeof(uint8_t));
            return (float)v;
        }
        case FIELD_I16: {
            int16_t v;
            memcpy(&v, ptr, sizeof(int16_t));
            return (float)v;
        }
        case FIELD_U16: {
            uint16_t v;
            memcpy(&v, ptr, sizeof(uint16_t));
            return (float)v;
        }
        case FIELD_I32: {
            int32_t v;
            memcpy(&v, ptr, sizeof(int32_t));
            return (float)v;
        }
        case FIELD_U32: {
            uint32_t v;
            memcpy(&v, ptr, sizeof(uint32_t));
            return (float)v;
        }
        default:
            return 0.0f;    /* unknown type — return neutral value */
    }
}

/* =========================================================================
 * Default Flash-resident descriptor table for GaitSample
 *
 * Physical ranges are conservative clinical/sports-science values.
 * Adjust min_val / max_val to match your sensor specifications.
 *
 * Stored as `static const` → placed in .rodata (Flash) on ESP32.
 * ========================================================================= */
static const GaitFieldDescriptor s_default_fields[] = {
    /* ---------------------------------------------------------------
     * Temporal features
     * ------------------------------------------------------------- */
    GAIT_FIELD(GaitSample, stride_length_mm,      FIELD_U16,     0.f,  2500.f),
    GAIT_FIELD(GaitSample, cadence_steps_per_min, FIELD_F32,     0.f,   200.f),
    GAIT_FIELD(GaitSample, stance_time_ms,         FIELD_U16,     0.f,  1500.f),
    GAIT_FIELD(GaitSample, swing_time_ms,          FIELD_U16,     0.f,  1000.f),
    GAIT_FIELD(GaitSample, double_support_ms,      FIELD_U16,     0.f,   600.f),

    /* ---------------------------------------------------------------
     * Accelerometer — range: ±4 g sensor at mg resolution
     * ------------------------------------------------------------- */
    GAIT_FIELD(GaitSample, accel_x_mg,            FIELD_I16, -4000.f,  4000.f),
    GAIT_FIELD(GaitSample, accel_y_mg,            FIELD_I16, -4000.f,  4000.f),
    GAIT_FIELD(GaitSample, accel_z_mg,            FIELD_I16, -4000.f,  4000.f),

    /* ---------------------------------------------------------------
     * Gyroscope — range: ±2000 dps sensor at 0.001 dps/bit resolution
     *   35000 mdps ≈ 35 dps (comfortable walking angular rate)
     * ------------------------------------------------------------- */
    GAIT_FIELD(GaitSample, gyro_x_mdps,           FIELD_I16, -35000.f, 35000.f),
    GAIT_FIELD(GaitSample, gyro_y_mdps,           FIELD_I16, -35000.f, 35000.f),
    GAIT_FIELD(GaitSample, gyro_z_mdps,           FIELD_I16, -35000.f, 35000.f),

    /* ---------------------------------------------------------------
     * Kinetics & spatial
     * ------------------------------------------------------------- */
    GAIT_FIELD(GaitSample, foot_pressure_kpa,     FIELD_F32,     0.f,  1000.f),
    GAIT_FIELD(GaitSample, step_symmetry_pct,     FIELD_F32,     0.f,   100.f),
    GAIT_FIELD(GaitSample, velocity_mm_s,         FIELD_F32,     0.f,  5000.f),
    GAIT_FIELD(GaitSample, step_width_mm,         FIELD_U16,     0.f,   500.f),
    GAIT_FIELD(GaitSample, step_count,            FIELD_U32,     0.f, 1e6f  ),
};

/** Number of entries in the default descriptor table. */
#define N_DEFAULT_FIELDS \
    ((int)(sizeof(s_default_fields) / sizeof(s_default_fields[0])))

/* =========================================================================
 * Public API — implementation
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * mamba_gait_feature_map_default
 * ---------------------------------------------------------------------- */
void mamba_gait_feature_map_default(GaitFeatureMap *map)
{
    map->fields     = s_default_fields;
    map->num_fields = N_DEFAULT_FIELDS;
}

/* -------------------------------------------------------------------------
 * mamba_gait_extract_normalize
 *
 * Static (pre-fitted) Min-Max normalization using the physical ranges
 * stored in each descriptor's min_val / max_val.
 *
 * Algorithm per feature d:
 *   raw   = read_field_as_float(struct, desc[d])
 *   x[d]  = clamp( (raw - desc[d].min_val) / (desc[d].max_val - desc[d].min_val),
 *                  0.0f, 1.0f )
 * ---------------------------------------------------------------------- */
int mamba_gait_extract_normalize(const GaitFeatureMap *map,
                                  const void           *gait_struct,
                                  float                 x_out[MAMBA_D])
{
    /* Determine how many features we can actually fill. */
    const int n_use = (map->num_fields < MAMBA_D) ? map->num_fields : MAMBA_D;

    for (int d = 0; d < n_use; ++d) {
        const GaitFieldDescriptor *desc = &map->fields[d];
        float raw = read_field_as_float(gait_struct, desc);
        x_out[d]  = minmax_norm(raw, desc->min_val, desc->max_val);
    }

    /* Zero-pad any remaining channels so the SSM sees a clean input. */
    for (int d = n_use; d < MAMBA_D; ++d) {
        x_out[d] = 0.0f;
    }

    return n_use;
}

/* -------------------------------------------------------------------------
 * mamba_minmax_scaler_reset
 * ---------------------------------------------------------------------- */
void mamba_minmax_scaler_reset(MambaMinMaxScaler *scaler)
{
    scaler->initialized = 0;
    scaler->alpha       = 0.01f;   /* Default: adapts over ~100 samples */
    /* Values are garbage until the first observation; that is intentional —
     * initialized == 0 is the gate in update() and normalize().           */
}

/* -------------------------------------------------------------------------
 * mamba_minmax_scaler_update
 *
 * Incremental running-extremes update:
 *   On first call: seed min and max with the current observation.
 *   Subsequent calls: expand the window if the new value exceeds bounds.
 *
 * Complexity: O(n_features) with 2 comparisons per element.
 * ---------------------------------------------------------------------- */
void mamba_minmax_scaler_update(MambaMinMaxScaler *scaler,
                                const float        raw[MAMBA_D],
                                int                n_features)
{
    if (n_features > MAMBA_D) n_features = MAMBA_D;

    if (!scaler->initialized) {
        /* Seed: first observation defines both min and max. */
        for (int d = 0; d < n_features; ++d) {
            scaler->min_observed[d] = raw[d];
            scaler->max_observed[d] = raw[d];
        }
        /* Channels beyond n_features: initialise to neutral range [0, 1]. */
        for (int d = n_features; d < MAMBA_D; ++d) {
            scaler->min_observed[d] = 0.0f;
            scaler->max_observed[d] = 1.0f;
        }
        scaler->initialized = 1;
        return;
    }

    /* Expand and relax window per channel. */
    for (int d = 0; d < n_features; ++d) {
        /* 1. Expand immediately if outside the window to keep output in [0, 1]. */
        if (raw[d] < scaler->min_observed[d]) scaler->min_observed[d] = raw[d];
        if (raw[d] > scaler->max_observed[d]) scaler->max_observed[d] = raw[d];

        /* 2. Relax the bounds slightly towards the current observation.
         * This allows the scaler to "forget" old outliers and adapt to
         * drift or shifting signal ranges. */
        scaler->min_observed[d] += (raw[d] - scaler->min_observed[d]) * scaler->alpha;
        scaler->max_observed[d] += (raw[d] - scaler->max_observed[d]) * scaler->alpha;
    }
}

/* -------------------------------------------------------------------------
 * mamba_minmax_scaler_normalize
 *
 * Normalise using running extremes.  Before the first update (initialized
 * == 0) all channels are set to 0.5f — a neutral, midpoint value.
 * ---------------------------------------------------------------------- */
void mamba_minmax_scaler_normalize(const MambaMinMaxScaler *scaler,
                                   const float              raw[MAMBA_D],
                                   int                      n_features,
                                   float                    x_out[MAMBA_D])
{
    if (n_features > MAMBA_D) n_features = MAMBA_D;

    if (!scaler->initialized) {
        /* No data seen yet: emit neutral values and zero-pad. */
        for (int d = 0; d < n_features; ++d) x_out[d] = 0.5f;
        for (int d = n_features; d < MAMBA_D; ++d) x_out[d] = 0.0f;
        return;
    }

    for (int d = 0; d < n_features; ++d) {
        x_out[d] = minmax_norm(raw[d],
                               scaler->min_observed[d],
                               scaler->max_observed[d]);
    }
    for (int d = n_features; d < MAMBA_D; ++d) {
        x_out[d] = 0.0f;
    }
}

/* -------------------------------------------------------------------------
 * mamba_gait_extract_scale_adaptive
 *
 * All-in-one pipeline:
 *   1. Extract raw floats from the generic struct via descriptor offsets.
 *   2. Update the adaptive scaler's running min/max with the raw values.
 *   3. Normalise with the now-updated extremes.
 *
 * Using the updated (not pre-update) extremes ensures that if the current
 * sample is a new extreme, it normalises to exactly 0 or 1 rather than
 * slightly outside, keeping the SSM input tightly bounded.
 * ---------------------------------------------------------------------- */
int mamba_gait_extract_scale_adaptive(const GaitFeatureMap *map,
                                       const void           *gait_struct,
                                       MambaMinMaxScaler    *scaler,
                                       float                 x_out[MAMBA_D])
{
    const int n_use = (map->num_fields < MAMBA_D) ? map->num_fields : MAMBA_D;

    /* Step A: extract raw float values into a temporary buffer. */
    float raw[MAMBA_D];
    for (int d = 0; d < n_use; ++d) {
        raw[d] = read_field_as_float(gait_struct, &map->fields[d]);
    }
    for (int d = n_use; d < MAMBA_D; ++d) {
        raw[d] = 0.0f;
    }

    /* Step B: update running extremes from the raw observation. */
    mamba_minmax_scaler_update(scaler, raw, n_use);

    /* Step C: normalise using the updated extremes → x_out. */
    mamba_minmax_scaler_normalize(scaler, raw, n_use, x_out);

    return n_use;
}

/* -------------------------------------------------------------------------
 * mamba_gait_debug_print
 *
 * Prints a table of:
 *   field_name | raw_value | [min, max] | normalised_value
 *
 * Redirect printf to UART on ESP32 for runtime diagnostics.
 * ---------------------------------------------------------------------- */
void mamba_gait_debug_print(const GaitFeatureMap *map,
                             const void           *gait_struct,
                             float                 raw_out[MAMBA_D])
{
    const int n_use = (map->num_fields < MAMBA_D) ? map->num_fields : MAMBA_D;

    printf("%-28s  %10s  [%8s, %8s]  %8s\n",
           "Field", "Raw", "Min", "Max", "Normed");
    printf("%-28s  %10s  [%8s, %8s]  %8s\n",
           "----------------------------", "----------",
           "--------", "--------", "--------");

    for (int d = 0; d < n_use; ++d) {
        const GaitFieldDescriptor *desc = &map->fields[d];
        float raw    = read_field_as_float(gait_struct, desc);
        float normed = minmax_norm(raw, desc->min_val, desc->max_val);

        printf("%-28s  %10.3f  [%8.1f, %8.1f]  %8.4f\n",
               desc->name, raw, desc->min_val, desc->max_val, normed);

        if (raw_out) raw_out[d] = raw;
    }
    printf("\n");
}
