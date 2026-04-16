/**
 * @file mamba_gait.h
 * @brief Serialization & Normalization Layer for the Mamba framework.
 *
 * Purpose
 * -------
 * The Mamba SSM operates on a fixed-width float vector x ∈ ℝ^D.  Real
 * sensor data arrives as heterogeneous C structs containing integer and
 * float fields with wildly different physical ranges.  This module bridges
 * that gap:
 *
 *   GaitSample  ─── descriptor table ──►  raw float[D]  ─── scaler ──►  x[D]
 *   (any struct)                          (read by type                  (NN input,
 *                                          + cast)                       ∈ [0, 1])
 *
 * Generic struct access without reflection
 * ----------------------------------------
 * C has no runtime type information, so we use a *field descriptor table*:
 * each entry stores the byte offset of a member (via stddef's `offsetof`
 * macro), the storage type, and the physical min/max for normalization.
 *
 * Usage example:
 *
 *   GaitFeatureMap map;
 *   mamba_gait_feature_map_default(&map);       // or build your own
 *
 *   MambaMinMaxScaler scaler;
 *   mamba_minmax_scaler_reset(&scaler);
 *
 *   GaitSample sample = { ... };                // fill from sensors
 *   float x[MAMBA_D];
 *
 *   // Static (pre-fitted) normalization — uses min/max from descriptors:
 *   mamba_gait_extract_normalize(&map, &sample, x);
 *
 *   // Or: adaptive normalization — updates scaler on every call:
 *   mamba_gait_extract_scale_adaptive(&map, &sample, &scaler, x);
 *
 * Supported field types
 * ---------------------
 *   FIELD_F32   float (32-bit)
 *   FIELD_I8    int8_t
 *   FIELD_U8    uint8_t
 *   FIELD_I16   int16_t
 *   FIELD_U16   uint16_t
 *   FIELD_I32   int32_t
 *   FIELD_U32   uint32_t
 *
 * Min-Max normalization
 * ---------------------
 * Static (pre-fitted) scaler — one-shot, reads min/max from the descriptor:
 *   x_norm = clamp( (x - min) / (max - min), 0, 1 )
 *
 * Adaptive (online) scaler — updates running min/max per channel:
 *   Updates: scaler.min[d] = min(scaler.min[d], x_raw[d])
 *            scaler.max[d] = max(scaler.max[d], x_raw[d])
 *   Then normalises with the running extremes.
 *   On the first observation the output is 0.5 (neutral) for each channel.
 *
 * Padding / truncation
 *   - If num_fields < MAMBA_D: remaining x[d] = 0.0f (zero-padded).
 *   - If num_fields > MAMBA_D: only the first MAMBA_D fields are used
 *     (a compile-time assertion alerts you if the default map overflows).
 */

#ifndef MAMBA_GAIT_H
#define MAMBA_GAIT_H

#include "../framework/mamba_s6.h"   /* MAMBA_D */

#include <stddef.h>     /* offsetof, size_t */
#include <stdint.h>     /* int8_t … uint32_t */

/* =========================================================================
 * Field type tag
 *
 * Tells the extractor how to read raw bytes from the struct and convert
 * them to float before normalisation.
 * ========================================================================= */
typedef enum {
    FIELD_F32 = 0,  /**< float (32-bit single precision)    */
    FIELD_I8,       /**< int8_t                              */
    FIELD_U8,       /**< uint8_t                             */
    FIELD_I16,      /**< int16_t (little-endian on ESP32)   */
    FIELD_U16,      /**< uint16_t                            */
    FIELD_I32,      /**< int32_t                             */
    FIELD_U32,      /**< uint32_t                            */
} GaitFieldType;

/* =========================================================================
 * Field descriptor
 *
 * One entry per feature to extract.  Build a static array of these and
 * wrap it in a GaitFeatureMap.
 *
 * Example — manual construction:
 *
 *   static const GaitFieldDescriptor my_fields[] = {
 *       { offsetof(GaitSample, stride_length_mm), FIELD_U16, 0.f, 2500.f, "stride_mm" },
 *       { offsetof(GaitSample, cadence_steps_per_min), FIELD_F32, 0.f, 200.f, "cadence" },
 *   };
 *
 * Or use the GAIT_FIELD() macro below for less typing.
 * ========================================================================= */
typedef struct {
    size_t         offset;    /**< Byte offset of the member (use offsetof). */
    GaitFieldType  type;      /**< Storage type — determines how bytes are read. */
    float          min_val;   /**< Physical minimum; maps to normalised 0.0.  */
    float          max_val;   /**< Physical maximum; maps to normalised 1.0.  */
    const char    *name;      /**< Human-readable name for debugging / logging. */
} GaitFieldDescriptor;

/**
 * @brief Convenience macro: build a GaitFieldDescriptor for one struct member.
 *
 * @param STYPE   The struct type  (e.g. GaitSample)
 * @param MEMBER  The member name  (e.g. stride_length_mm)
 * @param FTYPE   GaitFieldType   (e.g. FIELD_U16)
 * @param FMIN    Physical minimum (float literal)
 * @param FMAX    Physical maximum (float literal)
 *
 * Example:
 *   GAIT_FIELD(GaitSample, stride_length_mm, FIELD_U16, 0.f, 2500.f)
 */
#define GAIT_FIELD(STYPE, MEMBER, FTYPE, FMIN, FMAX) \
    { offsetof(STYPE, MEMBER), (FTYPE), (float)(FMIN), (float)(FMAX), #MEMBER }

/* =========================================================================
 * Feature map
 *
 * Groups a descriptor array with its element count.  The struct itself is
 * tiny (pointer + int) and lives in SRAM; the descriptor array should be
 * `const` so the linker puts it in Flash.
 * ========================================================================= */
typedef struct {
    const GaitFieldDescriptor *fields;      /**< Pointer to descriptor array (Flash). */
    int                        num_fields;  /**< Number of valid descriptors.          */
} GaitFeatureMap;

/* =========================================================================
 * Concrete gait sample struct
 *
 * A realistic example of what an IMU/pressure-insole gait analysis system
 * might produce per stride.  16 fields → matches default MAMBA_D = 16.
 *
 * Units are chosen to be integer-safe at typical sensor resolutions:
 *   mm, ms, mg (milli-g), mdps (milli-degrees/s), kPa, %.
 *
 * You can replace this with your own struct; the normalization layer only
 * uses it via descriptors and never accesses it directly by field name.
 * ========================================================================= */
typedef struct {
    /* Temporal features */
    uint16_t stride_length_mm;          /**< Stride length                 [0, 2500] mm    */
    float    cadence_steps_per_min;     /**< Steps per minute              [0, 200]  spm   */
    uint16_t stance_time_ms;            /**< Foot-on-ground duration       [0, 1500] ms    */
    uint16_t swing_time_ms;             /**< Foot-off-ground duration      [0, 1000] ms    */
    uint16_t double_support_ms;         /**< Both-feet-on-ground duration  [0,  600] ms    */

    /* Kinematics — 3-axis accelerometer (milli-g) */
    int16_t  accel_x_mg;                /**< Lateral acceleration          [-4000, 4000] mg */
    int16_t  accel_y_mg;                /**< Anterior-posterior accel      [-4000, 4000] mg */
    int16_t  accel_z_mg;                /**< Vertical acceleration         [-4000, 4000] mg */

    /* Kinematics — 3-axis gyroscope (milli-degrees/s) */
    int16_t  gyro_x_mdps;              /**< Roll rate                      [-35000, 35000] mdps */
    int16_t  gyro_y_mdps;             /**< Pitch rate                      [-35000, 35000] mdps */
    int16_t  gyro_z_mdps;              /**< Yaw rate                       [-35000, 35000] mdps */

    /* Kinetics & spatial */
    float    foot_pressure_kpa;         /**< Peak plantar pressure         [0, 1000] kPa   */
    float    step_symmetry_pct;         /**< Stride symmetry ratio         [0, 100]  %     */
    float    velocity_mm_s;             /**< Walking speed                 [0, 5000] mm/s  */
    uint16_t step_width_mm;             /**< Lateral step width            [0,  500] mm    */
    uint32_t step_count;                /**< Cumulative step counter       [0, 1e6]        */
} GaitSample;

/* =========================================================================
 * Static compile-time check
 *
 * The default feature map extracts all 16 fields of GaitSample.
 * MAMBA_D must be >= 16 for this to work without truncation.
 *
 * This is a C11 _Static_assert; older compilers can remove it safely.
 * ========================================================================= */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(MAMBA_D >= 16,
    "MAMBA_D must be >= 16 to hold all GaitSample features. "
    "Reduce the number of GAIT_FIELD() entries or increase MAMBA_D.");
#endif

/* =========================================================================
 * Adaptive Min-Max Scaler
 *
 * Tracks the observed minimum and maximum of each of the D input features
 * across successive calls to mamba_gait_extract_scale_adaptive().
 * Normalises with the running extremes rather than fixed physical bounds.
 *
 * Advantages over static scaler:
 *   - Automatically adapts to sensor drift or miscalibration.
 *   - Requires no prior knowledge of physical ranges.
 *
 * Disadvantages:
 *   - First observation always outputs 0.5 (undetermined range).
 *   - Normalisation changes over time (non-stationary during warm-up).
 *   - Must call mamba_minmax_scaler_reset() between unrelated sessions.
 * ========================================================================= */
typedef struct {
    float min_observed[MAMBA_D]; /**< Running per-channel minimum.  */
    float max_observed[MAMBA_D]; /**< Running per-channel maximum.  */
    float alpha;                 /**< Forgetting factor (0 = static, 1 = stateless). Default 0.01. */
    int   initialized;           /**< 0 = no data seen yet.         */
} MambaMinMaxScaler;

/* =========================================================================
 * Public API
 * ========================================================================= */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Feature map -------------------------------------------------------- */

/**
 * @brief Populate a GaitFeatureMap for the default GaitSample struct.
 *
 * Points `map->fields` to a Flash-resident const descriptor array covering
 * all 16 members of GaitSample with calibrated physical min/max ranges.
 *
 * If you have a custom struct, build your own descriptor array and set
 * map->fields / map->num_fields directly.
 *
 * @param map  Output: populated feature map (pointer + count, in SRAM).
 */
void mamba_gait_feature_map_default(GaitFeatureMap *map);

/* ---- Static (pre-fitted) normalization --------------------------------- */

/**
 * @brief Extract and normalize a gait struct into a float vector x[MAMBA_D].
 *
 * For each descriptor in `map`:
 *   1. Reads the raw field from `gait_struct` at the stored byte offset.
 *   2. Converts to float according to the GaitFieldType.
 *   3. Applies static Min-Max normalization:
 *        x_norm = clamp( (raw - desc.min_val) / (desc.max_val - desc.min_val), 0, 1 )
 *   4. Writes to x_out[d].
 *
 * If `map->num_fields < MAMBA_D`:  remaining x_out entries are set to 0.0f.
 * If `map->num_fields > MAMBA_D`:  only the first MAMBA_D fields are used.
 *
 * @param map         Feature map (descriptor array + count).
 * @param gait_struct Pointer to any struct containing the fields.
 * @param x_out       Output float array [MAMBA_D], written by this function.
 * @return            Number of features actually written (min(num_fields, D)).
 */
int mamba_gait_extract_normalize(const GaitFeatureMap *map,
                                  const void           *gait_struct,
                                  float                 x_out[MAMBA_D]);

/* ---- Adaptive (online) normalization ----------------------------------- */

/**
 * @brief Reset an adaptive scaler — call before starting a new recording.
 * @param scaler  Pointer to the scaler to reset.
 */
void mamba_minmax_scaler_reset(MambaMinMaxScaler *scaler);

/**
 * @brief Update scaler running extremes from a raw (un-normalised) float vector.
 *
 * Call this BEFORE normalising if you want the scaler to learn from the
 * current observation first.
 *
 * @param scaler      Adaptive scaler to update.
 * @param raw         Raw (un-normalised) feature vector [MAMBA_D].
 * @param n_features  Number of valid entries in `raw` (rest are ignored).
 */
void mamba_minmax_scaler_update(MambaMinMaxScaler *scaler,
                                const float        raw[MAMBA_D],
                                int                n_features);

/**
 * @brief Normalise a pre-extracted raw feature vector using the adaptive scaler.
 *
 * Applies:
 *   x_norm[d] = clamp( (raw[d] - scaler.min[d]) / (scaler.max[d] - scaler.min[d]),
 *                      0.0f, 1.0f )
 *
 * On the very first call (before any update), outputs 0.5f per channel.
 *
 * @param scaler     Adaptive scaler (must have seen >= 1 observation via update).
 * @param raw        Raw feature vector [MAMBA_D].
 * @param n_features Number of valid entries.
 * @param x_out      Normalised output [MAMBA_D].
 */
void mamba_minmax_scaler_normalize(const MambaMinMaxScaler *scaler,
                                   const float              raw[MAMBA_D],
                                   int                      n_features,
                                   float                    x_out[MAMBA_D]);

/**
 * @brief All-in-one: extract from struct → update scaler → normalize → x_out.
 *
 * Equivalent to calling:
 *   mamba_gait_extract_raw()        (internal, extracts & casts fields)
 *   mamba_minmax_scaler_update()    (learns from this observation)
 *   mamba_minmax_scaler_normalize() (normalises using updated extremes)
 *
 * Preferred for streaming sensor data where physical ranges are unknown.
 *
 * @param map         Feature map.
 * @param gait_struct Pointer to any sensor struct.
 * @param scaler      Adaptive scaler (updated in-place).
 * @param x_out       Normalised output [MAMBA_D].
 * @return            Number of features written.
 */
int mamba_gait_extract_scale_adaptive(const GaitFeatureMap *map,
                                       const void           *gait_struct,
                                       MambaMinMaxScaler    *scaler,
                                       float                 x_out[MAMBA_D]);

/* ---- Debug utility ------------------------------------------------------ */

/**
 * @brief Print a feature map entry with its raw and normalised value.
 *
 * Writes to stdout (or UART if you redirect printf on ESP32).
 * Useful for verifying that scaling is sane during development.
 *
 * @param map         Feature map.
 * @param gait_struct Pointer to the struct to inspect.
 * @param raw_out     Optional: if non-NULL, filled with the pre-norm raw vector.
 */
void mamba_gait_debug_print(const GaitFeatureMap *map,
                             const void           *gait_struct,
                             float                 raw_out[MAMBA_D]);

#ifdef __cplusplus
}
#endif

#endif /* MAMBA_GAIT_H */
