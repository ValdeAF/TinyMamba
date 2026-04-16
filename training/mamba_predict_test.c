#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../framework/mamba_s6.h"
#include "../framework/mamba_select.h"
#include "mamba_test_data.h"

int main(void)
{
    printf("==========================================\n");
    printf("   Mamba S6 ESP32 Inference Verification   \n");
    printf("==========================================\n\n");

    /* 1. Initialise the Mamba Parameters and Select Weights from Flash arrays */
    MambaS6Params params;
    mamba_s6_params_init_default(&params);

    MambaSelectWeights select_weights;
    mamba_select_weights_use_default(&select_weights);

    /* 2. Initialise the recurrent hidden state memory to zero */
    MambaS6State state = {0};

    /* 3. Run the sequence step-by-step */
    float max_error = 0.0f;
    float sum_sq_error = 0.0f;

    for (int t = 0; t < TEST_SEQ_LEN; ++t) {
        
        const float *x_in = mamba_test_inputs[t];
        const float *y_expected = mamba_test_outputs_expected[t];
        
        /* The output buffer */
        float y_out[MAMBA_D];
        MambaSelectOutput sel_out;

        /* A) Dense Selection Projections */
        mamba_select_compute(&select_weights, x_in, &sel_out);

        /* B) Continuous-Discrete Recurrent ZOH Step */
        mamba_s6_step_selective(&params, &state, x_in, &sel_out, y_out);

        /* C) Compare output against PyTorch and accumulate error */
        float step_max_err = 0.0f;
        for (int d = 0; d < MAMBA_D; ++d) {
            float err = fabsf(y_out[d] - y_expected[d]);
            if (err > step_max_err) {
                step_max_err = err;
            }
            if (err > max_error) {
                max_error = err;
            }
            sum_sq_error += err * err;
        }

        if (t < 5 || t > TEST_SEQ_LEN - 5) {
            printf("Step %3d : Max Absolute Error = %f\n", t, step_max_err);
            if (t == 4) {
                printf("  ... (skipping middle steps) ...\n");
            }
        }
    }

    float mse = sum_sq_error / (float)(TEST_SEQ_LEN * MAMBA_D);
    printf("\n------------------------------------------\n");
    printf("Verification Results\n");
    printf("Total sequence length : %d frames\n", TEST_SEQ_LEN);
    printf("Maximum absolute error: %f\n", max_error);
    printf("Mean Squared Error    : %E\n", mse);

    if (max_error < 1e-4f) {
        printf("\nSUCCESS! The C engine strictly mirrors PyTorch.\n");
        printf("------------------------------------------\n");
        return 0;
    } else {
        printf("\nFAILURE: Discrepancy too high!\n");
        printf("------------------------------------------\n");
        return 1;
    }
}
