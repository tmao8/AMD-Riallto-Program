#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define REL_WRITE 0
#define REL_READ 1
#include <aie_api/aie.hpp>

        
extern "C" {
void matrix_multiply_scalar(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, 
                        int32_t M, int32_t K, int32_t N) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                out_buffer[m * N + n] = 0;
                for (int k = 0; k < K; k++) {
                    out_buffer[m * N + n] += in_buffer1[m * K + k] * in_buffer2[k * N + n];
                }
            }
        }
}
} // extern end
