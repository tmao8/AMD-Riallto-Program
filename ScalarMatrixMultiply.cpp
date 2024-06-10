#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define REL_WRITE 0
#define REL_READ 1
#include <aie_api/aie.hpp>

template <typename T, int N>
void matrix_multiply_scalar(const T *in_buffer1, const T *in_buffer2, T *out_buffer,
                            const uint32_t col_size)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            out_buffer[i * 8 + j] = 0;
            for (int k = 0; k < 8; k++)
            {
                out_buffer[i * 8 + j] += in_buffer1[i * 8 + k] * in_buffer2[k * 8 + j];
            }
        }
    }
}

extern "C"
{
    void example(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer,
                 int32_t col_size)
    {
        matrix_multiply_scalar<uint8_t, 64>(in_buffer1, in_buffer2, out_buffer, col_size);
    }
}