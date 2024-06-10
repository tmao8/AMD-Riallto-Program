#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define REL_WRITE 0
#define REL_READ 1
#include <aie_api/aie.hpp>

template <typename T, int N>
void matrix_multiply_aie_vector(const T* in_buffer1, const T* in_buffer2, T* out_buffer, const uint32_t elems) {

    ::aie::vector<uint8_t,64> A;
    ::aie::vector<uint8_t,64> B;
    ::aie::mmul<8,8,8,uint8_t,uint8_t> C;
    A = ::aie::load_v<64>(in_buffer1);
    B = ::aie::load_v<64>(in_buffer2);
    C.mul(A,B);

    ::aie::store_v(out_buffer, C.to_vector<uint8_t>());
}



extern "C" {
void matrix_multiply_vector(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, 
                        int32_t col_size) {
    matrix_multiply_aie_vector<uint8_t, 64>(in_buffer1, in_buffer2, out_buffer, col_size);
}
}