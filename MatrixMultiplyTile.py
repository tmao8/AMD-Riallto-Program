import npu
import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.build.kernel import Kernel
from npu.build.mtkernel import MTSplit, MTConcat, MTPassThrough

matrix_multiply_scalar = Kernel('ScalarMMv2.cpp')


class Matrix_Tiled(AppBuilder):
    def __init__(self, kernel: Kernel):
        self.kernel = []
        for _ in range(2):
            self.kernel.append(Kernel(kernel.srccode))
            self.kernel[-1].out_buffer.array = np.ndarray(shape=(16), dtype=np.uint8)
        self.mtsplit = MTSplit(2)
        self.mtconcat = MTConcat()
        self.mtbuffer = MTPassThrough()
        super().__init__()

    def callgraph(
        self, x_in1: np.ndarray, x_in2: np.ndarray, x_out: np.ndarray
    ) -> None:
        kernel_outputs = [None] * 2
        for matrix in range(x_in1.shape[0]):
            multicast = self.mtbuffer(x_in1[matrix])
            inputs2 = self.mtsplit(x_in2[matrix])
            for i in range(2):
                kernel_outputs[i] = self.kernel[i](multicast, inputs2[i], 4, 4, 4)
            mtbuffer_out = self.mtconcat(kernel_outputs)
            x_out[matrix] = mtbuffer_out


appbuilder = Matrix_Tiled(matrix_multiply_scalar)
A = np.zeros(shape=(2, 16), dtype=np.uint8)
B = np.zeros(shape=(2, 32), dtype=np.uint8)
x_out = np.zeros(shape=(2, 32), dtype=np.uint8)
appbuilder(A, B, x_out)
appbuilder.display()

appbuilder.build(A, B, x_out)

np.random.seed(42)

input_matrix1 = np.random.randint(1, 5, (2, 8, 8))
input_matrix2 = np.random.randint(1, 5, (2, 8, 8))
check1 = input_matrix1[0].reshape(8, 8) @ input_matrix2[0].reshape(8, 8)
print(check1)
check2 = input_matrix1[1].reshape(8, 8) @ input_matrix2[1].reshape(8, 8)
print(check2)


def get_quadrants(matrix):
    rows, cols = matrix.shape
    mid_row = rows // 2
    mid_col = cols // 2

    top_left = matrix[:mid_row, :mid_col]
    top_right = matrix[:mid_row, mid_col:]
    bottom_left = matrix[mid_row:, :mid_col]
    bottom_right = matrix[mid_row:, mid_col:]
    return top_left, top_right, bottom_left, bottom_right


def get_quadrants_list(matrices):
    num_matrices = matrices.shape[0]
    rows, cols = matrices.shape[1], matrices.shape[2]
    mid_row, mid_col = rows // 2, cols // 2

    quadrants_list = np.empty((4, num_matrices, mid_row, mid_col), dtype=matrices.dtype)

    for i in range(num_matrices):
        matrix = matrices[i]
        top_left, top_right, bottom_left, bottom_right = get_quadrants(matrix)

        quadrants_list[0, i] = top_left
        quadrants_list[1, i] = top_right
        quadrants_list[2, i] = bottom_left
        quadrants_list[3, i] = bottom_right

    return quadrants_list


A = get_quadrants_list(input_matrix1)
B = get_quadrants_list(input_matrix2)

A1 = A[0]
A1 = A1.reshape(A1.shape[0], -1)
A2 = A[1]
A2 = A2.reshape(A2.shape[0], -1)
A3 = A[2]
A3 = A3.reshape(A3.shape[0], -1)
A4 = A[3]
A4 = A4.reshape(A4.shape[0], -1)

B1 = B[0].reshape(B[0].shape[0], -1)
B2 = B[1].reshape(B[1].shape[0], -1)
B1B2 = np.concatenate((B1, B2), axis=1)


B3B4 = np.concatenate(
    (B[2].reshape(B[2].shape[0], -1), B[3].reshape(B[3].shape[0], -1)), axis=1
)

# a is a quadrant of matrix, b is 2 quadrants (B1 and B2, or B3 and B4), the output of one app is [[a1 * b1], [a1 * b2]]
from npu.runtime import AppRunner

app = AppRunner('Matrix_Tiled.xclbin')


a = app.allocate(shape=(2, 16), dtype=np.uint8)
b = app.allocate(shape=(2, 32), dtype=np.uint8)
c = app.allocate(shape=(2, 32), dtype=np.uint8)

a[:] = A1
b[:] = B1B2
a.sync_to_npu()
b.sync_to_npu()

app.call(a, b, c)
c.sync_from_npu()

output = c.copy()
print(c)
del app
