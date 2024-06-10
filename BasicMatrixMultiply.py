import npu
import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.build.kernel import Kernel

example = Kernel('VectorMatrixMultiply.cpp')
print(example.srccode)

example.out_buffer.array = np.ndarray(shape=(64), dtype=np.uint8)


class Example(AppBuilder):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        super().__init__()

    def callgraph(
        self, x_in1: np.ndarray, x_in2: np.ndarray, x_out: np.ndarray
    ) -> None:
        for i in range(x_in1.shape[0]):
            x_out[i] = self.kernel(x_in1[i], x_in2[i], x_in1.shape[1])


app_builder = Example(kernel=example)
A = np.zeros(shape=(2, 64), dtype=np.uint8)
B = np.zeros(shape=(2, 64), dtype=np.uint8)
x_out = np.zeros(shape=(2, 64), dtype=np.uint8)
app_builder(A, B, x_out)
app_builder.display()
app_builder.build(A, B, x_out)
from npu.runtime import AppRunner

app = AppRunner('Example.xclbin')
np.random.seed(42)

input_matrix1 = app.allocate(shape=(2, 64), dtype=np.uint8)
input_matrix2 = app.allocate(shape=(2, 64), dtype=np.uint8)
output_matrix = app.allocate(shape=(2, 64), dtype=np.uint8)

input_matrix1[:] = np.random.randint(1, 5, (2, 64))
input_matrix2[:] = np.random.randint(1, 5, (2, 64))
input_matrix1.sync_to_npu()
input_matrix2.sync_to_npu()
check1 = input_matrix1[0].reshape(8, 8) @ input_matrix2[0].reshape(8, 8)
print(check1)
check2 = input_matrix1[1].reshape(8, 8) @ input_matrix2[1].reshape(8, 8)
print(check2)

app.call(input_matrix1, input_matrix2, output_matrix)
output_matrix.sync_from_npu()
check = output_matrix.copy()
print(output_matrix)

del app
check.reshape(2, 8, 8)
