import numpy as np
"""
    We assume that the input matrices A and B are 2D arrays.
    AND we assume that the tile size is a divisor of the dimensions of A and B.
    This module performs matrix multiplication using a tiled approach.
"""
class Tile_MatrixMul():
    def __init__(self, tile_size):
        super(Tile_MatrixMul, self).__init__()
        self.tile_size = tile_size

    def forward(self, A, B):
        # Assuming A and B are 2D arrays
        A_rows, A_cols = A.shape
        B_rows, B_cols = B.shape
        assert A_cols == B_rows, "Inner dimensions must match for matrix multiplication."
        assert A_rows % self.tile_size == 0 and B_cols % self.tile_size == 0, "Tile size must divide the dimensions of A and B."
        C = np.zeros((A_rows, B_cols), dtype=A.dtype)
        A_row_loop = A_rows // self.tile_size
        A_col_loop = A_cols // self.tile_size
        B_col_loop = B_cols // self.tile_size
        for i in range(A_row_loop):
            for j in range(A_col_loop):
                A_tile = A[i * self.tile_size:(i + 1) * self.tile_size, j * self.tile_size:(j + 1) * self.tile_size]
                for k in range(B_col_loop):
                    B_tile = B[j * self.tile_size:(j + 1) * self.tile_size, k * self.tile_size:(k + 1) * self.tile_size]
                    for size_1 in range(self.tile_size):
                        for size_2 in range(self.tile_size):
                            C[i * self.tile_size + size_1, k * self.tile_size + size_2] += A_tile[size_1, :] @ B_tile[:, size_2]
        return C
def huge_arry_test():
    tile_size = 8
    model = Tile_MatrixMul(tile_size)

    A = np.random.randn(1024 * 128, 648).astype(np.float32)
    B = np.random.randn(648, 2048 * 648).astype(np.float32)

    C = model.forward(A, B)

    assert C.shape == (512, 64), f"输出维度应为 (32, 32)，但实际为 {C.shape}"

    C_expected = A @ B
    assert np.allclose(C, C_expected, atol=1e-4), "结果与 NumPy 矩阵乘法不一致"

    print("测试2，结果正确。\n")

def test_tile_matrix_mul2():
    tile_size = 8
    model = Tile_MatrixMul(tile_size)

    A = np.random.randn(512, 32).astype(np.float32)
    B = np.random.randn(32, 64).astype(np.float32)

    C = model.forward(A, B)

    assert C.shape == (512, 64), f"输出维度应为 (32, 32)，但实际为 {C.shape}"

    C_expected = A @ B
    assert np.allclose(C, C_expected, atol=1e-4), "结果与 NumPy 矩阵乘法不一致"

    print("测试2，结果正确。\n")
def test_tile_matrix_mul1():
    tile_size = 4  
    model = Tile_MatrixMul(tile_size)

    A = np.random.randn(32, 32).astype(np.float32)
    B = np.random.randn(32, 32).astype(np.float32)

    C = model.forward(A, B)

    assert C.shape == (32, 32), f"输出维度应为 (32, 32)，但实际为 {C.shape}"

    C_expected = A @ B
    assert np.allclose(C, C_expected, atol=1e-4), "结果与 NumPy 矩阵乘法不一致"

    print("测试通过：输出为 32x32，结果正确。")

if __name__ == "__main__":
    test_tile_matrix_mul1()
    test_tile_matrix_mul2()
    huge_arry_test()