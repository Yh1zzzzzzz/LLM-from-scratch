import torch
from torch import nn
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
"""
    We assume that the itorchut matrices A and B are 2D arrays.
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
        C = torch.zeros((A_rows, B_cols), dtype=A.dtype, device=A.device)
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

    A = torch.rand(1024 * 64, 648).to(device)
    B = torch.rand(648, 32 * 648).to(device)

    C = model.forward(A, B)

    assert C.shape == (512, 64), f"输出维度应为 (32, 32)，但实际为 {C.shape}"

    C_expected = A @ B
    assert torch.allclose(C, C_expected, atol=1e-4), "结果与 NumPy 矩阵乘法不一致"

    print("测试2，结果正确。\n")

def test_tile_matrix_mul2():
    tile_size = 8
    model = Tile_MatrixMul(tile_size)

    A = torch.rand(512, 32).to(device)
    B = torch.rand(32, 64).to(device)

    C = model.forward(A, B)

    assert C.shape == (512, 64), f"输出维度应为 (32, 32)，但实际为 {C.shape}"

    C_expected = A @ B
    assert torch.allclose(C, C_expected, atol=1e-4), "结果与 NumPy 矩阵乘法不一致"

    print("测试2，结果正确。\n")
def test_tile_matrix_mul1():
    tile_size = 4  
    model = Tile_MatrixMul(tile_size)

    A = torch.rand(32, 32).to(device)
    B = torch.rand(32, 32).to(device)

    C = model.forward(A, B)

    assert C.shape == (32, 32), f"输出维度应为 (32, 32)，但实际为 {C.shape}"

    C_expected = A @ B
    assert torch.allclose(C, C_expected, atol=1e-4), "结果与 NumPy 矩阵乘法不一致"

    print("测试通过：输出为 32x32，结果正确。")

if __name__ == "__main__":
    import time
    test_tile_matrix_mul1()
    timebegin = time.time()
    test_tile_matrix_mul2()
    timeend = time.time()
    print(f"测试2耗时: {timeend - timebegin:.4f} 秒")
    time1 = time.time()
    huge_arry_test()
    time2 = time.time()
    print(f"测试大数组耗时: {time2 - time1:.4f} 秒")