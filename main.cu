#include <iostream>
#include <cuda_fp16.h>

__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

void eval(const unsigned m, const unsigned n) {
	float *A_f32;
	half *A_f16;

	cudaMalloc(&A_f32, sizeof(float) * m * n);
	cudaMemset(A_f32, 0, sizeof(float) * m * n);
	cudaMalloc(&A_f16, sizeof(half ) * m * n);
	cudaMemset(A_f16, 0, sizeof(half) * m * n);

	const auto block_size = dim3(32, 32);
	const auto grid_size = dim3((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

	s2h<<<grid_size, block_size>>>(
			m, n,
			A_f32, m,
			A_f16, m
			);

	cudaFree(A_f16);
	cudaFree(A_f32);
}

int main() {
	for (unsigned i = (1u << 10); i <= (1lu << 15); i <<= 1) {
		eval(i, i);
	}
}
