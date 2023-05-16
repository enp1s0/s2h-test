#include <iostream>
#include <chrono>
#include <cuda_fp16.h>

// Original s2h
__global__
void s2h(int m, int n, const float *as, int ldas, __half *ah, int ldah)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		ah[i + j*ldah] = __float2half(as[i + j*ldas]);
	}
}

// SW pipelined s2h
template <unsigned Size>
__device__ inline void cp(void* const dst, const void* const src) {
	static_assert(Size == 4 || Size == 8 || Size == 16, "Size must be one of 4, 8 and 16");
	if (Size == 4) {
		*(reinterpret_cast<uint32_t*>(dst)) = *(reinterpret_cast<const uint32_t*>(src));
	} else if (Size == 8) {
		*(reinterpret_cast<uint64_t*>(dst)) = *(reinterpret_cast<const uint64_t*>(src));
	} else {
		*(reinterpret_cast<ulong2*>(dst)) = *(reinterpret_cast<const ulong2*>(src));
	}
}

template <class IdxT, unsigned block_size, unsigned smem_len = block_size * 8>
__global__
void s2h_swpipe(const IdxT m, const IdxT n, const float * const as, int ldas, __half *ah, int ldah)
{
	__shared__ float smem_f32[smem_len];
	__shared__ half smem_f16[smem_len];

	const auto in = blockIdx.x;

	for (unsigned i = 0; i < m; i += smem_len) {
		if (i + smem_len <= m) {
			// Load FP32 elements
			if (reinterpret_cast<long>(ah) % 16 == 0 && ldah % 4 == 0) {
				for (unsigned j = 0; j < smem_len; j += block_size * 4) {
					const auto smem_i = j + threadIdx.x * 4;
					if (smem_len < block_size * 4 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<16>(&smem_f32[smem_i], &as[im + ldas * in]);
				}
				__syncthreads();
			} else if (reinterpret_cast<long>(ah) % 8 == 0 && ldah % 2 == 0) {
				for (unsigned j = 0; j < smem_len; j += block_size * 2) {
					const auto smem_i = j + threadIdx.x * 2;
					if (smem_len < block_size * 2 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<8>(&smem_f32[smem_i], &as[im + ldas * in]);
				}
				__syncthreads();
			} else {
				for (unsigned j = 0; j < smem_len; j += block_size) {
					const auto smem_i = j + threadIdx.x;
					const auto im = i + smem_i;
					cp<4>(&smem_f32[smem_i], &as[im + ldas * in]);
				}
			}
			// Convert to FP16
			for (unsigned j = 0; j < smem_len; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				smem_f16[smem_i] = __float2half(smem_f32[smem_i]);
			}
			// Store FP16s
			if (reinterpret_cast<long>(ah) % 16 == 0 && ldah % 8 == 0) {
				__syncthreads();
				for (unsigned j = 0; j < smem_len; j += block_size * 8) {
					const auto smem_i = j + threadIdx.x * 8;
					if (smem_len < block_size * 8 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<16>(&ah[im + ldah * in], &smem_f16[smem_i]);
				}
			} else if (reinterpret_cast<long>(ah) % 8 == 0 && ldah % 4 == 0) {
				__syncthreads();
				for (unsigned j = 0; j < smem_len; j += block_size * 4) {
					const auto smem_i = j + threadIdx.x * 4;
					if (smem_len < block_size * 4 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<8>(&ah[im + ldah * in], &smem_f16[smem_i]);
				}
			} else if (reinterpret_cast<long>(ah) % 4 == 0 && ldah % 2 == 0) {
				__syncthreads();
				for (unsigned j = 0; j < smem_len; j += block_size * 2) {
					const auto smem_i = j + threadIdx.x * 2;
					if (smem_len < block_size * 2 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<4>(&ah[im + ldah * in], &smem_f16[smem_i]);
				}
			} else {
				for (unsigned j = 0; j < smem_len; j += block_size) {
					const auto smem_i = j + threadIdx.x;
					const auto im = i + smem_i;
					ah[im + ldah * in] = smem_f16[smem_i];
				}
			}
		} else {
			// Load FP32 elements
			unsigned j = 0;
			for (; j < smem_len; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				const auto im = i + smem_i;
				if (im < m) {
					smem_f32[smem_i] = as[im + ldas * in];
				} else {
					break;
				}
			}
			const unsigned max_j = j;

			// Convert to FP16
			for (unsigned j = 0; j < max_j; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				smem_f16[smem_i] = __float2half(smem_f32[smem_i]);
			}
			// Store FP16s
			for (unsigned j = 0; j < max_j; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				const auto im = i + smem_i;
				ah[im + ldah * in] = smem_f16[smem_i];
			}
		}
	}
}

template <class Func>
void eval(const std::size_t m, const std::size_t n, const char* name, const Func func) {
	float *A_f32;
	half *A_f16;

	cudaMalloc(&A_f32, sizeof(float) * m * n);
	cudaMemset(A_f32, 0, sizeof(float) * m * n);
	cudaMalloc(&A_f16, sizeof(half ) * m * n);
	cudaMemset(A_f16, 0, sizeof(half) * m * n);


	cudaDeviceSynchronize();
	const auto start_clock = std::chrono::system_clock::now();

	func(m, n, A_f32, A_f16);

	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
	const auto throughput = m * n * (sizeof(float) + sizeof(half)) / elapsed_time;
	std::printf("%s,%lu,%lu,%e\n", name, m, n, throughput * 1e-9);

	cudaFree(A_f16);
	cudaFree(A_f32);
}

int main() {
	auto swpipe_s2h = [](const std::size_t m, const std::size_t n, const float* const A_f32, half* const A_f16) {
		constexpr auto block_size = 512;
		const auto grid_size = n;
		if (m * n >= (1lu << 32)) {
			s2h_swpipe<std::uint32_t, block_size, block_size * 2><<<grid_size, block_size>>>(
					m, n,
					A_f32, m,
					A_f16, m
					);
		} else {
			s2h_swpipe<std::uint64_t, block_size, block_size * 2><<<grid_size, block_size>>>(
					m, n,
					A_f32, m,
					A_f16, m
					);
		}
	};
	auto original_s2h = [](const std::size_t m, const std::size_t n, const float* const A_f32, half* const A_f16) {
		const auto block_size = dim3(32, 32);
		const auto grid_size = dim3((m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

		s2h<<<grid_size, block_size>>>(
				m, n,
				A_f32, m,
				A_f16, m
				);
	};

	std::printf("m,n,throughput_in_GB_per_sec\n");
	for (unsigned i = (1u << 10); i <= (1lu << 15); i <<= 1) {
		eval(i, i, "original", original_s2h);
		eval(i, i, "swpipe"  , swpipe_s2h);
	}
}
