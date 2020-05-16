
/// __syncthread()�� ����ϱ� ���� ���� ������ ���
/// �ٸ� intellisense ������ ���� ����
/// NVIDIA ����
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//// for syncthreads()
//#ifdef __INTELLISENSE___
//// in here put whatever is your favorite flavor of intellisense workarounds
//void __syncthreads();
//#endif
//#include "device_functions.h"

/// ������ ������ ũ�� ���� ������ ������ �ڵ�� ���� �� 
/// ���������� ������ ���� �� ������ �����ؾ� �ϴ� �ڵ�
/// NVIDIA �����
//for __syncthreads()
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>


#include <stdio.h>

/*
// define ��� ����
//
// TILE 
// �����̵� ������ ��������� ������ ���� ����
// ���� ���� TILE�� OUTPUT ����
//
// MASK_WIDTH, MAX_MASK_SIZE ��
// å������ mask�� ũ�⸦ �ٸ��� �ϱ� ���� MAX ����
// ���� ����� MASK_WIDTH�� ����
//
// PITCH
// �е��� �߰��� image�� ���� ���̸� ��Ī
// Pitch = Width + padding size
*/

// TODO : �ٽ� �� ���߾� �����丵 �� ��!!
// N, P �� ����� �ܾ �˱� ���� �̸��� �ٲپ� �ٽ� �ڵ��� ��

// 1D
#define ARR_SIZE 7
#define MASK_SIZE 5

// 2D
#define O_TILE_WIDTH 5
#define O_TILE_HEIGHT 5

#define TILE_WIDTH 5
#define MASK_WIDTH 3

#define PITCH ((TILE_WIDTH) + (MASK_WIDTH) - 1)

void printArr(float* arr) {
	for (int i = 0; i < ARR_SIZE; i++)
		printf("%4.0f  ", arr[i]);
	printf("\n");
}

void printArr2D(float arr[PITCH][PITCH]) {
	for (int i = 0; i < PITCH; i++) {
		for (int j = 0; j < PITCH; j++)
			printf("%4.3f  ", arr[i][j]);
		printf("\n");
	}
}



/// 7.2 Simple Convolution 1D
//__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float Pvalue = 0.f;
//	int N_start_point = i - (Mask_Width / 2);
//	for (int j = 0; j < Mask_Width; j++) {
//		if (N_start_point + j >= 0 && N_start_point + j < Width) {
//			Pvalue += N[N_start_point + j] * M[j];
//		}
//	}
//	P[i] = Pvalue;
//}
//
//int main()
//{
//	float hstSrc[ARR_SIZE] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f };
//	float hstMask[MASK_SIZE] = { 3.f, 4.f, 5.f, 4.f, 3.f };
//	float hstResult[ARR_SIZE] = { 0.f };
//
//	float* devSrc = nullptr;
//	cudaMalloc(&devSrc, sizeof(float) * ARR_SIZE);
//	cudaMemcpy(devSrc, hstSrc, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//	float* devMask = nullptr;
//	cudaMalloc(&devMask, sizeof(float) * MASK_SIZE);
//	cudaMemcpy(devMask, hstMask, sizeof(float) * MASK_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//
//	float* devResult = nullptr;
//	cudaMalloc(&devResult, sizeof(float) * ARR_SIZE);
//
//	convolution_1D_basic_kernel << < 1, ARR_SIZE >> > (devSrc, devMask, devResult, MASK_SIZE, ARR_SIZE);
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.3 Convolution 1D with Contant Mask
//__constant__ float Mask[MASK_SIZE];
//
//__global__ void convolution_1D_const_kernel(float *N, float *P, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float Pvalue = 0.f;
//	int N_start_point = i - (Mask_Width / 2);
//	for (int j = 0; j < Mask_Width; j++) {
//		if (N_start_point + j >= 0 && N_start_point + j < Width) {
//			Pvalue += N[N_start_point + j] * Mask[j];
//		}
//	}
//	P[i] = Pvalue;
//}
//
//int main()
//{
//	float hstSrc[ARR_SIZE] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f };
//	float hstMask[MASK_SIZE] = { 3.f, 4.f, 5.f, 4.f, 3.f };
//	float hstResult[ARR_SIZE] = { 0.f };
//
//	float* devSrc = nullptr;
//	cudaMalloc(&devSrc, sizeof(float) * ARR_SIZE);
//	cudaMemcpy(devSrc, hstSrc, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//	
//	cudaMemcpyToSymbol(Mask, hstMask, sizeof(float) * MASK_SIZE);
//
//
//	float* devResult = nullptr;
//	cudaMalloc(&devResult, sizeof(float) * ARR_SIZE);
//
//	convolution_1D_const_kernel << < 1, ARR_SIZE >> > (devSrc, devResult, MASK_SIZE, ARR_SIZE);
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.4 Convolution 1D with Halo cells
//__constant__ float Mask[MASK_SIZE];
//__global__ void convolution_1D_halo_kernel(float *N, float *P, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	__shared__ float N_halo[ARR_SIZE + MASK_SIZE - 1];
//
//	int n = Mask_Width / 2;
//
//	int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
//	if (threadIdx.x >= blockDim.x - n) {
//		N_halo[threadIdx.x - (blockDim.x - n)]
//			= (halo_index_left < 0) ? 0 : N[halo_index_left];
//	}
//
//	N_halo[n + threadIdx.x] = N[i];
//
//	int halo_index_right = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
//	if (threadIdx.x < n) {
//		N_halo[n + blockDim.x + threadIdx.x] =
//			(halo_index_right >= Width) ? 0 : N[halo_index_right];
//	}
//	__syncthreads();
//
//	float Pvalue = 0.f;
//	for (int j = 0; j < Mask_Width; j++) {
//		Pvalue += N_halo[threadIdx.x + j] * Mask[j];
//	}
//	P[i] = Pvalue;
//}
//
//int main()
//{
//	float hstSrc[ARR_SIZE] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f };
//	float hstMask[MASK_SIZE] = { 3.f, 4.f, 5.f, 4.f, 3.f };
//	float hstResult[ARR_SIZE] = { 0.f };
//
//	float* devSrc = nullptr;
//	cudaMalloc(&devSrc, sizeof(float) * ARR_SIZE);
//	cudaMemcpy(devSrc, hstSrc, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//
//	cudaMemcpyToSymbol(Mask, hstMask, sizeof(float) * MASK_SIZE);
//
//
//	float* devResult = nullptr;
//	cudaMalloc(&devResult, sizeof(float) * ARR_SIZE);
//
//	convolution_1D_halo_kernel << < 1, ARR_SIZE >> > (devSrc, devResult, MASK_SIZE, ARR_SIZE);
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.5 Convolution 1D using general caching
//__constant__ float Mask[MASK_SIZE];
//__global__ void convolution_1D_caching_kernel(float *N, float *P, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	__shared__ float N_sm[ARR_SIZE];
//
//	N_sm[threadIdx.x] = N[i];
//
//	__syncthreads();
//
//	int This_tile_start_point = blockIdx.x * blockDim.x;
//	int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
//	int N_start_point = i - (Mask_Width / 2);
//
//	float Pvalue = 0.f;
//	for (int j = 0; j < Mask_Width; j++) {
//		int N_index = N_start_point + j;
//		if (N_index >= 0 && N_index < Width) {
//			if ((N_index >= This_tile_start_point)
//				&& (N_index < Next_tile_start_point)) {
//				Pvalue += N_sm[threadIdx.x + j - (Mask_Width / 2)] * Mask[j];
//			}
//			else {
//				Pvalue += N_sm[threadIdx.x + j] * Mask[j];
//			}
//		}
//	}
//	P[i] = Pvalue;
//}
//
//int main()
//{
//	float hstSrc[ARR_SIZE] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f };
//	float hstMask[MASK_SIZE] = { 3.f, 4.f, 5.f, 4.f, 3.f };
//	float hstResult[ARR_SIZE] = { 0.f };
//
//	float* devSrc = nullptr;
//	cudaMalloc(&devSrc, sizeof(float) * ARR_SIZE);
//	cudaMemcpy(devSrc, hstSrc, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//
//	cudaMemcpyToSymbol(Mask, hstMask, sizeof(float) * MASK_SIZE);
//
//
//	float* devResult = nullptr;
//	cudaMalloc(&devResult, sizeof(float) * ARR_SIZE);
//
//	convolution_1D_caching_kernel << < 1, ARR_SIZE >> > (devSrc, devResult, MASK_SIZE, ARR_SIZE);
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.6 Convolution 2D with Halo cells
// å������ const float __restrict__ *M ������, 
// const float* __restrict__ M�� �ùٸ� ���
// data�� �̹� �е��� �Ǿ� �ִٰ� ����
__global__ void convolution_2D_tiled_kernel(float *data, float *P, int Mask_Width,
	int width, int pitch, int height, int channels,
	const float* __restrict__ M)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * O_TILE_WIDTH * ty;
	int col_o = blockIdx.x * O_TILE_WIDTH * tx;

	int row_i = row_o - (Mask_Width / 2);
	int col_i = col_o - (Mask_Width / 2);

	__shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
	if ((row_i >= 0) && (row_i < height) &&
		(col_i >= 0) && (col_i < width)) {
		N_ds[ty][tx] = data[row_i * pitch + col_i];
	}
	else {
		N_ds[ty][tx] = 0.0f;
	}
	__syncthreads();

	float output = 0.0f;
	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
		for (int i = 0; i < MASK_WIDTH; i++) {
			for (int j = 0; j < MASK_WIDTH; j++) {
				output += M[i * MASK_WIDTH + j] * N_ds[i + ty][j + tx];
			}
		}
		if (row_o < height && col_o < width) {
			P[row_o * width + col_o] = output; // ��� ����
		}
	}
	__syncthreads();
}

int main()
{
	float hstSrc[PITCH][PITCH]
		= {
			{ 0, 0, 0, 0, 0, 0, 0 },
			{ 0, 1, 2, 3, 4, 5, 0 },
			{ 0, 2, 4, 6, 8, 0, 0 },
			{ 0, 1, 2, 3, 4, 5, 0 },
			{ 0, 2, 4, 6, 8, 0, 0 },
			{ 0, 1, 2, 3, 4, 5, 0 },
			{ 0, 0, 0, 0, 0, 0, 0 }
	};

	float hstMask[MASK_WIDTH][MASK_WIDTH] = { 1 / 9.f }; // 3*3 kernel mean filter
	float hstResult[O_TILE_WIDTH][O_TILE_WIDTH] = { 0.f };

	float* devSrc = nullptr;
	cudaMalloc(&devSrc, sizeof(float) * PITCH * PITCH);
	cudaMemcpy(devSrc, hstSrc, sizeof(float) * PITCH * PITCH, cudaMemcpyKind::cudaMemcpyHostToDevice);

	float* devMask = nullptr;
	cudaMalloc(&devMask, sizeof(float) * MASK_WIDTH * MASK_WIDTH);
	cudaMemcpy(devMask, hstMask, sizeof(float) * MASK_WIDTH * MASK_WIDTH, cudaMemcpyKind::cudaMemcpyHostToDevice);


	float* devResult = nullptr;
	cudaMalloc(&devResult, sizeof(float) * O_TILE_WIDTH * O_TILE_WIDTH);


	convolution_2D_tiled_kernel << < dim3( 1, 1), dim3(O_TILE_WIDTH, O_TILE_WIDTH) >> > (devSrc, devResult, MASK_WIDTH,
		TILE_WIDTH, PITCH, TILE_WIDTH, 3, devMask);

	cudaMemcpy(hstSrc, devSrc, sizeof(float) * O_TILE_WIDTH * O_TILE_WIDTH, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	printArr2D(hstSrc);

	return 0;
}
