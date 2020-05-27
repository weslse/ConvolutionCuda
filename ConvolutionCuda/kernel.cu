
/*
 * 2020.05.20 대량병렬처리 수업 발표
 * Chapter 7. Parallel Patterns : Convolution Example Code
 * Created by 이상은
 */

 // 해당 부분 주석을 풀고 사용하여 주시기 바랍니다.
 // * 7.4 의 코드는 발표 후 수정하여 정상적으로 작동합니다.




 /// __syncthread()를 사용하기 위한 보다 안전한 방법
 /// 다만 intellisense 문제로 빨간 밑줄
 /// NVIDIA 권장
 //#include "cuda_runtime.h"
 //#include "device_launch_parameters.h"
 //
 //// for syncthreads()
 //#ifdef __INTELLISENSE__
 //// in here put whatever is your favorite flavor of intellisense workarounds
 //void __syncthreads();
 //#endif
 //#include "device_functions.h"


 /// 간단한 문제는 크게 문제 없지만 복잡한 코드로 갔을 때 
 /// 잠재적으로 문제가 생길 수 있으니 지양해야 하는 코드
 /// NVIDIA 비권장
 //for __syncthreads()
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "device_functions.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>




// 1D
#define ARR_SIZE 7
#define MASK_SIZE 5
#define TILE_1D_SIZE 5


// 2D
#define IMG_WIDTH  200
#define TILE_WIDTH 5
#define MASK_WIDTH 3

#define O_TILE_WIDTH IMG_WIDTH
#define O_TILE_HEIGHT IMG_WIDTH

#define PITCH ((TILE_WIDTH) + (MASK_WIDTH) - 1)



// 배열 출력 함수들
void printArr(float* arr) {
	for (int i = 0; i < ARR_SIZE; i++)
		printf("%4.0f  ", arr[i]);
	printf("\n");
}

void printArr2D(float arr[O_TILE_WIDTH * O_TILE_HEIGHT]) {
	for (int i = 0; i < O_TILE_WIDTH; i++) {
		for (int j = 0; j < O_TILE_HEIGHT; j++)
			printf("%4.3f  ", arr[i * O_TILE_WIDTH + j]);
		printf("\n");
	}
}



/// 7.2 Simple Convolution 1D
//__global__ void convolution_1D_basic_kernel(float *src, float *mask, float *dst, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float result = 0.f;
//	int start_point = i - (Mask_Width / 2); // thread마다 시작 지점 계산
//	for (int j = 0; j < Mask_Width; j++) {
//		if (start_point + j >= 0 && start_point + j < Width) {
//			result += src[start_point + j] * mask[j];
//		}
//	}
//	dst[i] = result;
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
//	std::cout << "start parallelizing" << std::endl;
//	std::cout << "elapsed in time: ";
//	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//
//	convolution_1D_basic_kernel << < 1, ARR_SIZE >> > (devSrc, devMask, devResult, MASK_SIZE, ARR_SIZE);
//	cudaDeviceSynchronize();
//
//	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double>  duration = end - start;
//	std::cout << duration.count() * 1000 << std::endl;
//	std::cout << "----------------------------------" << std::endl;
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
//__global__ void convolution_1D_const_kernel(float *Src, float *Dst, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float result = 0.f;
//	int Conv_start_point = i - (Mask_Width / 2); // thread마다 시작 지점 계산
//	for (int j = 0; j < Mask_Width; j++) {
//		if (Conv_start_point + j >= 0 && Conv_start_point + j < Width) {
//			result += Src[Conv_start_point + j] * Mask[j];
//		}
//	}
//	Dst[i] = result;
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
//	std::cout << "start parallelizing" << std::endl;
//	std::cout << "elapsed in time: ";
//	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//
//	convolution_1D_const_kernel << < 1, ARR_SIZE >> > (devSrc, devResult, MASK_SIZE, ARR_SIZE);
//	cudaDeviceSynchronize();
//
//	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double>  duration = end - start;
//	std::cout << duration.count() * 1000 << std::endl;
//	std::cout << "----------------------------------" << std::endl;
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.4 Convolution 1D with Halo cells
//__constant__ float Mask[MASK_SIZE];
//__global__ void convolution_1D_halo_kernel(float *Src, float *Dst, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	__shared__ float tile_halo[TILE_1D_SIZE + MASK_SIZE - 1];
//
//	int n = Mask_Width / 2;
//
//	int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
//	if (threadIdx.x >= blockDim.x - n) {
//		tile_halo[threadIdx.x - (blockDim.x - n)]
//			= (halo_index_left < 0) ? 0 : Src[halo_index_left];
//	}
//
//	tile_halo[n + threadIdx.x] = Src[i];
//
//	int halo_index_right = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
//	if (threadIdx.x < n) {
//		tile_halo[n + blockDim.x + threadIdx.x] =
//			(halo_index_right >= Width) ? 0 : Src[halo_index_right];
//	}
//	__syncthreads();
//
//	float result = 0.f;
//	for (int j = 0; j < Mask_Width; j++) {
//		result += tile_halo[threadIdx.x + j] * Mask[j];
//	}
//	Dst[i] = result;
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
//	std::cout << "start parallelizing" << std::endl;
//	std::cout << "elapsed in time: ";
//	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//
//	convolution_1D_halo_kernel << < ceil((float)ARR_SIZE / TILE_1D_SIZE), TILE_1D_SIZE >> > (devSrc, devResult, MASK_SIZE, ARR_SIZE);
//	cudaDeviceSynchronize();
//
//	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double>  duration = end - start;
//	std::cout << duration.count() * 1000 << std::endl;
//	std::cout << "----------------------------------" << std::endl;
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.5 Convolution 1D using general caching
//__constant__ float Mask[MASK_SIZE];
//__global__ void convolution_1D_caching_kernel(float *Src, float *Dst, int Mask_Width, int Width)
//{
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	__shared__ float tile_sm[TILE_1D_SIZE];
//
//	tile_sm[threadIdx.x] = Src[i];
//
//	__syncthreads();
//
//	int This_tile_start_point = blockIdx.x * blockDim.x;
//	int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
//	int Conv_start_point = i - (Mask_Width / 2);
//
//	float result = 0.f;
//	for (int j = 0; j < Mask_Width; j++) {
//		int N_index = Conv_start_point + j;
//		if (N_index >= 0 && N_index < Width) {
//
//
//			if ((N_index >= This_tile_start_point)
//				&& (N_index < Next_tile_start_point)) {
//				result += tile_sm[threadIdx.x + j - (Mask_Width / 2)] * Mask[j];
//			}
//
//			else {
//				result += Src[N_index] * Mask[j];
//			}
//
//
//		}
//	}
//	Dst[i] = result;
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
//	std::cout << "start parallelizing" << std::endl;
//	std::cout << "elapsed in time: ";
//	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
//
//	convolution_1D_caching_kernel << < ceil((float)ARR_SIZE / TILE_1D_SIZE), TILE_1D_SIZE >> > (devSrc, devResult, MASK_SIZE, ARR_SIZE);
//	cudaDeviceSynchronize();
//
//	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double>  duration = end - start;
//	std::cout << duration.count() * 1000 << std::endl;
//	std::cout << "----------------------------------" << std::endl;
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * ARR_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	printArr(hstResult);
//
//	return 0;
//}


/// 7.6 Convolution 2D with Halo cells
// 책에서는 const float __restrict__ *M 이지만, 
// const float* __restrict__ M가 올바른 사용
// 또한 책에서 소개한 부분에 오류가 있어
// 인터넷에서 찾은 코드를 응용하였음
//#define W_SM ((TILE_WIDTH) + (MASK_WIDTH) - 1)
//__global__ void convolution_2D_tiled_kernel(float *inputData, float *outputData,
//	int width, int pitch, int height, int channels,
//	const float* __restrict__ M)
//{
//	__shared__ float N_ds[W_SM][W_SM];
//
//	int maskRadius = MASK_WIDTH / 2;
//
//	int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
//	int destY = dest / W_SM;		//col of shared memory
//	int destX = dest % W_SM;		//row of shared memory
//	int srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius; //row index to fetch data from input image
//	int srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;	//col index to fetch data from input image
//
//	if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
//		N_ds[destY][destX] = inputData[srcY * width + srcX];
//	else
//		N_ds[destY][destX] = 0;
//
//
//	dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
//	destY = dest / W_SM;
//	destX = dest % W_SM;
//	srcY = blockIdx.y *TILE_WIDTH + destY - maskRadius;
//	srcX = blockIdx.x *TILE_WIDTH + destX - maskRadius;
//	if (destY < W_SM) {
//		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
//			N_ds[destY][destX] = inputData[srcY *width + srcX];
//		else
//			N_ds[destY][destX] = 0;
//	}
//
//	__syncthreads();
//
//	float output = 0.0f;
//	for (int y = 0; y < MASK_WIDTH; y++)
//		for (int x = 0; x < MASK_WIDTH; x++)
//			output += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * MASK_WIDTH + x];
//
//	int oY = blockIdx.y * TILE_WIDTH + threadIdx.y;
//	int oX = blockIdx.x * TILE_WIDTH + threadIdx.x;
//
//	if (oY < height && oX < width)
//		outputData[oY * width + oX] = output;
//
//	__syncthreads();
//}
//
//int main()
//{
//	float hstSrc[IMG_WIDTH * IMG_WIDTH];
//	srand((unsigned int)time(NULL));
//
//	for (int i = 0; i < IMG_WIDTH; i++) {
//		for (int j = 0; j < IMG_WIDTH; j++) {
//			hstSrc[i * IMG_WIDTH + j] = 3;//rand() % 10;
//		}
//	}
//	float hstMask[MASK_WIDTH * MASK_WIDTH]
//		= { (1 / 9.f), (1 / 9.f), (1 / 9.f),
//			(1 / 9.f), (1 / 9.f), (1 / 9.f),
//			(1 / 9.f), (1 / 9.f), (1 / 9.f)
//	}; // 3*3 kernel mean filter
//
//	float hstResult[O_TILE_WIDTH * O_TILE_WIDTH] = { 0.f };
//
//	float* devSrc;
//	cudaError_t status;
//	status = cudaMalloc(&devSrc, sizeof(float) * IMG_WIDTH * IMG_WIDTH);
//	cudaMemcpy(devSrc, hstSrc, sizeof(float) * IMG_WIDTH * IMG_WIDTH, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//	float* devMask;
//	cudaMalloc(&devMask, sizeof(float) * MASK_WIDTH * MASK_WIDTH);
//	cudaMemcpy(devMask, hstMask, sizeof(float) * MASK_WIDTH * MASK_WIDTH, cudaMemcpyKind::cudaMemcpyHostToDevice);
//
//
//	float* devResult;
//	cudaMalloc(&devResult, sizeof(float) * O_TILE_WIDTH * O_TILE_WIDTH);
//
//	// CPU 계산 결과
//	std::cout << "start Convolution CPU" << std::endl;
//	std::cout << "elapsed in time: ";
//	std::chrono::high_resolution_clock::time_point start1 = std::chrono::high_resolution_clock::now();
//	for (int i = 0; i < IMG_WIDTH; i++) {
//		for (int j = 0; j < IMG_WIDTH; j++) {
//
//			float res = 0.f;
//			int x = j - MASK_WIDTH / 2;
//			int y = i - MASK_WIDTH / 2;
//			for (int mW = 0; mW < MASK_WIDTH; mW++) {
//				for (int mH = 0; mH < MASK_WIDTH; mH++) {
//
//					if ((x + mW) < 0 || (x + mW) >= IMG_WIDTH || (y + mH) < 0 || (y + mH) >= IMG_WIDTH)
//						continue;
//
//					res += hstSrc[(y + mH)* IMG_WIDTH + (x + mW)] * hstMask[mH * MASK_WIDTH + mW];
//				}
//			}
//
//			hstResult[i * O_TILE_HEIGHT + j] = res;
//
//		}
//	}
//
//	std::chrono::high_resolution_clock::time_point end1 = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double>  duration1 = end1 - start1;
//	std::cout << duration1.count() * 1000 << std::endl;
//	std::cout << "----------------------------------" << std::endl << std::endl;
//
//	//printArr2D(hstResult);
//
//	memset(hstResult, 0, sizeof(float) * O_TILE_WIDTH * O_TILE_WIDTH);
//
//
//	// GPU 계산 결과
//
//	std::cout << "start parallelizing" << std::endl;
//	std::cout << "elapsed in time: ";
//	std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
//
//	dim3 gridDim = dim3(ceil((float)IMG_WIDTH / TILE_WIDTH), ceil((float)IMG_WIDTH / TILE_WIDTH));
//	dim3 blockDim = dim3(TILE_WIDTH, TILE_WIDTH);
//
//	convolution_2D_tiled_kernel << < gridDim, blockDim >> > (devSrc, devResult, IMG_WIDTH, PITCH, IMG_WIDTH, 1, devMask);
//	cudaDeviceSynchronize();
//
//	std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
//	std::chrono::duration<double>  duration2 = end2 - start2;
//	std::cout << duration2.count() * 1000 << std::endl;
//	std::cout << "----------------------------------" << std::endl;
//
//	cudaMemcpy(hstResult, devResult, sizeof(float) * O_TILE_WIDTH * O_TILE_WIDTH, cudaMemcpyKind::cudaMemcpyDeviceToHost);
//
//	//printArr2D(hstResult);
//
//	return 0;
//}
