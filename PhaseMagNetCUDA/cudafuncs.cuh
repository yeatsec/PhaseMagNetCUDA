
#ifndef CUDAFUNCS_CUH
#define CUDAFUNCS_CUH

#include "pmncudautils.cuh"
#include "cuda_runtime.h"

__global__ void vecMatMultKernel(const CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>);
cudaError_t vecMatMultWithCuda(const Matrix<DTYPE>& A, const Matrix<DTYPE>& B, Matrix<DTYPE>& C);

__global__ void updateWeightsKernel(const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>);
cudaError_t updateWeightsWithCuda(const Matrix<DTYPE>& prevActs, Matrix<DTYPE>& weights, const Matrix<DTYPE>& nextError);

#endif // CUDAFUNCS_CUH