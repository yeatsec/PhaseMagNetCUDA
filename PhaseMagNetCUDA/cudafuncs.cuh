
#ifndef CUDAFUNCS_CUH
#define CUDAFUNCS_CUH

#include "pmncudautils.cuh"
#include "cuda_runtime.h"

__global__ void vecMatMultKernel(const CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>);
cudaError_t vecMatMultWithCuda(const Matrix<DTYPE>& Ar, const Matrix<DTYPE>& Ai, 
    const Matrix<DTYPE>& Br, const Matrix<DTYPE>& Bi, 
    Matrix<DTYPE>& Cr, Matrix<DTYPE>& Ci);

//__global__ void updateWeightsKernel(const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>);
//cudaError_t updateWeightsWithCuda(const Matrix<DTYPE>& prevActs, Matrix<DTYPE>& weights, const Matrix<DTYPE>& nextError);

// needs activation, error of prev and next, needs weights 2*(2+1) + 2 = 8 args
__global__ void complexBackpropKernel(const CudaMatrixArg<DTYPE> prevActR, const CudaMatrixArg<DTYPE> prevActI,
    CudaMatrixArg<DTYPE> prevError, const CudaMatrixArg<DTYPE> weightsR, const CudaMatrixArg<DTYPE> weightsI,
    const CudaMatrixArg<DTYPE> nextActR, const CudaMatrixArg<DTYPE> nextActI, const CudaMatrixArg<DTYPE> nextError);

cudaError_t complexBackpropWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI,
    Matrix<DTYPE>& prevError, Matrix<DTYPE>& weightsR, Matrix<DTYPE>& weightsI, Matrix<DTYPE>& nextBiasR, Matrix<DTYPE>& nextBiasI,
    const Matrix<DTYPE>& nextActR, const Matrix<DTYPE>& nextActI, const Matrix<DTYPE>& nextError);

#endif // CUDAFUNCS_CUH