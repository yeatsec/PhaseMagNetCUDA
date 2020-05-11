
#ifndef CUDAFUNCS_CUH
#define CUDAFUNCS_CUH

#include "pmncudautils.cuh"
#include "cuda_runtime.h"

//__global__ void vecMatMultKernel(const CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>);
cudaError_t vecMatMultWithCuda(const Matrix<DTYPE>& Ar, const Matrix<DTYPE>& Ai, 
    const Matrix<DTYPE>& Br, const Matrix<DTYPE>& Bi, 
    Matrix<DTYPE>& Cr, Matrix<DTYPE>& Ci);
cudaError_t complexConvolutionWithCuda(const Matrix3D<DTYPE>& prevActR, const Matrix3D<DTYPE>& prevActI,
    const Matrix3D<DTYPE>& ConvR, const Matrix3D<DTYPE>& ConvI,
    Matrix3D<DTYPE>& postActR, Matrix3D<DTYPE>& postActI);

//__global__ void updateWeightsKernel(const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>);
//cudaError_t updateWeightsWithCuda(const Matrix<DTYPE>& prevActs, Matrix<DTYPE>& weights, const Matrix<DTYPE>& nextError);


//__global__ void complexBackpropKernel(const CudaMatrixArg<DTYPE> prevActR, const CudaMatrixArg<DTYPE> prevActI,
//    CudaMatrixArg<DTYPE> prevError, const CudaMatrixArg<DTYPE> weightsR, const CudaMatrixArg<DTYPE> weightsI,
//    const CudaMatrixArg<DTYPE> nextActR, const CudaMatrixArg<DTYPE> nextActI, const CudaMatrixArg<DTYPE> nextError);

cudaError_t complexBackpropWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI,
    Matrix<DTYPE>& prevError, Matrix<DTYPE>& weightsR, Matrix<DTYPE>& weightsI, Matrix<DTYPE>& nextBiasR, Matrix<DTYPE>& nextBiasI,
    const Matrix<DTYPE>& nextActR, const Matrix<DTYPE>& nextActI, const Matrix<DTYPE>& nextError);

cudaError_t complexConvBackpropWithCuda(const Matrix3D<DTYPE>& prevActR, const Matrix3D<DTYPE>& prevActI,
    Matrix3D<DTYPE>& prevError, Matrix3D<DTYPE>& weightsR, Matrix3D<DTYPE>& weightsI, Matrix<DTYPE>& nextBiasR, Matrix<DTYPE>& nextBiasI,
    const Matrix3D<DTYPE>& nextActR, const Matrix3D<DTYPE> nextActI, const Matrix3D<DTYPE>& nextError);

#endif // CUDAFUNCS_CUH