
#ifndef CUDAFUNCS_CUH
#define CUDAFUNCS_CUH

#include "pmncudautils.cuh"
#include "cuda_runtime.h"

//__global__ void vecMatMultKernel(const CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>);
cudaError_t vecMatMultWithCuda(const CudaMatrix<DTYPE>& d_Ar, const CudaMatrix<DTYPE>& d_Ai, 
    const CudaMatrix<DTYPE>& d_Br, const CudaMatrix<DTYPE>& d_Bi, 
    CudaMatrix<DTYPE>& d_Cr, CudaMatrix<DTYPE>& d_Ci);

cudaError_t setValueWithCuda(CudaMatrix<DTYPE>& d_Mat, DTYPE value);

cudaError_t complexAddBiasWithCuda(CudaMatrix<DTYPE>& d_ActR, CudaMatrix<DTYPE>& d_ActI,
    const CudaMatrix<DTYPE>& d_BiasR, const CudaMatrix<DTYPE>& d_BiasI);

cudaError_t complexConvolutionWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI,
    CudaMatrix<DTYPE>* d_convR, CudaMatrix<DTYPE>* d_convI, const ConvParams& convParams,
    CudaMatrix<DTYPE>& d_nextActR, CudaMatrix<DTYPE>& d_nextActI);

cudaError_t complexAveragePoolWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI,
    const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextActR, CudaMatrix<DTYPE>& d_nextActI);

//__global__ void updateWeightsKernel(const CudaMatrixArg<DTYPE>, CudaMatrixArg<DTYPE>, const CudaMatrixArg<DTYPE>);
//cudaError_t updateWeightsWithCuda(const Matrix<DTYPE>& prevActs, Matrix<DTYPE>& weights, const Matrix<DTYPE>& nextError);


//__global__ void complexBackpropKernel(const CudaMatrixArg<DTYPE> prevActR, const CudaMatrixArg<DTYPE> prevActI,
//    CudaMatrixArg<DTYPE> prevError, const CudaMatrixArg<DTYPE> weightsR, const CudaMatrixArg<DTYPE> weightsI,
//    const CudaMatrixArg<DTYPE> nextActR, const CudaMatrixArg<DTYPE> nextActI, const CudaMatrixArg<DTYPE> nextError);

cudaError_t complexBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI,
    CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>& d_weightsR, CudaMatrix<DTYPE>& d_weightsI, CudaMatrix<DTYPE>& d_nextBiasR, CudaMatrix<DTYPE>& d_nextBiasI,
    const CudaMatrix<DTYPE>& d_nextActR, const CudaMatrix<DTYPE>& d_nextActI, const CudaMatrix<DTYPE>& d_nextError, float lrnRate);

cudaError_t complexConvBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI,
    CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>* d_weightsR, CudaMatrix<DTYPE>* d_weightsI, const ConvParams& convParams,
    const CudaMatrix<DTYPE>& d_nextActR, const CudaMatrix<DTYPE> d_nextActI, const CudaMatrix<DTYPE>& d_nextError, float lrnRate);

cudaError_t complexAvgPoolBackpropWithCuda(const CudaMatrix<DTYPE>& prevActR, const CudaMatrix<DTYPE>& prevActI, CudaMatrix<DTYPE>& prevError,
    const ConvParams& convParams, const CudaMatrix<DTYPE>& nextActR, const CudaMatrix<DTYPE> nextActI, const CudaMatrix<DTYPE>& nextError);

#endif // CUDAFUNCS_CUH