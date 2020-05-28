
#ifndef CUDAFUNCS_CUH
#define CUDAFUNCS_CUH
#include "pmncudautils.cuh"
#include "cuda_runtime.h"

#define BLOCK_SIZE 8
#define VEC_SIZE 32

cudaError_t addSubAndClipWithCuda(const CudaMatrix<DTYPE>& d_A, const CudaMatrix<DTYPE>& d_B, CudaMatrix<DTYPE>& d_C, float eps, bool add, DTYPE clipMin = 0.0f, DTYPE clipMax = 1.0f);

cudaError_t setValueWithCuda(CudaMatrix<DTYPE>& d_Mat, DTYPE value);

cudaError_t complexConvolutionWithCuda(const CudaMatrix<DTYPE>& d_prevAct,
    CudaMatrix<DTYPE>* d_convR, CudaMatrix<DTYPE>* d_convI,
    const CudaMatrix<DTYPE>& d_convBias, const ConvParams& convParams,
    CudaMatrix<DTYPE>& d_nextActR, CudaMatrix<DTYPE>& d_nextActAng);


cudaError_t complexConvBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevAct,
    CudaMatrix<DTYPE>& d_prevError,  CudaMatrix<DTYPE>* d_weightsR, CudaMatrix<DTYPE>* d_weightsI, CudaMatrix<DTYPE>& d_bias, const ConvParams& convParams,
    const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextActAng, const CudaMatrix<DTYPE>& d_nextError, float lrnRate);

cudaError_t scalarConvolutionWithCuda(const CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>* d_conv, const CudaMatrix<DTYPE>& d_convBias,
    const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextAct, ActivationType actType);

cudaError_t scalarConvolutionBackpropWithCuda(CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>* d_conv, CudaMatrix<DTYPE>& d_convBias,
    const ConvParams& convParams, const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextError, float lrnRate, ActivationType actType);

cudaError_t scalarFCForwardPropWithCuda(const CudaMatrix<DTYPE>& d_opVec, const CudaMatrix<DTYPE>& d_opMat, const CudaMatrix<DTYPE>& d_bias, CudaMatrix<DTYPE>& d_resVec, ActivationType actType);

cudaError_t scalarAvgPoolWithCuda(const CudaMatrix<DTYPE>& d_prevAct, const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextAct, ActivationType actType);

cudaError_t scalarMaxPoolWithCuda(const CudaMatrix<DTYPE>& d_prevAct, const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextAct, ActivationType actType, curandState* curandptr = 0, float dropout = 0.0f);

cudaError_t scalarFCBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>& d_weights, const CudaMatrix<DTYPE>& d_nextAct,
    const CudaMatrix<DTYPE>& d_nextError, CudaMatrix<DTYPE>& d_nextBias, ActivationType actType, float lrnRate);

cudaError_t scalarAvgPoolBackpropWithCuda(CudaMatrix<DTYPE>& d_prevError, const ConvParams& convParams, const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextError,
    ActivationType actType);

cudaError_t scalarMaxPoolBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>& d_prevError, const ConvParams& convParams, const CudaMatrix<DTYPE>& d_nextAct, 
    const CudaMatrix<DTYPE>& d_nextError, ActivationType actType);

cudaError_t setupRNGWithCuda(const MatrixDim& mdim, curandState* state, unsigned long long seed);

#endif // CUDAFUNCS_CUH