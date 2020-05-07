

#include "pmncudautils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// define matrix multiplication kernel w/ corresponding helper function. transpose option on pre-multiply element
__global__ void vecMatMultKernel(const CudaMatrixArg<DTYPE> A, const CudaMatrixArg<DTYPE> B, CudaMatrixArg<DTYPE> C)
{
	// get block column
	// get thread column
	size_t col = threadIdx.x + (blockIdx.x * VEC_SIZE);

	DTYPE Cvalue = 0;

	size_t num_a_its = A.mdim.cdim / VEC_SIZE;
	if (A.mdim.cdim % VEC_SIZE != 0)
		++num_a_its;

	for (size_t a_it = 0; a_it < num_a_its; ++a_it) { // select block of A to do 
		__shared__ DTYPE As[VEC_SIZE];
		size_t A_col = a_it * VEC_SIZE + threadIdx.x;
		if (A_col < A.mdim.cdim) // don't index A_col if past end
			As[threadIdx.x] = getElem(A, 0, A_col);
		__syncthreads(); // all threads must load in A before we can use it to compute
		if (col < C.mdim.cdim) { // don't continue if you're past the end of C
			for (size_t dot_it = 0; dot_it < VEC_SIZE; ++dot_it) {
				size_t b_row_ind = a_it * VEC_SIZE + dot_it;
				if (b_row_ind < A.mdim.cdim) // don't index into b past its row dimension. also implies As[dot_it] will not be accessed
					Cvalue += As[dot_it] * getElem(B, b_row_ind, col);
			}
		}
		__syncthreads();
	}

	setElem(C, 0, col, Cvalue);
}

// helper function for MatMul
cudaError_t vecMatMultWithCuda(const Matrix<DTYPE>& A, const Matrix<DTYPE>& B, Matrix<DTYPE>& C) {
	assert(A.mdim.cdim == B.mdim.rdim && A.mdim.rdim == C.mdim.rdim && B.mdim.cdim == C.mdim.cdim);
	cudaError_t cudaStatus = cudaSuccess;

	// load A, B to device memory
	CudaMatrix<DTYPE> d_A(A);
	CudaMatrix<DTYPE> d_B(B);

	// allocate C in device memory of size C
	CudaMatrix<DTYPE> d_C(C.mdim);

	// invoke kernel
	unsigned int num_vecs = d_C.mdim.cdim / VEC_SIZE;
	if (d_C.mdim.cdim % VEC_SIZE != 0)
		++num_vecs;
	vecMatMultKernel <<<num_vecs, VEC_SIZE>>> (d_A.getCudaMatrixArg(), d_B.getCudaMatrixArg(), d_C.getCudaMatrixArg());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "vecMatMultKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Copy output vector from GPU buffer to host memory.
	C.fillFromCuda(d_C);

	return cudaStatus;
}

__global__ void updateWeightsKernel(const CudaMatrixArg<DTYPE> prevActs, CudaMatrixArg<DTYPE> weights, const CudaMatrixArg<DTYPE> nextError) {
	// get block column
	// get thread column
	size_t col = threadIdx.x + (blockIdx.x * VEC_SIZE); // col location on nextError

	unsigned int num_prevActs_its = prevActs.mdim.cdim / VEC_SIZE;
	if (prevActs.mdim.cdim % VEC_SIZE != 0)
		++num_prevActs_its;

	for (unsigned int prevActs_it = 0; prevActs_it < num_prevActs_its; ++prevActs_it) {
		__shared__ DTYPE prevActs_s[VEC_SIZE];
		unsigned int prevActs_col = prevActs_it * VEC_SIZE + threadIdx.x;
		if (prevActs_col < prevActs.mdim.cdim)
			prevActs_s[threadIdx.x] = getElem(prevActs, 0, prevActs_col);
		__syncthreads(); // wait until shared memory is initialized before computation starts
		if (col < nextError.mdim.cdim) {
			for (unsigned int up_it = 0; up_it < VEC_SIZE; ++up_it) {
				unsigned int weights_row_ind = prevActs_it * VEC_SIZE + up_it;
				if (weights_row_ind < weights.mdim.rdim) {
					DTYPE weightVal = getElem(weights, weights_row_ind, col);
					weightVal += LRN_RATE * prevActs_s[up_it] * getElem(nextError, 0, col);
					setElem(weights, weights_row_ind, col, weightVal);
				}
			}
		}
		__syncthreads(); // finish computation before next sharedmem is loaded
	}

}

cudaError_t updateWeightsWithCuda(const Matrix<DTYPE>& prevActs, Matrix<DTYPE>& weights, const Matrix<DTYPE>& nextError) {

	// check that dimensions fit
	assert(prevActs.mdim.cdim == weights.mdim.rdim && prevActs.mdim.rdim == nextError.mdim.rdim && weights.mdim.cdim == nextError.mdim.cdim);

	CudaMatrix<DTYPE> d_prevActs(prevActs);
	CudaMatrix<DTYPE> d_weights(weights);
	CudaMatrix<DTYPE> d_nextError(nextError);

	unsigned int num_vecs = d_nextError.mdim.cdim / VEC_SIZE;
	if (d_nextError.mdim.cdim % VEC_SIZE != 0)
		++num_vecs;
	updateWeightsKernel <<< num_vecs, VEC_SIZE >>> (d_prevActs.getCudaMatrixArg(), d_weights.getCudaMatrixArg(), d_nextError.getCudaMatrixArg());

	cudaError_t cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "updateWeightsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	weights.fillFromCuda(d_weights);

	return cudaStatus; // is this checked?
}