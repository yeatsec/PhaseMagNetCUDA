

#include "pmncudautils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudafuncs.cuh"

__device__ void d_cmp_mult(DTYPE ar, DTYPE ai, DTYPE br, DTYPE bi, DTYPE& cr, DTYPE& ci) {
	cr = (ar * br) - (ai * bi);
	ci = (ar * bi) + (br * ai);
}

__device__ DTYPE d_abs2(DTYPE r, DTYPE i) {
	return sqrt((r * r) + (i * i));
}

__device__ DTYPE d_ang2(const DTYPE r, const DTYPE i) { // -> [-pi, pi]
	return atan2f(i, r);
}

__device__ void d_phi_to_comp(DTYPE phi, DTYPE& r, DTYPE& i) {
	r = cos(phi);
	i = sin(phi);
}

__device__ const size_t getNumElems(const MatrixDim& mdim) {
	return mdim.adim * mdim.rdim * mdim.cdim;
}

__device__ void get_dLdMag_dLdPhi(DTYPE Yr, DTYPE Yi, DTYPE wxr, DTYPE wxi, DTYPE err, DTYPE& dLdMag, DTYPE& dLdPhi) {
	DTYPE dPhi = atan2f(wxi, wxr) - atan2f(Yi - wxi, Yr - wxr); // dPhi = ang(wx) - ang(Y-wx)
	DTYPE abswx = d_abs2(wxr, wxi);
	DTYPE absY = d_abs2(Yr, Yi);
	DTYPE abswx_Y = d_abs2(Yr - wxr, Yi - wxi);
	dLdMag = ((abswx + (abswx_Y * cosf(dPhi))) / absY) * err;
	dLdPhi = (-1.0f) * ((abswx * abswx_Y * sinf(dPhi)) / absY) * err; // radians
}


// define matrix multiplication kernel w/ corresponding helper function. transpose option on pre-multiply element
__global__ void vecMatMultKernel(const CudaMatrixArg<DTYPE> Ar, const CudaMatrixArg<DTYPE> Ai,
	const CudaMatrixArg<DTYPE> Br, const CudaMatrixArg<DTYPE> Bi, 
	CudaMatrixArg<DTYPE> Cr, CudaMatrixArg<DTYPE> Ci)
{
	// get block column
	// get thread column
	size_t col = threadIdx.x + (blockIdx.x * VEC_SIZE);

	DTYPE Cvalue_r = 0;
	DTYPE Cvalue_i = 0;

	size_t num_a_its = getNumElems(Ar.mdim) / VEC_SIZE;
	if (getNumElems(Ar.mdim) % VEC_SIZE != 0)
		++num_a_its;

	__shared__ DTYPE Ar_s[VEC_SIZE];
	__shared__ DTYPE Ai_s[VEC_SIZE];
	for (size_t a_it = 0; a_it < num_a_its; ++a_it) { // select block of A to do 
		size_t A_col = a_it * VEC_SIZE + threadIdx.x;
		if (A_col < getNumElems(Ar.mdim)) { // don't index A_col if past end
			Ar_s[threadIdx.x] = getElemFlatten(Ar, A_col);
			Ai_s[threadIdx.x] = getElemFlatten(Ai, A_col);
		}
		__syncthreads(); // all threads must load in A before we can use it to compute
		if (col < getNumElems(Cr.mdim)) { // don't continue if you're past the end of C
			for (size_t dot_it = 0; dot_it < VEC_SIZE; ++dot_it) {
				size_t b_row_ind = a_it * VEC_SIZE + dot_it;
				if (b_row_ind < getNumElems(Ar.mdim)) { // don't index into b past its row dimension. also implies As[dot_it] will not be accessed
					// multiply complex A[dot_it] with complex B[b_row_ind, col];
					DTYPE tempr, tempi;
					d_cmp_mult(Ar_s[dot_it], Ai_s[dot_it], getElem(Br, b_row_ind, col), getElem(Bi, b_row_ind, col), tempr, tempi);
					Cvalue_r += tempr;
					Cvalue_i += tempi;
				}
			}
		}
		__syncthreads();
	}
	if (col < getNumElems(Cr.mdim)) {
		setElemFlatten(Cr, col, Cvalue_r);
		setElemFlatten(Ci, col, Cvalue_i);
	}
}

// helper function for MatMul
cudaError_t vecMatMultWithCuda(const Matrix<DTYPE>& Ar, const Matrix<DTYPE>& Ai, 
	const Matrix<DTYPE>& Br, const Matrix<DTYPE>& Bi, 
	Matrix<DTYPE>& Cr, Matrix<DTYPE>& Ci) {
	// flattened A must have same size as b row
	assert(Ar.mdim.getNumElems() == Br.mdim.rdim && Br.mdim.cdim == Cr.mdim.getNumElems()); // check mult compat
	assert(Ar.mdim == Ai.mdim); // real and imag must have same shape
	assert(Br.mdim == Bi.mdim);
	assert(Cr.mdim == Ci.mdim);
	cudaError_t cudaStatus = cudaSuccess;

	// load A, B to device memory
	CudaMatrix<DTYPE> d_Ar(Ar);
	CudaMatrix<DTYPE> d_Ai(Ai);
	CudaMatrix<DTYPE> d_Br(Br);
	CudaMatrix<DTYPE> d_Bi(Bi);
	CudaMatrix<DTYPE> d_Cr(Cr.mdim);
	CudaMatrix<DTYPE> d_Ci(Ci.mdim);

	// invoke kernel; result of this operation will always be row vector
	unsigned int num_vecs = d_Cr.mdim.getNumElems() / VEC_SIZE;
	if (d_Cr.mdim.getNumElems() % VEC_SIZE != 0)
		++num_vecs;
	vecMatMultKernel <<<num_vecs, VEC_SIZE>>> (d_Ar.getCudaMatrixArg(), d_Ai.getCudaMatrixArg(),
		d_Br.getCudaMatrixArg(), d_Bi.getCudaMatrixArg(),
		d_Cr.getCudaMatrixArg(), d_Ci.getCudaMatrixArg());

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "vecMatMultKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// Copy output vector from GPU buffer to host memory.
	Cr.fillFromCuda(d_Cr);
	Ci.fillFromCuda(d_Ci);
	return cudaStatus;
}

__global__ void complexConvolutionKernel(const CudaMatrixArg<DTYPE> d_prevActR, const CudaMatrixArg<DTYPE> d_prevActI,
	const CudaMatrixArg<DTYPE> d_convR, const CudaMatrixArg<DTYPE> d_convI, const ConvParams convParams, const size_t filterNum,
	CudaMatrixArg<DTYPE> d_nextActR, CudaMatrixArg<DTYPE> d_nextActI) {
	/* <><><><><><><><> Hardcoded to work with stride=1 <><><><><><><><><> */
	// identify which block, which thread, and location in nextAct
	int ownNextActRow = threadIdx.y + (blockIdx.y * BLOCK_SIZE);
	int ownNextActCol = threadIdx.x + (blockIdx.x * BLOCK_SIZE);
	int prevRowB = d_prevActR.mdim.rdim;
	int prevColB = d_prevActR.mdim.cdim;
	bool inPrevAct = ownNextActRow < d_prevActR.mdim.rdim && ownNextActCol < d_prevActR.mdim.cdim;
	// fetch shared data
	const int sharedDim = BLOCK_SIZE + (2 * PAD); // == 20 x 20 == 400
	__shared__ DTYPE prevActR_s[sharedDim][sharedDim];
	__shared__ DTYPE prevActI_s[sharedDim][sharedDim];
	// fetch data from shared memory and synchronize
	// fetch data in own receptive field
	/* <><><><><><><><> Hardcoded for 1 color channel, 5x5 filter <><><><><><><><> */
	if (threadIdx.x < sharedDim/2 && threadIdx.y < sharedDim/2) {
		int prevActRow = ownNextActRow - PAD;
		int prevActCol = ownNextActCol - PAD;
		// get top left
		if (prevActRow >= 0 && prevActRow < prevRowB && prevActCol >= 0 && prevActCol < prevColB) {
			prevActR_s[threadIdx.y][threadIdx.x] = getElem(d_prevActR, prevActRow, prevActCol);
			prevActI_s[threadIdx.y][threadIdx.x] = getElem(d_prevActI, prevActRow, prevActCol);
		}
		else {
			prevActR_s[threadIdx.y][threadIdx.x] = 0;
			prevActI_s[threadIdx.y][threadIdx.x] = 0;
		}
		// get top right
		prevActCol += (sharedDim / 2);
		if (prevActRow >= 0 && prevActRow < prevRowB && prevActCol >= 0 && prevActCol < prevColB) {
			prevActR_s[threadIdx.y][threadIdx.x + (sharedDim/2)] = getElem(d_prevActR, prevActRow, prevActCol);
			prevActI_s[threadIdx.y][threadIdx.x + (sharedDim/2)] = getElem(d_prevActI, prevActRow, prevActCol);
		}
		else {
			prevActR_s[threadIdx.y][threadIdx.x + (sharedDim / 2)] = 0;
			prevActI_s[threadIdx.y][threadIdx.x + (sharedDim / 2)] = 0;
		}
		// get bottom right
		prevActRow += (sharedDim / 2);
		if (prevActRow >= 0 && prevActRow < prevRowB && prevActCol >= 0 && prevActCol < prevColB) {
			prevActR_s[threadIdx.y + (sharedDim/2)][threadIdx.x + (sharedDim/2)] = getElem(d_prevActR, prevActRow, prevActCol);
			prevActI_s[threadIdx.y + (sharedDim/2)][threadIdx.x + (sharedDim/2)] = getElem(d_prevActI, prevActRow, prevActCol);
		}
		else {
			prevActR_s[threadIdx.y + (sharedDim / 2)][threadIdx.x + (sharedDim / 2)] = 0;
			prevActI_s[threadIdx.y + (sharedDim / 2)][threadIdx.x + (sharedDim / 2)] = 0;
		}
		// get bottom left
		prevActCol -= (sharedDim / 2);
		if (prevActRow >= 0 && prevActRow < prevRowB && prevActCol >= 0 && prevActCol < prevColB) {
			prevActR_s[threadIdx.y + (sharedDim/2)][threadIdx.x] = getElem(d_prevActR, prevActRow, prevActCol);
			prevActI_s[threadIdx.y + (sharedDim/2)][threadIdx.x] = getElem(d_prevActI, prevActRow, prevActCol);
		}
		else {
			prevActR_s[threadIdx.y + (sharedDim/2)][threadIdx.x] = 0;
			prevActI_s[threadIdx.y + (sharedDim/2)][threadIdx.x] = 0;
		}
	}
	

	// fetch the filter (assumes dimensions 1xFILTER_DIMxFILTER_DIM)
	__shared__ DTYPE filterR_s[FILTER_DIM][FILTER_DIM];
	__shared__ DTYPE filterI_s[FILTER_DIM][FILTER_DIM];
	if (threadIdx.x < FILTER_DIM && threadIdx.y < FILTER_DIM) {
		filterR_s[threadIdx.y][threadIdx.x] = 0;// getElem(d_convR, threadIdx.y, threadIdx.x, filterNum);
		filterI_s[threadIdx.y][threadIdx.x] = 0;// getElem(d_convI, threadIdx.y, threadIdx.x, filterNum);
	}
	__syncthreads(); // shared input and weights have been fetched. compute the result
	if (ownNextActCol < d_nextActR.mdim.cdim && ownNextActRow < d_nextActR.mdim.rdim) { // only do this if result matters
		DTYPE dotValR = 0.0;
		DTYPE dotValI = 0.0;
		for (int f_row = 0; f_row < FILTER_DIM; ++f_row) {
			for (int f_col = 0; f_col < FILTER_DIM; ++f_col) {
				int s_row = threadIdx.y + f_row;
				int s_col = threadIdx.x + f_col;
				DTYPE weightR = filterR_s[f_row][f_col];
				DTYPE weightI = filterI_s[f_row][f_col];
				DTYPE actR = prevActR_s[s_row][s_col];
				DTYPE actI = prevActI_s[s_row][s_col];
				DTYPE resR, resI;
				d_cmp_mult(actR, actI, weightR, weightI, resR, resI);
				dotValR += resR;
				dotValI += resI;
			}
		}
		setElem(d_nextActR, ownNextActRow, ownNextActCol, dotValR, filterNum);
		setElem(d_nextActI, ownNextActRow, ownNextActCol, dotValI, filterNum); // set in correct filter position
	}
}

cudaError_t complexConvolutionWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI,
	Matrix<DTYPE>* convR, Matrix<DTYPE>* convI, const ConvParams& convParams,
	Matrix<DTYPE>& nextActR, Matrix<DTYPE>& nextActI) {
	assert(prevActR.mdim == prevActI.mdim && convR[0].mdim == convI[0].mdim && nextActR.mdim == nextActI.mdim);
	assert(convR[0].mdim.adim == prevActR.mdim.adim);

	CudaMatrix<DTYPE> d_prevActR(prevActR);
	CudaMatrix<DTYPE> d_prevActI(prevActI);
	CudaMatrix<DTYPE> d_nextActR(nextActR);
	CudaMatrix<DTYPE> d_nextActI(nextActI);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	size_t numRowIts = nextActR.mdim.rdim / BLOCK_SIZE;
	if (nextActR.mdim.rdim % BLOCK_SIZE != 0)
		++numRowIts;
	size_t numColIts = nextActR.mdim.cdim / BLOCK_SIZE;
	if (nextActR.mdim.cdim % BLOCK_SIZE != 0)
		++numColIts;
	dim3 gridDim(numColIts, numRowIts); // x, y

	cudaError_t cudaStatus(cudaSuccess);

	// loop through the activation maps (filters)
	// call 2D grid of activation map
	// filter is shared memory
	// subset of input is shared memory
	for (unsigned int filterNum = 0; filterNum < convParams.numFilters; ++filterNum) {
		CudaMatrix<DTYPE> d_convR(convR[filterNum]);
		CudaMatrix<DTYPE> d_convI(convI[filterNum]);
		
		complexConvolutionKernel <<< gridDim, blockDim >>> (d_prevActR.getCudaMatrixArg(), d_prevActI.getCudaMatrixArg(),
			d_convR.getCudaMatrixArg(), d_convI.getCudaMatrixArg(), convParams, filterNum,
			d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg());

		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "complexConvolutionKernel %ud launch failed: %s\n", filterNum, cudaGetErrorString(cudaStatus));
		}

	}

	nextActR.fillFromCuda(d_nextActR);
	nextActI.fillFromCuda(d_nextActI);

	return cudaStatus;
}

__global__ void complexAveragePoolKernel(const CudaMatrixArg<DTYPE> d_prevActR, const CudaMatrixArg<DTYPE> d_prevActI,
	const ConvParams convParams, const size_t actMap, CudaMatrixArg<DTYPE> d_nextActR, CudaMatrixArg<DTYPE> d_nextActI) {
	// where am i in d_nextAct?
	int nextActRow = threadIdx.y + (blockIdx.y * BLOCK_SIZE);
	int nextActCol = threadIdx.x + (blockIdx.x * BLOCK_SIZE);

	if (nextActRow < d_nextActR.mdim.rdim && nextActCol < d_nextActR.mdim.cdim) {
		// compute result, no shared memory would help
		int prevActRow = nextActRow * convParams.stride;
		int prevActCol = nextActCol * convParams.stride;
		// compute aggregate
		DTYPE resR = 0;
		DTYPE resI = 0;
		for (int f_row = prevActRow; f_row < prevActRow + convParams.stride; ++f_row) {
			for (int f_col = prevActCol; f_col < prevActCol + convParams.stride; ++f_col) {
				DTYPE actR = getElem(d_prevActR, f_row, f_col, actMap);
				DTYPE actI = getElem(d_prevActI, f_row, f_col, actMap);
				resR += actR;
				resI += actI;
			}
		}
		// compute average, 
		DTYPE denom = 1.0 / (DTYPE)(convParams.stride * convParams.stride);
		resR *= denom;
		resI *= denom;
		setElem(d_nextActR, nextActRow, nextActCol, resR, actMap);
		setElem(d_nextActI, nextActRow, nextActCol, resI, actMap);
	}
}

cudaError_t complexAveragePoolWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI, 
	const ConvParams& convParams, Matrix<DTYPE>& nextActR, Matrix<DTYPE>& nextActI) {
	assert(prevActR.mdim == prevActI.mdim);
	assert(nextActR.mdim == nextActI.mdim);
	assert(prevActR.mdim.adim == nextActR.mdim.adim); // same number of activation maps
	assert(prevActR.mdim.rdim / convParams.stride == nextActR.mdim.rdim); // downsample dim
	assert(prevActR.mdim.cdim / convParams.stride == nextActR.mdim.cdim); // downsample dim
	
	CudaMatrix<DTYPE> d_prevActR(prevActR);
	CudaMatrix<DTYPE> d_prevActI(prevActI);
	CudaMatrix<DTYPE> d_nextActR(nextActR);
	CudaMatrix<DTYPE> d_nextActI(nextActI);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	size_t numRowIts = nextActR.mdim.rdim / BLOCK_SIZE;
	if (nextActR.mdim.rdim % BLOCK_SIZE != 0)
		++numRowIts;
	size_t numColIts = nextActR.mdim.cdim / BLOCK_SIZE;
	if (nextActR.mdim.cdim % BLOCK_SIZE != 0)
		++numColIts;
	dim3 gridDim(numColIts, numRowIts); // x, y

	cudaError_t cudaStatus(cudaSuccess);

	for (unsigned int actMap = 0; actMap < convParams.numFilters; ++actMap) {
		complexAveragePoolKernel <<< gridDim, blockDim >>> (d_prevActR.getCudaMatrixArg(), d_prevActI.getCudaMatrixArg(), convParams,
			actMap, d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg());

		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "complexAvgPool %ud launch failed: %s\n", actMap, cudaGetErrorString(cudaStatus));
		}
	}

	nextActR.fillFromCuda(d_nextActR);
	nextActI.fillFromCuda(d_nextActI);

	return cudaStatus;
}

__global__ void updateWeightsKernel(const CudaMatrixArg<DTYPE> prevActs, CudaMatrixArg<DTYPE> weights, const CudaMatrixArg<DTYPE> nextError) {
	// get block column
	// get thread column
	size_t col = threadIdx.x + (blockIdx.x * VEC_SIZE); // col location on nextError
	__shared__ DTYPE prevActs_s[VEC_SIZE];
	unsigned int num_prevActs_its = prevActs.mdim.cdim / VEC_SIZE;
	if (prevActs.mdim.cdim % VEC_SIZE != 0)
		++num_prevActs_its;
	for (unsigned int prevActs_it = 0; prevActs_it < num_prevActs_its; ++prevActs_it) {
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
	assert(prevActs.mdim.getNumElems() == weights.mdim.rdim && prevActs.mdim.rdim == nextError.mdim.rdim && weights.mdim.cdim == nextError.mdim.cdim);

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

__global__ void complexBackpropKernel(const CudaMatrixArg<DTYPE> prevActR, const CudaMatrixArg<DTYPE> prevActI,
	CudaMatrixArg<DTYPE> prevError, CudaMatrixArg<DTYPE> weightsR, CudaMatrixArg<DTYPE> weightsI,
	const CudaMatrixArg<DTYPE> nextActR, const CudaMatrixArg<DTYPE> nextActI, const CudaMatrixArg<DTYPE> nextError) {

	unsigned int prevError_col = threadIdx.x + (blockIdx.x * VEC_SIZE); // column of prevError "owned" by the thread
	unsigned int num_nextError_its = getNumElems(nextError.mdim) / VEC_SIZE;
	DTYPE prevError_val = 0.0;
	if (getNumElems(nextError.mdim) % VEC_SIZE != 0) {
		++num_nextError_its;
	}
	// no need to reallocate shared each iteration
	__shared__ DTYPE nextError_s[VEC_SIZE];
	__shared__ DTYPE nextActR_s[VEC_SIZE];
	__shared__ DTYPE nextActI_s[VEC_SIZE];
	for (unsigned int nextError_it = 0; nextError_it < num_nextError_its; ++nextError_it) {
		unsigned int own_nextErr_col = nextError_it * VEC_SIZE + threadIdx.x;
		if (own_nextErr_col < getNumElems(nextError.mdim)) {
			nextError_s[threadIdx.x] = getElemFlatten(nextError, own_nextErr_col);
			nextActR_s[threadIdx.x] = getElemFlatten(nextActR, own_nextErr_col);
			nextActI_s[threadIdx.x] = getElemFlatten(nextActI, own_nextErr_col);
		}
		__syncthreads();
		// iterate through weights along row axis (nextError dim) to MAC into own prevError value with w(prevError_col, ...)*nextError(...)
		for (unsigned int dot_it = 0; dot_it < VEC_SIZE; ++dot_it) {
			unsigned int wgt_col = nextError_it * VEC_SIZE + dot_it;
			if (prevError_col < getNumElems(prevError.mdim) && wgt_col < getNumElems(nextError.mdim)) {
				DTYPE wr = getElem(weightsR, prevError_col, wgt_col);
				DTYPE wi = getElem(weightsI, prevError_col, wgt_col);
				DTYPE xr = getElemFlatten(prevActR, prevError_col);
				DTYPE xi = getElemFlatten(prevActI, prevError_col);
				DTYPE Yr = nextActR_s[dot_it];
				DTYPE Yi = nextActI_s[dot_it];
				DTYPE err = nextError_s[dot_it];
				DTYPE wxr, wxi, dLdMag, dLdPhi;
				d_cmp_mult(xr, xi, wr, wi, wxr, wxi);
				get_dLdMag_dLdPhi(Yr, Yi, wxr, wxi, err, dLdMag, dLdPhi);
				// propagate error back
				prevError_val += (d_abs2(wr, wi) * dLdMag);
				// adjust weight
				// apply LRN_RATE
				dLdMag *= (LRN_RATE * d_abs2(xr, xi));
				dLdPhi *= LRN_RATE;
				DTYPE dLdPhiR, dLdPhiI;
				d_phi_to_comp(dLdPhi, dLdPhiR, dLdPhiI);
				// multiply into result
				wr *= (dLdMag + 1.0);
				wi *= (dLdMag + 1.0);
				d_cmp_mult(wr, wi, dLdPhiR, dLdPhiI, wr, wi); // write back the rotation into the weights
				setElem(weightsR, prevError_col, wgt_col, wr);
				setElem(weightsI, prevError_col, wgt_col, wi);
			}
		}
		__syncthreads();
	}
	if (prevError_col < getNumElems(prevError.mdim)) {
		setElemFlatten(prevError, prevError_col, prevError_val);
	}
}

__global__ void complexUpdateBiasKernel(const CudaMatrixArg<DTYPE> actR, const CudaMatrixArg<DTYPE> actI,
	const CudaMatrixArg<DTYPE> error, CudaMatrixArg<DTYPE> biasR, CudaMatrixArg<DTYPE> biasI) {
	// each kernel is called on a subvector of activations. update bias according to its contribution to the activation
	// each thread is responsible for one activation + bias + error combination
	unsigned int col = threadIdx.x + (blockIdx.x * VEC_SIZE); // a & b * err combination
	if (col < getNumElems(biasR.mdim)) {
		DTYPE Yr = getElemFlatten(actR, col);
		DTYPE Yi = getElemFlatten(actI, col);
		DTYPE wr = getElemFlatten(biasR, col);
		DTYPE wi = getElemFlatten(biasI, col);
		DTYPE err = getElemFlatten(error, col);
		DTYPE dLdMag, dLdPhi;
		get_dLdMag_dLdPhi(Yr, Yi, wr, wi, err, dLdMag, dLdPhi);
		// apply LRN_RATE
		dLdMag *= LRN_RATE;
		dLdPhi *= LRN_RATE;
		DTYPE dLdPhiR, dLdPhiI;
		d_phi_to_comp(dLdPhi, dLdPhiR, dLdPhiI);
		// multiply into result
		wr *= (dLdMag + 1.0f);
		wi *= (dLdMag + 1.0f);
		d_cmp_mult(wr, wi, dLdPhiR, dLdPhiI, wr, wi); // write back the rotation into the weights
		setElemFlatten(biasR, col, wr);
		setElemFlatten(biasI, col, wi);
	}
}

// dPhi = ang(wx) - ang(Y-wx)
// dL/d|w| = lrn_rate * |x|(|wx| + |Y-wx|cos(dPhi))/|Y| * err
// dL/ang(w) = lrn_rate * (|wx||Y-wx|sin(dPhi))/|Y| * err
cudaError_t complexBackpropWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI, // if input layer, error can be used to create adv examples
	Matrix<DTYPE>& prevError, Matrix<DTYPE>& weightsR, Matrix<DTYPE>& weightsI, Matrix<DTYPE>& nextBiasR, Matrix<DTYPE>& nextBiasI,
	const Matrix<DTYPE>& nextActR, const Matrix<DTYPE>& nextActI, const Matrix<DTYPE>& nextError) {
	
	// check that the dimensions fit
	assert(prevActR.mdim == prevActI.mdim && prevActR.mdim == prevError.mdim); // prev parallel
	assert(nextActR.mdim == nextActI.mdim && nextActR.mdim == nextError.mdim); // next parallel
	assert(nextBiasR.mdim == nextBiasI.mdim && nextBiasR.mdim == nextActR.mdim); // next bias parallel
	assert(weightsR.mdim == weightsI.mdim); // weights parallel
	assert(prevActR.mdim.getNumElems() == weightsR.mdim.rdim && weightsR.mdim.cdim == nextActR.mdim.getNumElems()); // transfer

	// transfer data to device
	CudaMatrix<DTYPE> d_prevActR(prevActR);
	CudaMatrix<DTYPE> d_prevActI(prevActI);
	CudaMatrix<DTYPE> d_prevError(prevError);
	CudaMatrix<DTYPE> d_weightsR(weightsR);
	CudaMatrix<DTYPE> d_weightsI(weightsI);
	CudaMatrix<DTYPE> d_nextBiasR(nextBiasR);
	CudaMatrix<DTYPE> d_nextBiasI(nextBiasI);
	CudaMatrix<DTYPE> d_nextActR(nextActR);
	CudaMatrix<DTYPE> d_nextActI(nextActI);
	CudaMatrix<DTYPE> d_nextError(nextError);

	// compute next bias with the nextAct, nextBias, nextError
	unsigned int num_vecs = d_nextBiasR.mdim.getNumElems() / VEC_SIZE;
	if (d_nextBiasR.mdim.getNumElems() % VEC_SIZE != 0) {
		++num_vecs;
	}
	complexUpdateBiasKernel <<< num_vecs, VEC_SIZE >>> (d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg(),
		d_nextError.getCudaMatrixArg(), d_nextBiasR.getCudaMatrixArg(), d_nextBiasI.getCudaMatrixArg());

	cudaError_t cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "complexUpdateBiasKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// backpropagate error into prevError with prevAct, weights, nextAct, nextError, update weights
	num_vecs = d_prevError.mdim.getNumElems() / VEC_SIZE;
	if (d_prevError.mdim.getNumElems() % VEC_SIZE != 0) {
		++num_vecs;
	}
	complexBackpropKernel <<< num_vecs, VEC_SIZE >>> (d_prevActR.getCudaMatrixArg(), d_prevActI.getCudaMatrixArg(),
		d_prevError.getCudaMatrixArg(), d_weightsR.getCudaMatrixArg(), d_weightsI.getCudaMatrixArg(),
		d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg(), d_nextError.getCudaMatrixArg());

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "complexBackpropKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// push back to host CPU memory
	prevError.fillFromCuda(d_prevError);
	weightsR.fillFromCuda(d_weightsR);
	weightsI.fillFromCuda(d_weightsI);
	nextBiasR.fillFromCuda(d_nextBiasR);
	nextBiasI.fillFromCuda(d_nextBiasI);
	return cudaStatus;
}

cudaError_t complexConvBackpropWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI,
	Matrix<DTYPE>& prevError, Matrix<DTYPE>& weightsR, Matrix<DTYPE>& weightsI, const ConvParams& convParams,
	const Matrix<DTYPE>& nextActR, const Matrix<DTYPE> nextActI, const Matrix<DTYPE>& nextError) {
	// check that the dimensions fit
	assert(prevActR.mdim == prevActI.mdim && prevActR.mdim == prevError.mdim); // prev parallel
	assert(nextActR.mdim == nextActI.mdim && nextActR.mdim == nextError.mdim); // next parallel
	assert(weightsR.mdim == weightsI.mdim); // weights parallel
	assert(prevActR.mdim.getNumElems() == weightsR.mdim.rdim && weightsR.mdim.cdim == nextActR.mdim.getNumElems()); // transfer
	return cudaSuccess;
}

cudaError_t complexAvgPoolBackpropWithCuda(const Matrix<DTYPE>& prevActR, const Matrix<DTYPE>& prevActI, Matrix<DTYPE>& prevError,
	const ConvParams& convParams, const Matrix<DTYPE>& nextActR, const Matrix<DTYPE> nextActI, const Matrix<DTYPE>& nextError) {
	return cudaSuccess;
}