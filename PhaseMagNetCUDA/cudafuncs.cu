

#include "pmncudautils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudafuncs.cuh"
#include <curand_kernel.h>
#include <assert.h>
#include <math.h>

constexpr auto ALPHA = 0.000f;
constexpr auto GRADIENT_CLIP = 1.0f;

void phi_to_comp(DTYPE phi, DTYPE& r, DTYPE& i) {
	r = cos(phi);
	i = sin(phi);
}

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
	r = cosf(phi);
	i = sinf(phi);
}

__device__ const size_t getNumElems(const MatrixDim& mdim) {
	return mdim.adim * mdim.rdim * mdim.cdim;
}

__device__ DTYPE reluFunc(DTYPE act) {
	return fmaxf(act, 0.0f);
}

__device__ DTYPE reluDerivFunc(DTYPE act) {
	if (act > 0.0f) {
		return ((DTYPE)1.0f);
	}
	else {
		return ((DTYPE)0.0f);
	}
}
__device__ DTYPE sigmoidFunc(DTYPE act) {
	return ((DTYPE)(1.0f / (1.0f + expf(-1.0f * act))));
}

__device__ DTYPE sigmoidDerivFunc(DTYPE act) {
	return ((DTYPE)((act) * (1.0f - act)));
}

/* Softmax + softmaxderiv funcs are just placeholders; the work is done by the processor */
__device__ DTYPE softmaxFunc(DTYPE act) {
	return act;
}

__device__ DTYPE softmaxDerivFunc(DTYPE act) {
	return 1.0f;
}

// static pointers to device functions
__device__ scalarActFunc p_reluFunc = reluFunc;
__device__ scalarActFunc p_reluDerivFunc = reluDerivFunc;
__device__ scalarActFunc p_sigmoidFunc = sigmoidFunc;
__device__ scalarActFunc p_sigmoidDerivFunc = sigmoidDerivFunc;
__device__ scalarActFunc p_softmaxFunc = softmaxFunc;
__device__ scalarActFunc p_softmaxDerivFunc = softmaxDerivFunc;


__device__ void get_dLdMag_dLdPhi(DTYPE Yr, DTYPE Yi, DTYPE wxr, DTYPE wxi, DTYPE errMag, DTYPE errAng, DTYPE& dLdMag, DTYPE& dLdPhi) {
	DTYPE abswx = d_abs2(wxr, wxi);
	DTYPE absY = d_abs2(Yr, Yi);
	DTYPE Y_wxr = Yr - wxr;
	DTYPE Y_wxi = Yi - wxi;
	DTYPE absY_wx = d_abs2(Y_wxr, Y_wxi);
	if (abswx > 0 && absY > 0) {
		// rotate wx by complex conjugate of Y_wx (used to determine sign of gradient in dmag/dang or dang/dmag cases)
		if (absY_wx > 0) {
			DTYPE invRotwxr, invRotwxi;
			d_cmp_mult(wxr, wxi, Y_wxr, -1.0f * Y_wxi, invRotwxr, invRotwxi);
			DTYPE absInvRotwx = d_abs2(invRotwxr, invRotwxi);
			
			dLdPhi = ((abswx * absY_wx * ((-invRotwxi) / absInvRotwx)) / absY) * errMag; // sign of invRotwxi is equal to dGamma/dang(wx)
			
		}
		else { // magnitude does not change wrt angle if there are no other phasors to add
			dLdPhi = 0;
		}
		dLdMag = ((abswx + (Y_wxr * (wxr / abswx)) + (Y_wxi * (wxi / abswx))) / absY) * errMag;
		
		//// gradient clipping here
		if (abs(dLdMag) > GRADIENT_CLIP) {
			dLdMag = copysignf(GRADIENT_CLIP, dLdMag);
		}
		if (abs(dLdPhi) > GRADIENT_CLIP) {
			dLdPhi = copysignf(GRADIENT_CLIP, dLdPhi);
		}
	}
	else { // ReLU-like discontinuity at zero activation
		dLdMag = 0;
		dLdPhi = 0;
	}
}

__device__ void scalarAdjustWeight(DTYPE& wgt, const DTYPE inp, const DTYPE err, const DTYPE lrnRate) {
	wgt += ((inp * err) - wgt * ALPHA) * lrnRate; // with L2 regularization
}

__device__ int getInd(MatrixDim mdim, int row, int col, int aisle = 0) {
	return (aisle * mdim.astride) + (row * mdim.rstride) + col;
}

__global__ void addAndClipKernel(const CudaMatrixArg<DTYPE> d_A, const CudaMatrixArg<DTYPE> d_B, CudaMatrixArg<DTYPE> d_C, float eps, DTYPE clipMin, DTYPE clipMax) {
	int flatInd = threadIdx.x + (blockIdx.x * blockDim.x);

	if (flatInd < getNumElems(d_A.mdim)) {
		DTYPE result = getElemFlatten(d_A, flatInd) + copysignf(eps, getElemFlatten(d_B, flatInd));
		if (result < clipMin)
			result = clipMin;
		if (result > clipMax)
			result = clipMax;
		setElemFlatten(d_C, flatInd, result);
	}
}

__global__ void subAndClipKernel(const CudaMatrixArg<DTYPE> d_A, const CudaMatrixArg<DTYPE> d_B, CudaMatrixArg<DTYPE> d_C, float eps, DTYPE clipMin, DTYPE clipMax) {
	int flatInd = threadIdx.x + (blockIdx.x * blockDim.x);

	if (flatInd < getNumElems(d_A.mdim)) {
		DTYPE result = getElemFlatten(d_A, flatInd) - copysignf(eps, getElemFlatten(d_B, flatInd));
		if (result < clipMin)
			result = clipMin;
		if (result > clipMax)
			result = clipMax;
		setElemFlatten(d_C, flatInd, result);
	}
}

cudaError_t addSubAndClipWithCuda(const CudaMatrix<DTYPE>& d_A, const CudaMatrix<DTYPE>& d_B, CudaMatrix<DTYPE>& d_C, float eps, bool add, DTYPE clipMin, DTYPE clipMax) {
	assert(d_A.mdim == d_B.mdim && d_A.mdim == d_C.mdim);
	assert(clipMin <= clipMax);

	cudaError_t cudaStatus(cudaSuccess);

	unsigned int numIts = d_A.mdim.getNumElems() / VEC_SIZE;
	if (numIts * VEC_SIZE < d_A.mdim.getNumElems())
		++numIts;
	if (add) {
		addAndClipKernel <<< numIts, VEC_SIZE >>> (d_A.getCudaMatrixArg(), d_B.getCudaMatrixArg(), d_C.getCudaMatrixArg(), eps, clipMin, clipMax);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addAndClipKernel launch failed! %s\n", cudaGetErrorString(cudaStatus));
		}
	}
	else { // this is redundant but it is simple
		subAndClipKernel <<< numIts, VEC_SIZE >>> (d_A.getCudaMatrixArg(), d_B.getCudaMatrixArg(), d_C.getCudaMatrixArg(), eps, clipMin, clipMax);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "subAndClipKernel launch failed! %s\n", cudaGetErrorString(cudaStatus));
		}
	}

	return cudaStatus;
}


__global__ void setValueKernel(CudaMatrixArg<DTYPE> d_Mat, DTYPE value) {
	const unsigned int colVal = threadIdx.x + (blockIdx.x * blockDim.x);
	if (colVal < getNumElems(d_Mat.mdim))
		setElemFlatten(d_Mat, colVal, value);
}

cudaError_t setValueWithCuda(CudaMatrix<DTYPE>& d_Mat, DTYPE value) {
	unsigned int numVecs = d_Mat.mdim.getNumElems() / VEC_SIZE;
	if (d_Mat.mdim.getNumElems() % VEC_SIZE != 0)
		++numVecs;
	setValueKernel <<< numVecs, VEC_SIZE >>> (d_Mat.getCudaMatrixArg(), value);

	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "setValueWithCudaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void complexConvolutionKernel(const CudaMatrixArg<DTYPE> d_prevAct, CudaMatrixArg<DTYPE>* d_convR, 
	CudaMatrixArg<DTYPE>* d_convI, const CudaMatrixArg<DTYPE> d_bias, const ConvParams convParams, CudaMatrixArg<DTYPE> d_nextAct, CudaMatrixArg<DTYPE> d_nextActAng) {
	extern __shared__ DTYPE s[];
	const int filterNum = blockIdx.z;
	const int sharedInputDim = blockDim.x; // == 20 x 20 == 400
	const int numElemsInput = sharedInputDim * sharedInputDim * d_prevAct.mdim.adim;
	DTYPE* prevAct_s = s;
	DTYPE* filterR_s = &(s[numElemsInput]);
	DTYPE* filterI_s = &(filterR_s[getNumElems(d_convR[filterNum].mdim)]);
	// identify which block, which thread, and location in nextAct
	const int ownNextActRow = threadIdx.y + (blockIdx.y * blockDim.y);
	const int ownNextActCol = threadIdx.x + (blockIdx.x * blockDim.x);
	const int flatInd = (threadIdx.y * blockDim.x) + threadIdx.x;
	const int prevRowB = d_prevAct.mdim.rdim;
	const int prevColB = d_prevAct.mdim.cdim;
	const int pad = convParams.pad;
	const int chanStride = sharedInputDim * sharedInputDim;
	// <><><><><><><><><> Hardcoded for stride of 1 <><><><><><><>
	// fetch actual shared input if your ownNextActRow/col is in bounds.
	for (int chan = 0; chan < d_prevAct.mdim.adim; ++chan) {
		if (ownNextActRow < prevRowB && ownNextActCol < prevColB) {
			prevAct_s[(chan * chanStride) + ((threadIdx.y) * sharedInputDim) + threadIdx.x] = getElem(d_prevAct, ownNextActRow, ownNextActCol, chan);
		}
		else {
			prevAct_s[(chan * chanStride) + ((threadIdx.y) * sharedInputDim) + threadIdx.x] = 0;
		}
	}
	

	// fetch the filter (assumes dimensions 1xFILTER_DIMxFILTER_DIM)
	if (threadIdx.x < convParams.filterDim.cdim && threadIdx.y < convParams.filterDim.rdim) {
		for (int chan = 0; chan < d_convR[filterNum].mdim.adim; ++chan) {
			filterR_s[getInd(d_convR[filterNum].mdim, threadIdx.y, threadIdx.x, chan)] = getElem(d_convR[filterNum], threadIdx.y, threadIdx.x, chan);
			filterI_s[getInd(d_convI[filterNum].mdim, threadIdx.y, threadIdx.x, chan)] = getElem(d_convI[filterNum], threadIdx.y, threadIdx.x, chan);
		}
	}
	__syncthreads(); // shared input and weights have been fetched. compute the result
	if (ownNextActCol < d_nextAct.mdim.cdim && ownNextActRow < d_nextAct.mdim.rdim) { // only do this if result matters
		DTYPE dotValR = 0.0;
		DTYPE dotValI = 0.0;
		for (int f_chan = 0; f_chan < convParams.filterDim.adim; ++f_chan) {
			for (int f_row = 0; f_row < convParams.filterDim.rdim; ++f_row) {
				for (int f_col = 0; f_col < convParams.filterDim.cdim; ++f_col) {
					int s_row = threadIdx.y + f_row - pad;
					int s_col = threadIdx.x + f_col - pad;
					int tmpRow = ownNextActRow - pad + f_row;
					int tmpCol = ownNextActCol - pad + f_col;
					DTYPE weightR = filterR_s[getInd(d_convR[filterNum].mdim, f_row, f_col, f_chan)];
					DTYPE weightI = filterI_s[getInd(d_convI[filterNum].mdim, f_row, f_col, f_chan)];
					DTYPE actR = 0; // default is to have no effect on dotVal
					DTYPE actI = 0;
					if (tmpRow >= 0 && tmpRow < d_prevAct.mdim.rdim && tmpCol >= 0 && tmpCol < d_prevAct.mdim.cdim) {
						DTYPE act;
						if (s_row >= 0 && s_row < sharedInputDim && s_col >= 0 && s_col < sharedInputDim) {
							act = prevAct_s[(f_chan * chanStride) + (s_row * sharedInputDim) + s_col];
						}
						else {
							act = getElem(d_prevAct, tmpRow, tmpCol, f_chan);
						}
						act *= PI;
						actR = cosf(act); // prevActR_s[(f_chan * chanStride) + (s_row * sharedInputDim) + s_col];// from f_col to s_col
						actI = sinf(act); // prevActI_s[(f_chan * chanStride) + (s_row * sharedInputDim) + s_col];
					}
					DTYPE resR, resI;
					d_cmp_mult(actR, actI, weightR, weightI, resR, resI);
					dotValR += resR;
					dotValI += resI;
				}
			}
		}
		// calculate the magnitude with bias and pass through activation function
		DTYPE dotValMag = d_abs2(dotValR, dotValI) + getElemFlatten(d_bias, filterNum);
		DTYPE dotValAng = d_ang2(dotValR, dotValI);
		setElem(d_nextAct, ownNextActRow, ownNextActCol, dotValMag, filterNum);
		setElem(d_nextActAng, ownNextActRow, ownNextActCol, dotValAng, filterNum);
	}
}

cudaError_t complexConvolutionWithCuda(const CudaMatrix<DTYPE>& d_prevAct,
	CudaMatrix<DTYPE>* d_convR, CudaMatrix<DTYPE>* d_convI,
	const CudaMatrix<DTYPE>& d_convBias, const ConvParams& convParams,
	CudaMatrix<DTYPE>& d_nextAct, CudaMatrix<DTYPE>& d_nextActAng) {
	assert(d_convR[0].mdim == d_convI[0].mdim);
	assert(d_convR[0].mdim.adim == d_prevAct.mdim.adim);
	assert(convParams.numFilters == d_nextAct.mdim.adim);

	cudaError_t cudaStatus(cudaSuccess);

	dim3 bDim(BLOCK_SIZE, BLOCK_SIZE, 1);
	unsigned int numRowIts = d_nextAct.mdim.rdim / BLOCK_SIZE;
	if (d_nextAct.mdim.rdim % BLOCK_SIZE != 0)
		++numRowIts;
	unsigned int numColIts = d_nextAct.mdim.cdim / BLOCK_SIZE;
	if (d_nextAct.mdim.cdim % BLOCK_SIZE != 0)
		++numColIts;
	dim3 gridDim(numColIts, numRowIts, d_nextAct.mdim.adim); // x, y, z

	const int sharedSize = ((BLOCK_SIZE) *
		(BLOCK_SIZE) * d_prevAct.mdim.adim // number of elems of shared input
		+ d_convR[0].mdim.getNumElems() * 2) * sizeof(DTYPE); // number of elems of convKernel in bytes, re, im

	CudaMatrixArg<DTYPE>* h_cmasR = new CudaMatrixArg<DTYPE>[convParams.numFilters];
	CudaMatrixArg<DTYPE>* h_cmasI = new CudaMatrixArg<DTYPE>[convParams.numFilters];
	CudaMatrixArg<DTYPE>* d_cmasR, *d_cmasI;
	for (int i = 0; i < convParams.numFilters; ++i) {
		h_cmasR[i] = d_convR[i].getCudaMatrixArg();
		h_cmasI[i] = d_convI[i].getCudaMatrixArg();
	}
	cudaMalloc(&d_cmasR, sizeof(CudaMatrixArg<DTYPE>) * convParams.numFilters);
	cudaMalloc(&d_cmasI, sizeof(CudaMatrixArg<DTYPE>) * convParams.numFilters);
	cudaMemcpy(d_cmasR, h_cmasR, sizeof(CudaMatrixArg<DTYPE>) * convParams.numFilters, cudaMemcpyHostToDevice);
	cudaMemcpy(d_cmasI, h_cmasI, sizeof(CudaMatrixArg<DTYPE>) * convParams.numFilters, cudaMemcpyHostToDevice);
	// loop through the activation maps (filters)
	// call 2D grid of activation map
	// filter is shared memory
	// subset of input is shared memory
	complexConvolutionKernel <<< gridDim, bDim, sharedSize >>> (d_prevAct.getCudaMatrixArg(),
		d_cmasR, d_cmasI, d_convBias.getCudaMatrixArg(), convParams,
		d_nextAct.getCudaMatrixArg(), d_nextActAng.getCudaMatrixArg());

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "complexConvolutionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		printf("complexConvKernel failed %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaFree(d_cmasR);
	cudaFree(d_cmasI);
	delete[] h_cmasR;
	delete[] h_cmasI;
	return cudaStatus;
}

__global__ void complexConvBackpropKernel(CudaMatrixArg<DTYPE> d_prevAct, CudaMatrixArg<DTYPE> d_prevError,
	CudaMatrixArg<DTYPE> d_weightsR, CudaMatrixArg<DTYPE> d_weightsI, CudaMatrixArg<DTYPE> d_bias,
	CudaMatrixArg<DTYPE> d_filterErrMag, CudaMatrixArg<DTYPE> d_filterErrPhi, const int actMap, const ConvParams convParams,
	CudaMatrixArg<DTYPE> d_nextAct, CudaMatrixArg<DTYPE> d_nextActAng, CudaMatrixArg<DTYPE> d_nextError) {
	extern __shared__ DTYPE s[];
	DTYPE* wgtR_s = s;
	DTYPE* wgtI_s = &(s[getNumElems(d_weightsR.mdim)]);
	DTYPE* wgtErrMag_s = &(s[2 * getNumElems(d_weightsR.mdim)]);
	DTYPE* wgtErrPhi_s = &(s[3 * getNumElems(d_weightsR.mdim)]);
	// where am i in prevAct?
	int prevFlatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	int astride = d_prevAct.mdim.astride;
	int rstride = d_prevAct.mdim.rstride;
	int prevAisleInd = prevFlatInd / astride;
	int prevRowInd = (prevFlatInd - (prevAisleInd * astride)) / rstride;
	int prevColInd = prevFlatInd - (prevAisleInd * astride) - (prevRowInd * rstride);
	// fetch the shared filter data if your prevFlatInd < filter.getNumElems()
	int numWgtElems = getNumElems(d_weightsR.mdim);
	int numWgtElemIts = numWgtElems / blockDim.x;
	if (numWgtElemIts * blockDim.x < numWgtElems)
		++numWgtElemIts;
	for (int i = 0; i < numWgtElemIts; ++i) {
		int fetchInd = threadIdx.x + (blockDim.x * i);
		if (fetchInd < numWgtElems) {
			wgtR_s[fetchInd] = getElemFlatten(d_weightsR, fetchInd);
			wgtI_s[fetchInd] = getElemFlatten(d_weightsI, fetchInd);
			// set 0 and atomic add later - don't have getElem on global memory race with writes from threads of other blocks
			wgtErrMag_s[fetchInd] = 0;
			wgtErrPhi_s[fetchInd] = 0;
		}
	}
	__syncthreads();
	// if location in prevAct is valid
	if (prevFlatInd < getNumElems(d_prevAct.mdim)) {
		DTYPE eAngPrev = 0;
		DTYPE x = PI * getElemFlatten(d_prevAct, prevFlatInd);
		DTYPE xR = cosf(x);
		DTYPE xI = sinf(x);
		// iterate through the filter to calculate the contributed error to prev the filter
		for (int fRow = 0; fRow < d_weightsR.mdim.rdim; ++fRow) {
			for (int fCol = 0; fCol < d_weightsR.mdim.cdim; ++fCol) {
				int nextRowInd = prevRowInd + (convParams.filterDim.rdim / 2) - fRow;
				int nextColInd = prevColInd + (convParams.filterDim.cdim / 2) - fCol;
				if (nextRowInd >= 0 && nextRowInd < d_nextAct.mdim.rdim && nextColInd >= 0 && nextColInd < d_nextAct.mdim.cdim) {
					// use aisle of prev to select filter aisle
					DTYPE wgtR = wgtR_s[getInd(d_weightsR.mdim, fRow, fCol, prevAisleInd)];
					DTYPE wgtI = wgtI_s[getInd(d_weightsI.mdim, fRow, fCol, prevAisleInd)];
					DTYPE errMag = getElem(d_nextError, nextRowInd, nextColInd, actMap); // error aisle corresponds to actMap
					DTYPE errAng = 0.0f;
					DTYPE Y = getElem(d_nextAct, nextRowInd, nextColInd, actMap) - getElemFlatten(d_bias, actMap); // correct for bias
					DTYPE Yang = getElem(d_nextActAng, nextRowInd, nextColInd, actMap);
					DTYPE YR = Y * cosf(Yang);
					DTYPE YI = Y * sinf(Yang);
					// calculate gradient and atomic add to shared ErrMag, ErrPhi, add to local eMagPrev
					DTYPE wxr, wxi;
					d_cmp_mult(xR, xI, wgtR, wgtI, wxr, wxi);
					DTYPE dLdMag, dLdPhi;
					get_dLdMag_dLdPhi(YR, YI, wxr, wxi, errMag, errAng, dLdMag, dLdPhi);
					eAngPrev += dLdPhi;
					atomicAdd(&(wgtErrMag_s[getInd(d_filterErrMag.mdim, fRow, fCol, prevAisleInd)]), dLdMag);
					atomicAdd(&(wgtErrPhi_s[getInd(d_filterErrPhi.mdim, fRow, fCol, prevAisleInd)]), dLdPhi);
				}
			}
		}
		eAngPrev /= PI; // in radians
		eAngPrev += getElemFlatten(d_prevError, prevFlatInd);
		setElemFlatten(d_prevError, prevFlatInd, eAngPrev);
	}
	__syncthreads();
	if (threadIdx.x < getNumElems(d_weightsR.mdim)) {
		atomicAdd(&(d_filterErrMag.data[threadIdx.x]), wgtErrMag_s[threadIdx.x]); // can potentially race with threads from other blocks of same filter
		atomicAdd(&(d_filterErrPhi.data[threadIdx.x]), wgtErrPhi_s[threadIdx.x]);
	}
}

__global__ void complexUpdateKernelWeights(CudaMatrixArg<DTYPE> d_weightsR, CudaMatrixArg<DTYPE> d_weightsI,
	CudaMatrixArg<DTYPE> d_filterErrMag, CudaMatrixArg<DTYPE> d_filterErrPhi, const float lrnRate) {
	const int kRow = threadIdx.y + (blockIdx.y * blockDim.y);
	const int kCol = threadIdx.x + (blockIdx.x * blockDim.x);
	const int kAisle = threadIdx.z + (blockIdx.z * blockDim.z);
	if (kRow < d_weightsR.mdim.rdim && kCol < d_weightsR.mdim.cdim && kAisle < d_weightsR.mdim.adim) {
		DTYPE wR = getElem(d_weightsR, kRow, kCol, kAisle);
		DTYPE wI = getElem(d_weightsI, kRow, kCol, kAisle);
		DTYPE eMag = getElem(d_filterErrMag, kRow, kCol, kAisle);
		DTYPE ePhi = getElem(d_filterErrPhi, kRow, kCol, kAisle) * lrnRate; // learn rate is applied to ePhi
		DTYPE ePhiR, ePhiI;
		d_phi_to_comp(ePhi, ePhiR, ePhiI);
		// stretch weights by dMag
		// CHANGE - scale magnitude change by current magnitude
		DTYPE absw = d_abs2(wR, wI);
		if (absw > 0.0f) {
			wR += ((eMag - ALPHA * absw) * (wR / absw)) * lrnRate;
			wI += ((eMag - ALPHA * absw) * (wI / absw)) * lrnRate;
			// rotate weights by dPhi
			d_cmp_mult(wR, wI, ePhiR, ePhiI, wR, wI);
			// write back
			setElem(d_weightsR, kRow, kCol, wR, kAisle);
			setElem(d_weightsI, kRow, kCol, wI, kAisle);
		}
	}
}

__global__ void updateKernelBias(const CudaMatrixArg<DTYPE> d_nextError, CudaMatrixArg<DTYPE> d_bias, const int filterNum, float lrnRate) {
	extern __shared__ DTYPE s[]; // size of s is blockDim.x
	unsigned int astride = d_nextError.mdim.astride;
	s[threadIdx.x] = 0; // calculating sum
	for (unsigned int fetchInd = threadIdx.x + (astride * filterNum); fetchInd < (filterNum + 1) * astride; fetchInd += blockDim.x) {
		s[threadIdx.x] += getElemFlatten(d_nextError, fetchInd);
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		// sum over the shared
		DTYPE errSum = 0;
		for (unsigned int i = 0; i < blockDim.x; ++i) {
			errSum += s[i];
		}
		DTYPE bias = getElemFlatten(d_bias, filterNum);
		scalarAdjustWeight(bias, 1.0f, errSum, lrnRate); // / ((DTYPE)astride), lrnRate);
		setElemFlatten(d_bias, filterNum, bias);
	}
}

cudaError_t complexConvBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevAct,
	CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>* d_weightsR, CudaMatrix<DTYPE>* d_weightsI, CudaMatrix<DTYPE>& d_bias, const ConvParams& convParams,
	const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextActAng, const CudaMatrix<DTYPE>& d_nextError, float lrnRate) {
	// check that the dimensions fit
	assert(d_prevAct.mdim == d_prevError.mdim); // prev parallel
	assert(d_nextError.mdim == d_nextAct.mdim);
	assert(d_nextAct.mdim == d_nextActAng.mdim); // next parallel
	assert(d_weightsR[0].mdim == d_weightsI[0].mdim); // weights parallel

	cudaError_t cudaStatus(cudaSuccess);

	const unsigned int linearBlock = BLOCK_SIZE * BLOCK_SIZE;
	unsigned int numLinearBlocks = d_prevAct.mdim.getNumElems() / linearBlock;
	if (d_prevAct.mdim.getNumElems() % linearBlock != 0) {
		++numLinearBlocks;
	}

	CudaMatrix<DTYPE> d_filterErrMag(convParams.filterDim);
	CudaMatrix<DTYPE> d_filterErrPhi(convParams.filterDim);

	const unsigned int sharedSize = 4 * d_weightsR[0].mdim.size;
	
	// dynamic shared memory is cumulative mag/phi error for the filter
	for (unsigned int actMap = 0; actMap < convParams.numFilters; ++actMap) {
		// global scratchpad memory for filter error
		setValueWithCuda(d_filterErrMag, 0);
		setValueWithCuda(d_filterErrPhi, 0);

		complexConvBackpropKernel <<< numLinearBlocks, linearBlock, sharedSize >>>
			(d_prevAct.getCudaMatrixArg(), d_prevError.getCudaMatrixArg(), 
				d_weightsR[actMap].getCudaMatrixArg(), d_weightsI[actMap].getCudaMatrixArg(), d_bias.getCudaMatrixArg(),
				d_filterErrMag.getCudaMatrixArg(), d_filterErrPhi.getCudaMatrixArg(), actMap, convParams, 
				d_nextAct.getCudaMatrixArg(), d_nextActAng.getCudaMatrixArg(), 
				d_nextError.getCudaMatrixArg());

		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "complexConvBackprop launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		/*Matrix<DTYPE> hErrMag(d_filterErrMag);
		Matrix<DTYPE> hErrPhi(d_filterErrPhi);
		printf("Filter %d ErrorMag: %5.5f\n", actMap, hErrMag.getElem(2, 0, 0));
		printf("Filter %d ErrorPhi: %5.5f\n", actMap, hErrPhi.getElem(2, 0, 0));*/
		MatrixDim fDim(convParams.filterDim);
		// filter error needs to be reincorporated into weights
		complexUpdateKernelWeights <<< dim3(1, 1, fDim.adim), dim3(fDim.cdim, fDim.rdim, 1) >>> (d_weightsR[actMap].getCudaMatrixArg(),
			d_weightsI[actMap].getCudaMatrixArg(), d_filterErrMag.getCudaMatrixArg(), d_filterErrPhi.getCudaMatrixArg(), lrnRate);
		
		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "complexUpdateKernelWeights launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		updateKernelBias <<< 1, VEC_SIZE, VEC_SIZE * sizeof(DTYPE) >>> (d_nextError.getCudaMatrixArg(), d_bias.getCudaMatrixArg(), actMap, lrnRate);

	}

	return cudaStatus;
}

__global__ void scalarActFuncKernel(CudaMatrixArg<DTYPE> d_act, scalarActFunc actFunc) {
	unsigned int flatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	if (flatInd < getNumElems(d_act.mdim)) {
		DTYPE res = (*actFunc)(getElemFlatten(d_act, flatInd));
		setElemFlatten(d_act, flatInd, res);
	}
}

__global__ void scalarActDerivFuncKernel(const CudaMatrixArg<DTYPE> d_act, CudaMatrixArg<DTYPE> d_err, scalarActFunc actDerivFunc) {
	unsigned int flatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	if (flatInd < getNumElems(d_act.mdim)) {
		DTYPE deriv = (*actDerivFunc)(getElemFlatten(d_act, flatInd));
		DTYPE product = deriv * getElemFlatten(d_err, flatInd);
		setElemFlatten(d_err, flatInd, product);
	}
}

__global__ void scalarConvolutionKernel(CudaMatrixArg<DTYPE> d_prevAct, CudaMatrixArg<DTYPE>* d_conv,
	CudaMatrixArg<DTYPE> d_bias, ConvParams convParams, CudaMatrixArg<DTYPE> d_nextAct, scalarActFunc actFunc) {
	extern __shared__ DTYPE s[];
	const int filterNum = blockIdx.z;
	const int sharedInputDim = blockDim.x; // == 20 x 20 == 400
	const int numElemsInput = sharedInputDim * sharedInputDim * d_prevAct.mdim.adim;
	DTYPE* prevAct_s = s; // will implement shared memory later
	DTYPE* filter_s = &(s[numElemsInput]);
	// identify which block, which thread, and location in nextAct
	const int ownNextActRow = threadIdx.y + (blockIdx.y * blockDim.y);
	const int ownNextActCol = threadIdx.x + (blockIdx.x * blockDim.x);
	const int prevRowB = d_prevAct.mdim.rdim;
	const int prevColB = d_prevAct.mdim.cdim;
	const int pad = convParams.pad;
	const int chanStride = sharedInputDim * sharedInputDim;
	const MatrixDim convMdim(d_conv[filterNum].mdim);
	for (int chan = 0; chan < d_prevAct.mdim.adim; ++chan) {
		if (ownNextActRow < prevRowB && ownNextActCol < prevColB) {
			prevAct_s[(chan * chanStride) + ((threadIdx.y) * sharedInputDim) + threadIdx.x] = getElem(d_prevAct, ownNextActRow, ownNextActCol, chan);
		}
		else {
			prevAct_s[(chan * chanStride) + ((threadIdx.y) * sharedInputDim) + threadIdx.x] = 0;
		}
	}

	// fetch the filter (assumes dimensions 1xFILTER_DIMxFILTER_DIM)
	if (threadIdx.x < convParams.filterDim.cdim && threadIdx.y < convParams.filterDim.rdim) {
		for (int chan = 0; chan < convMdim.adim; ++chan) {
			filter_s[getInd(convMdim, threadIdx.y, threadIdx.x, chan)] = getElem(d_conv[filterNum], threadIdx.y, threadIdx.x, chan);
		}
	}
	__syncthreads(); // shared input and weights have been fetched. compute the result
	if (ownNextActCol < d_nextAct.mdim.cdim && ownNextActRow < d_nextAct.mdim.rdim) { // only do this if result matters
		DTYPE dotVal = 0.0;
		for (int f_chan = 0; f_chan < convParams.filterDim.adim; ++f_chan) {
			for (int f_row = 0; f_row < convParams.filterDim.rdim; ++f_row) {
				for (int f_col = 0; f_col < convParams.filterDim.cdim; ++f_col) {
					int s_row = threadIdx.y + f_row - pad;
					int s_col = threadIdx.x + f_col - pad;
					int tmpRow = ownNextActRow - pad + f_row;
					int tmpCol = ownNextActCol - pad + f_col;
					DTYPE weight = filter_s[getInd(convMdim, f_row, f_col, f_chan)];
					DTYPE act = 0;
					if (tmpRow >= 0 && tmpRow < d_prevAct.mdim.rdim && tmpCol >= 0 && tmpCol < d_prevAct.mdim.cdim) { // is it inside input
						if (s_row >= 0 && s_row < sharedInputDim && s_col >= 0 && s_col < sharedInputDim) { // is it inside shared
							act = prevAct_s[(f_chan * chanStride) + (s_row * sharedInputDim) + s_col];
						}
						else {
							act = getElem(d_prevAct, tmpRow, tmpCol, f_chan);
						}
					}
					dotVal += weight * act;
				}
			}
		}
		// calculate the magnitude with bias and pass through activation function
		dotVal += getElemFlatten(d_bias, filterNum);
		dotVal = (*actFunc)(dotVal);
		setElem(d_nextAct, ownNextActRow, ownNextActCol, dotVal, filterNum);
	}
}

cudaError_t scalarConvolutionWithCuda(const CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>* d_conv, const CudaMatrix<DTYPE>& d_convBias,
	const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextAct, ActivationType actType) {
	assert(d_conv[0].mdim.adim == d_prevAct.mdim.adim);
	assert(convParams.numFilters == d_nextAct.mdim.adim);

	cudaError_t cudaStatus(cudaSuccess);

	scalarActFunc actFunc;

	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_sigmoidFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_reluFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_softmaxFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actfunc not yet implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol scalar conv failed! %s\n", cudaGetErrorString(cudaStatus));
	}

	CudaMatrixArg<DTYPE>* h_cmas = new CudaMatrixArg<DTYPE>[convParams.numFilters];
	CudaMatrixArg<DTYPE>* d_cmas;
	for (int i = 0; i < convParams.numFilters; ++i) {
		h_cmas[i] = d_conv[i].getCudaMatrixArg();
	}
	cudaMalloc(&d_cmas, sizeof(CudaMatrixArg<DTYPE>) * convParams.numFilters);
	cudaMemcpy(d_cmas, h_cmas, sizeof(CudaMatrixArg<DTYPE>) * convParams.numFilters, cudaMemcpyHostToDevice);

	dim3 bDim(BLOCK_SIZE, BLOCK_SIZE, 1);
	unsigned int numRowIts = d_nextAct.mdim.rdim / BLOCK_SIZE;
	if (d_nextAct.mdim.rdim % BLOCK_SIZE != 0)
		++numRowIts;
	unsigned int numColIts = d_nextAct.mdim.cdim / BLOCK_SIZE;
	if (d_nextAct.mdim.cdim % BLOCK_SIZE != 0)
		++numColIts;
	dim3 gridDim(numColIts, numRowIts, convParams.numFilters); // x, y
	
	const int sharedSize = ((BLOCK_SIZE) *
		(BLOCK_SIZE) * d_prevAct.mdim.adim // number of elems of shared input
		+ d_conv[0].mdim.getNumElems()) * sizeof(DTYPE); // number of elems of convKernel in bytes, re only
	// loop through the activation maps (filters)
	// call 2D grid of activation map
	// filter is shared memory
	// subset of input is shared memory

	scalarConvolutionKernel <<< gridDim, bDim, sharedSize >>> (d_prevAct.getCudaMatrixArg(),
		d_cmas, d_convBias.getCudaMatrixArg(), convParams,
		d_nextAct.getCudaMatrixArg(), actFunc);

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "scalarConvolutionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaFree(d_cmas);
	delete[] h_cmas;

	return cudaStatus;
}

__global__ void scalarConvBackpropKernel(CudaMatrixArg<DTYPE> d_prevAct, CudaMatrixArg<DTYPE> d_prevError,
	CudaMatrixArg<DTYPE> d_weights, CudaMatrixArg<DTYPE> d_bias,
	CudaMatrixArg<DTYPE> d_filterErr, const int actMap, const ConvParams convParams,
	CudaMatrixArg<DTYPE> d_nextAct, CudaMatrixArg<DTYPE> d_nextError) {
	extern __shared__ DTYPE s[];
	DTYPE* wgt_s = s;
	DTYPE* wgtErr_s = &(s[getNumElems(d_weights.mdim)]);
	// where am i in prevAct?
	int prevFlatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	int astride = d_prevError.mdim.astride;
	int rstride = d_prevError.mdim.rstride;
	int prevAisleInd = prevFlatInd / astride;
	int prevRowInd = (prevFlatInd - (prevAisleInd * astride)) / rstride;
	int prevColInd = prevFlatInd - (prevAisleInd * astride) - (prevRowInd * rstride);
	// fetch the shared filter data if your prevFlatInd < filter.getNumElems()
	int numWgtElems = getNumElems(d_weights.mdim);
	int numWgtElemIts = numWgtElems / blockDim.x;
	if (numWgtElemIts * blockDim.x < numWgtElems)
		++numWgtElemIts;
	for (int i = 0; i < numWgtElemIts; ++i) {
		int fetchInd = threadIdx.x + (blockDim.x * i);
		if (fetchInd < numWgtElems) {
			wgt_s[fetchInd] = getElemFlatten(d_weights, fetchInd);
			wgtErr_s[fetchInd] = 0; // rather than get element here (can race with other blocks), set 0 here and atomic add later
		}
	}
	__syncthreads();
	// if location in prevAct is valid
	if (prevFlatInd < getNumElems(d_prevError.mdim)) {
		DTYPE errPrev = 0;
		DTYPE x = getElemFlatten(d_prevAct, prevFlatInd);
		// iterate through the filter to calculate the contributed error to prev the filter
		for (int fRow = 0; fRow < d_weights.mdim.rdim; ++fRow) {
			for (int fCol = 0; fCol < d_weights.mdim.cdim; ++fCol) {
				int nextRowInd = prevRowInd + (convParams.filterDim.rdim / 2) - fRow;
				int nextColInd = prevColInd + (convParams.filterDim.cdim / 2) - fCol;
				if (nextRowInd >= 0 && nextRowInd < d_nextAct.mdim.rdim && nextColInd >= 0 && nextColInd < d_nextAct.mdim.cdim) {
					// use aisle of prev to select filter aisle
					DTYPE wgt = wgt_s[getInd(d_weights.mdim, fRow, fCol, prevAisleInd)];
					DTYPE err = getElem(d_nextError, nextRowInd, nextColInd, actMap); // error aisle corresponds to actMap
					errPrev += wgt * err;
					DTYPE errWgt = x * err;
					atomicAdd(&(wgtErr_s[getInd(d_filterErr.mdim, fRow, fCol, prevAisleInd)]), errWgt); // races with threads in same block
				}
			}
		}
		errPrev += getElemFlatten(d_prevError, prevFlatInd);
		setElemFlatten(d_prevError, prevFlatInd, errPrev);
	}
	__syncthreads();
	if (threadIdx.x < getNumElems(d_weights.mdim)) {
		atomicAdd(&(d_filterErr.data[threadIdx.x]), wgtErr_s[threadIdx.x]);
	}
}

__global__ void scalarUpdateKernelWeights(CudaMatrixArg<DTYPE> d_weights, 
	CudaMatrixArg<DTYPE> d_filterErr, const float lrnRate) {
	const int kRow = threadIdx.y + (blockDim.y * blockIdx.y);
	const int kCol = threadIdx.x + (blockDim.x * blockIdx.x);
	const int kAisle = threadIdx.z + (blockDim.z * blockIdx.z);
	if (kRow < d_weights.mdim.rdim && kCol < d_weights.mdim.cdim && kAisle < d_weights.mdim.adim) {
		DTYPE w = getElem(d_weights, kRow, kCol, kAisle);
		DTYPE err = getElem(d_filterErr, kRow, kCol, kAisle);

		scalarAdjustWeight(w, 1.0f, err, lrnRate); // input has already been multiplied into error
		// write back
		setElem(d_weights, kRow, kCol, w, kAisle);
	}
}

cudaError_t scalarConvolutionBackpropWithCuda(CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>* d_conv, CudaMatrix<DTYPE>& d_convBias,
	const ConvParams& convParams, const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextError, float lrnRate, ActivationType actType) {
	// check that the dimensions fit
	assert(d_nextError.mdim == d_nextAct.mdim);
	assert(d_conv[0].mdim.adim == d_prevError.mdim.adim && convParams.numFilters == d_nextAct.mdim.adim);

	cudaError_t cudaStatus(cudaSuccess);

	scalarActFunc actDerivFunc;


	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_sigmoidDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_reluDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_softmaxDerivFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actDerivfunc not implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol reluDeriv failed! %s\n", cudaGetErrorString(cudaStatus));
	}
	unsigned int numFlatIts = d_nextAct.mdim.getNumElems() / VEC_SIZE;
	if (numFlatIts * VEC_SIZE < d_nextAct.mdim.getNumElems())
		++numFlatIts;

	// calculate activation function derivative of nextError
	scalarActDerivFuncKernel <<< numFlatIts, VEC_SIZE >>> (d_nextAct.getCudaMatrixArg(), d_nextError.getCudaMatrixArg(), actDerivFunc);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarActFuncKerel FCBackprop launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	const unsigned int linearBlock = BLOCK_SIZE * BLOCK_SIZE;
	unsigned int numLinearBlocks = d_prevError.mdim.getNumElems() / linearBlock;
	if (d_prevError.mdim.getNumElems() % linearBlock != 0) {
		++numLinearBlocks;
	}

	CudaMatrix<DTYPE> d_filterErr(convParams.filterDim);

	const unsigned int sharedSize = 2 * d_conv[0].mdim.size;

	// dynamic shared memory is cumulative mag/phi error for the filter
	for (unsigned int actMap = 0; actMap < convParams.numFilters; ++actMap) {
		// global scratchpad memory for filter error
		setValueWithCuda(d_filterErr, 0);

		scalarConvBackpropKernel <<< numLinearBlocks, linearBlock, sharedSize >>>
			(d_prevAct.getCudaMatrixArg(), d_prevError.getCudaMatrixArg(),
				d_conv[actMap].getCudaMatrixArg(), d_convBias.getCudaMatrixArg(),
				d_filterErr.getCudaMatrixArg(), actMap, convParams,
				d_nextAct.getCudaMatrixArg(), d_nextError.getCudaMatrixArg());

		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "scalarConvBackprop launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		
		MatrixDim fDim(convParams.filterDim);
		// filter error needs to be reincorporated into weights
		scalarUpdateKernelWeights <<< dim3(1, 1, fDim.adim), dim3(fDim.cdim, fDim.rdim, 1) >>> (d_conv[actMap].getCudaMatrixArg(),
			d_filterErr.getCudaMatrixArg(), lrnRate);

		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "scalarUpdateKernelWeights launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		updateKernelBias <<< 1, VEC_SIZE, VEC_SIZE * sizeof(DTYPE) >>> (d_nextError.getCudaMatrixArg(), d_convBias.getCudaMatrixArg(), actMap, lrnRate);

	}

	return cudaStatus;
}

__global__ void scalarVecMatMulKernel(const CudaMatrixArg<DTYPE> d_opVec, const CudaMatrixArg<DTYPE> d_opMat, CudaMatrixArg<DTYPE> d_resVec) {
	
	unsigned int resVecInd = threadIdx.x + (VEC_SIZE * blockIdx.x);
	
	// iterate through the opVec, fetch shared memory
	unsigned int numOpVecIts = getNumElems(d_opVec.mdim) / VEC_SIZE;
	if (numOpVecIts * VEC_SIZE < getNumElems(d_opVec.mdim))
		++numOpVecIts;

	__shared__ DTYPE d_opVec_s[VEC_SIZE];

	DTYPE resVal = 0;

	for (unsigned int opVecIt = 0; opVecIt < numOpVecIts; ++opVecIt) {
		unsigned int fetchOpVecInd = threadIdx.x + (VEC_SIZE * opVecIt);
		// fetch shared data
		if (fetchOpVecInd < getNumElems(d_opVec.mdim)) {
			d_opVec_s[threadIdx.x] = getElemFlatten(d_opVec, fetchOpVecInd);
		}

		__syncthreads();

		// compute dot product and add to result
		for (unsigned int dot_it = 0; dot_it < VEC_SIZE; ++dot_it) {
			unsigned int opVecInd = opVecIt * VEC_SIZE + dot_it;
			if (resVecInd < getNumElems(d_resVec.mdim) && opVecInd < getNumElems(d_opVec.mdim)) {
				resVal += d_opVec_s[dot_it] * getElem(d_opMat, opVecInd, resVecInd);
			}
		}
		__syncthreads();
	}
	if (resVecInd < getNumElems(d_resVec.mdim)) {
		setElemFlatten(d_resVec, resVecInd, resVal);
	}
}

__global__ void scalarAddBiasKernel(CudaMatrixArg<DTYPE> d_act, const CudaMatrixArg<DTYPE> d_bias) {
	unsigned int flatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	if (flatInd < getNumElems(d_act.mdim)) {
		DTYPE res = getElemFlatten(d_act, flatInd) + getElemFlatten(d_bias, flatInd);
		setElemFlatten(d_act, flatInd, res);
	}
}

cudaError_t scalarFCForwardPropWithCuda(const CudaMatrix<DTYPE>& d_opVec, const CudaMatrix<DTYPE>& d_opMat, const CudaMatrix<DTYPE>& d_bias, CudaMatrix<DTYPE>& d_resVec, ActivationType actType) {
	assert(d_opVec.mdim.getNumElems() == d_opMat.mdim.rdim && d_opMat.mdim.cdim == d_resVec.mdim.getNumElems());
	assert(d_bias.mdim == d_resVec.mdim);
	
	scalarActFunc actFunc;
	cudaError_t cudaStatus;
	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_sigmoidFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_reluFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_softmaxFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actfunc not yet implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol relu failed! %s\n", cudaGetErrorString(cudaStatus));
	}
	unsigned int numVecIts = d_resVec.mdim.getNumElems() / VEC_SIZE;
	if (numVecIts * VEC_SIZE < d_resVec.mdim.getNumElems())
		++numVecIts;

	scalarVecMatMulKernel <<< numVecIts, VEC_SIZE >>> (d_opVec.getCudaMatrixArg(), d_opMat.getCudaMatrixArg(), d_resVec.getCudaMatrixArg());

	cudaStatus = cudaGetLastError();
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarVecMatMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	scalarAddBiasKernel <<< numVecIts, VEC_SIZE >>> (d_resVec.getCudaMatrixArg(), d_bias.getCudaMatrixArg());

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarAddBiasKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	scalarActFuncKernel <<< numVecIts, VEC_SIZE >>> (d_resVec.getCudaMatrixArg(), actFunc);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarActFuncKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	
	return cudaStatus;
}

__global__ void scalarAvgPoolKernel(const CudaMatrixArg<DTYPE> d_prevAct, ConvParams convParams, CudaMatrixArg<DTYPE> d_nextAct, scalarActFunc actFunc) {
	int nextRowInd = threadIdx.y + (blockIdx.y * blockDim.y);
	int nextColInd = threadIdx.x + (blockIdx.x * blockDim.x);
	const int actMap = blockIdx.z;
	if (nextRowInd < d_nextAct.mdim.rdim && nextColInd < d_nextAct.mdim.cdim) {
		int prevRowInd = nextRowInd * convParams.stride;
		int prevColInd = nextColInd * convParams.stride;
		DTYPE result = 0;
		for (int f_row = 0; f_row < convParams.stride; ++f_row) {
			for (int f_col = 0; f_col < convParams.stride; ++f_col) {
				result += getElem(d_prevAct, prevRowInd + f_row, prevColInd + f_col, actMap);
			}
		}
		result /= ((DTYPE)(convParams.stride * convParams.stride));
		result = actFunc(result);
		setElem(d_nextAct, nextRowInd, nextColInd, result, actMap);
	}
}

cudaError_t scalarAvgPoolWithCuda(const CudaMatrix<DTYPE>& d_prevAct, const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextAct, ActivationType actType) {

	assert(d_prevAct.mdim.adim == d_nextAct.mdim.adim);
	assert(d_prevAct.mdim.rdim / convParams.stride == d_nextAct.mdim.rdim && d_prevAct.mdim.cdim / convParams.stride == d_nextAct.mdim.cdim);

	scalarActFunc actFunc;
	cudaError_t cudaStatus;
	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_sigmoidFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_reluFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_softmaxFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actfunc not yet implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol avgpool failed! %s\n", cudaGetErrorString(cudaStatus));
	}
	
	// tile up nextAct and average over d_prevAct receptive field

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	unsigned int numRowIts = d_nextAct.mdim.rdim / BLOCK_SIZE;
	if (numRowIts * BLOCK_SIZE < d_nextAct.mdim.rdim)
		++numRowIts;
	unsigned int numColIts = d_nextAct.mdim.cdim / BLOCK_SIZE;
	if (numColIts * BLOCK_SIZE < d_nextAct.mdim.cdim)
		++numColIts;

	dim3 gridDim(numColIts, numRowIts, d_prevAct.mdim.adim); // x, y
	scalarAvgPoolKernel <<< gridDim, blockDim >>> (d_prevAct.getCudaMatrixArg(), convParams, d_nextAct.getCudaMatrixArg(), actFunc);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarAvgPoolKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void scalarMaxPoolKernel(const CudaMatrixArg<DTYPE> d_prevAct, ConvParams convParams, CudaMatrixArg<DTYPE> d_nextAct, scalarActFunc actFunc, curandState* curandptr, float dropout = 0.0f) {
	float prn = 1.0f;
	int nextRowInd = threadIdx.y + (blockIdx.y * blockDim.y);
	int nextColInd = threadIdx.x + (blockIdx.x * blockDim.x);
	const int actMap = blockIdx.z;
	if (nextRowInd < d_nextAct.mdim.rdim && nextColInd < d_nextAct.mdim.cdim) {
		if (dropout > 0.0f) {
			prn = curand_uniform(&curandptr[getInd(d_nextAct.mdim, nextRowInd, nextColInd, actMap)]);
		}
		int prevRowInd = nextRowInd * convParams.stride;
		int prevColInd = nextColInd * convParams.stride;
		DTYPE max = 0;
		if (prn >= dropout) {
			for (int f_row = 0; f_row < convParams.stride; ++f_row) {
				for (int f_col = 0; f_col < convParams.stride; ++f_col) {
					DTYPE val = getElem(d_prevAct, prevRowInd + f_row, prevColInd + f_col, actMap);
					if (val > max) {
						max = val;
					}
				}
			}
		}
		max = actFunc(max);
		setElem(d_nextAct, nextRowInd, nextColInd, max, actMap);
	}
}

cudaError_t scalarMaxPoolWithCuda(const CudaMatrix<DTYPE>& d_prevAct, const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextAct, ActivationType actType, curandState* curandptr, float dropout) {
	assert(d_prevAct.mdim.adim == d_nextAct.mdim.adim);
	assert(d_prevAct.mdim.rdim / convParams.stride == d_nextAct.mdim.rdim && d_prevAct.mdim.cdim / convParams.stride == d_nextAct.mdim.cdim);
	assert(dropout >= 0.0f && dropout <= 1.0f);
	assert(actType == ActivationType::relu || dropout == 0.0f); // backpropagation will not work correctly with dropout otherwise

	scalarActFunc actFunc;
	cudaError_t cudaStatus;
	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_sigmoidFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_reluFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actFunc, p_softmaxFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actfunc not yet implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol maxpool failed! %s\n", cudaGetErrorString(cudaStatus));
	}

	// tile up nextAct and average over d_prevAct receptive field

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	unsigned int numRowIts = d_nextAct.mdim.rdim / BLOCK_SIZE;
	if (numRowIts * BLOCK_SIZE < d_nextAct.mdim.rdim)
		++numRowIts;
	unsigned int numColIts = d_nextAct.mdim.cdim / BLOCK_SIZE;
	if (numColIts * BLOCK_SIZE < d_nextAct.mdim.cdim)
		++numColIts;
	dim3 gridDim(numColIts, numRowIts, d_prevAct.mdim.adim); // x, y, z


	scalarMaxPoolKernel <<< gridDim, blockDim >>> (d_prevAct.getCudaMatrixArg(), convParams, d_nextAct.getCudaMatrixArg(), actFunc, curandptr, dropout);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("scalarMaxpool error %s \n", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

__global__ void setupRNGKernel(curandState* state, MatrixDim mdim, unsigned long long seed) {
	int idx = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx < getNumElems(mdim)) {
		curand_init(seed, idx, 0, &(state[idx]));
	}
}

cudaError_t setupRNGWithCuda(const MatrixDim& mdim, curandState* state, unsigned long long seed) {
	int numIts = mdim.getNumElems() / VEC_SIZE;
	if (numIts * VEC_SIZE < mdim.getNumElems()) {
		++numIts;
	}
	setupRNGKernel <<< numIts, VEC_SIZE >>> (state, mdim, seed);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("setupRNGWithCuda failed! %s\n", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

__global__ void transposeMatKernel(CudaMatrixArg<DTYPE> d_orig, CudaMatrixArg<DTYPE> d_trans) {
	unsigned int colInd = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int rowInd = threadIdx.y + (blockIdx.y * blockDim.y);
	if (colInd < d_orig.mdim.cdim && rowInd < d_orig.mdim.rdim) {
		setElem(d_trans, colInd, rowInd, getElem(d_orig, rowInd, colInd));
	}
}

__global__ void scalarUpdateBiasKernel(CudaMatrixArg<DTYPE> d_nextError, CudaMatrixArg<DTYPE> d_nextBias, float lrnRate) {
	unsigned int flatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	if (flatInd < getNumElems(d_nextError.mdim)) {
		DTYPE err = getElemFlatten(d_nextError, flatInd);
		DTYPE wgt = getElemFlatten(d_nextBias, flatInd);
		scalarAdjustWeight(wgt, ((DTYPE)1.0f), err, lrnRate);
		setElemFlatten(d_nextBias, flatInd, wgt);
	}
}

__global__ void scalarUpdateMatrixKernel(CudaMatrixArg<DTYPE> d_prevAct, CudaMatrixArg<DTYPE> d_weights, CudaMatrixArg<DTYPE> d_nextError, float lrnRate) {
	unsigned int wgtRowInd = threadIdx.y + (blockIdx.y * blockDim.y);
	unsigned int wgtColInd = threadIdx.x + (blockIdx.x * blockDim.x);
	if (wgtRowInd < d_weights.mdim.rdim && wgtColInd < d_weights.mdim.cdim) {
		DTYPE wgt = getElem(d_weights, wgtRowInd, wgtColInd);
		DTYPE inp = getElemFlatten(d_prevAct, wgtRowInd);
		DTYPE err = getElemFlatten(d_nextError, wgtColInd);
		scalarAdjustWeight(wgt, inp, err, lrnRate);
		setElem(d_weights, wgtRowInd, wgtColInd, wgt);
	}
}

cudaError_t scalarFCBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>& d_weights, const CudaMatrix<DTYPE>& d_nextAct,
	const CudaMatrix<DTYPE>& d_nextError, CudaMatrix<DTYPE>& d_nextBias, ActivationType actType, float lrnRate) {
	assert(d_prevError.mdim.getNumElems() == d_weights.mdim.rdim && d_weights.mdim.cdim == d_nextAct.mdim.getNumElems());
	assert(d_nextAct.mdim == d_nextError.mdim && d_nextAct.mdim == d_nextBias.mdim);

	unsigned int numRowIts = d_weights.mdim.rdim / BLOCK_SIZE;
	if (numRowIts * BLOCK_SIZE < d_weights.mdim.rdim)
		++numRowIts;
	unsigned int numColIts = d_weights.mdim.cdim / BLOCK_SIZE;
	if (numColIts * BLOCK_SIZE < d_weights.mdim.cdim)
		++numColIts;

	scalarActFunc actDerivFunc;
	
	cudaError_t cudaStatus;
	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_sigmoidDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_reluDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_softmaxDerivFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actDerivfunc not implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol reluDeriv failed! %s\n", cudaGetErrorString(cudaStatus));
	}

	int numNextIts = d_nextAct.mdim.getNumElems() / VEC_SIZE;
	if (numNextIts * VEC_SIZE < d_nextAct.mdim.getNumElems())
		++numNextIts;

	// calculate activation function derivative of nextError
	scalarActDerivFuncKernel <<< numNextIts, VEC_SIZE >>> (d_nextAct.getCudaMatrixArg(), d_nextError.getCudaMatrixArg(), actDerivFunc);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarActFuncKerel FCBackprop launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// update the bias
	scalarUpdateBiasKernel <<< numNextIts, VEC_SIZE >>> (d_nextError.getCudaMatrixArg(), d_nextBias.getCudaMatrixArg(), lrnRate);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarUpdateBiasKernel FCBackprop launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// transpose the matrix

	dim3 gridDim(numColIts, numRowIts); // x, y
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	CudaMatrix<DTYPE> d_weightsT(d_weights.mdim.transpose());
	transposeMatKernel <<< gridDim, blockDim >>> (d_weights.getCudaMatrixArg(), d_weightsT.getCudaMatrixArg());

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "transposeMatKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	int numPrevIts = d_prevError.mdim.getNumElems() / VEC_SIZE;
	if (numPrevIts * VEC_SIZE < d_prevError.mdim.getNumElems())
		++numPrevIts;

	// propagate error backwards
	scalarVecMatMulKernel <<< numPrevIts, VEC_SIZE >>>(d_nextError.getCudaMatrixArg(), d_weightsT.getCudaMatrixArg(), d_prevError.getCudaMatrixArg());
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarVecMatMulKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// update FC matrix
	scalarUpdateMatrixKernel <<< gridDim, blockDim >>> (d_prevAct.getCudaMatrixArg(), d_weights.getCudaMatrixArg(), d_nextError.getCudaMatrixArg(), lrnRate);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarUpdateMatrixKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void scalarAvgPoolBackpropKernel(CudaMatrixArg<DTYPE> d_prevError, ConvParams convParams,
	CudaMatrixArg<DTYPE> d_nextAct, CudaMatrixArg<DTYPE> d_nextError, scalarActFunc actDerivFunc) {
	const int actMap = blockIdx.z;
	int prevRowInd = threadIdx.y + (blockIdx.y * blockDim.y);
	int prevColInd = threadIdx.x + (blockIdx.x * blockDim.x);
	int nextRowInd = prevRowInd / convParams.stride;
	int nextColInd = prevColInd / convParams.stride;
	if (prevRowInd < d_prevError.mdim.rdim && prevColInd < d_prevError.mdim.cdim && nextRowInd < d_nextAct.mdim.rdim && nextColInd < d_nextAct.mdim.cdim) {
		DTYPE err = getElem(d_nextError, nextRowInd, nextColInd, actMap);
		err *= actDerivFunc(getElem(d_nextAct, nextRowInd, nextColInd, actMap));
		err /= ((DTYPE)(convParams.stride * convParams.stride));
		setElem(d_prevError, prevRowInd, prevColInd, err, actMap);
	}
}

cudaError_t scalarAvgPoolBackpropWithCuda(CudaMatrix<DTYPE>& d_prevError, const ConvParams& convParams, 
	const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextError, ActivationType actType) {

	assert(d_prevError.mdim.adim == d_nextAct.mdim.adim);
	assert(d_nextAct.mdim == d_nextError.mdim);
	assert(d_prevError.mdim.rdim / convParams.stride == d_nextAct.mdim.rdim && d_prevError.mdim.cdim / convParams.stride == d_nextAct.mdim.cdim);

	scalarActFunc actDerivFunc;
	cudaError_t cudaStatus;
	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_sigmoidDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_reluDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_softmaxDerivFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actfunc not yet implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol relu failed! %s\n", cudaGetErrorString(cudaStatus));
	}

	// tile up nextAct and average over d_prevAct receptive field

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	unsigned int numRowIts = d_prevError.mdim.rdim / BLOCK_SIZE;
	if (numRowIts * BLOCK_SIZE < d_prevError.mdim.rdim)
		++numRowIts;
	unsigned int numColIts = d_prevError.mdim.cdim / BLOCK_SIZE;
	if (numColIts * BLOCK_SIZE < d_prevError.mdim.cdim)
		++numColIts;

	dim3 gridDim(numColIts, numRowIts, d_prevError.mdim.adim); // x, y
	scalarAvgPoolBackpropKernel <<< gridDim, blockDim >>> (d_prevError.getCudaMatrixArg(), convParams,
		d_nextAct.getCudaMatrixArg(), d_nextError.getCudaMatrixArg(), actDerivFunc);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarAvgPoolKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void scalarMaxPoolBackpropKernel(CudaMatrixArg<DTYPE> d_prevAct, CudaMatrixArg<DTYPE> d_prevError, ConvParams convParams,
	CudaMatrixArg<DTYPE> d_nextAct, CudaMatrixArg<DTYPE> d_nextError, scalarActFunc actDerivFunc) {
	int actMap = blockIdx.z;
	int prevRowInd = threadIdx.y + (blockIdx.y * blockDim.y);
	int prevColInd = threadIdx.x + (blockIdx.x * blockDim.x);
	int nextRowInd = prevRowInd / convParams.stride;
	int nextColInd = prevColInd / convParams.stride;
	if (prevRowInd < d_prevError.mdim.rdim && prevColInd < d_prevError.mdim.cdim && nextRowInd < d_nextAct.mdim.rdim && nextColInd < d_nextAct.mdim.cdim) {
		DTYPE nextAct = getElem(d_nextAct, nextRowInd, nextColInd, actMap);
		if (getElem(d_prevAct, prevRowInd, prevColInd, actMap) == nextAct) { // is max
			DTYPE err = getElem(d_nextError, nextRowInd, nextColInd, actMap);
			err *= actDerivFunc(nextAct); // dropout can make this zero and if it is relu, this is zero too
			setElem(d_prevError, prevRowInd, prevColInd, err, actMap); // default is zero, so if this doesn't get set, it will be zero
		}
	}
}

cudaError_t scalarMaxPoolBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevAct, CudaMatrix<DTYPE>& d_prevError, const ConvParams& convParams, const CudaMatrix<DTYPE>& d_nextAct, const CudaMatrix<DTYPE>& d_nextError,
	ActivationType actType) {

	assert(d_prevError.mdim.adim == d_nextAct.mdim.adim);
	assert(d_nextAct.mdim == d_nextError.mdim);
	assert(d_prevError.mdim.rdim / convParams.stride == d_nextAct.mdim.rdim && d_prevError.mdim.cdim / convParams.stride == d_nextAct.mdim.cdim);

	scalarActFunc actDerivFunc;
	cudaError_t cudaStatus;
	switch (actType) {
	case ActivationType::sigmoid:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_sigmoidDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::relu:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_reluDerivFunc, sizeof(scalarActFunc));
		break;
	case ActivationType::softmax:
		cudaStatus = cudaMemcpyFromSymbol(&actDerivFunc, p_softmaxDerivFunc, sizeof(scalarActFunc));
		break;
	default:
		throw std::logic_error("actfunc not yet implemented \n");
		break;
	}
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpyFromSymbol relu failed! %s\n", cudaGetErrorString(cudaStatus));
	}

	// tile up nextAct and average over d_prevAct receptive field

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

	unsigned int numRowIts = d_prevError.mdim.rdim / BLOCK_SIZE;
	if (numRowIts * BLOCK_SIZE < d_prevError.mdim.rdim)
		++numRowIts;
	unsigned int numColIts = d_prevError.mdim.cdim / BLOCK_SIZE;
	if (numColIts * BLOCK_SIZE < d_prevError.mdim.cdim)
		++numColIts;

	dim3 gridDim(numColIts, numRowIts, d_prevError.mdim.adim); // x, y
	scalarMaxPoolBackpropKernel <<< gridDim, blockDim >>> (d_prevAct.getCudaMatrixArg(), d_prevError.getCudaMatrixArg(), convParams,
		d_nextAct.getCudaMatrixArg(), d_nextError.getCudaMatrixArg(), actDerivFunc);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scalarAvgPoolKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}