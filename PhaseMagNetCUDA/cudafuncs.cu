

#include "pmncudautils.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudafuncs.cuh"
#include <assert.h>

constexpr auto ALPHA = 0.002f;
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


__device__ void get_dLdMag_dLdPhi(DTYPE Yr, DTYPE Yi, DTYPE wxr, DTYPE wxi, DTYPE err, DTYPE& dLdMag, DTYPE& dLdPhi) {
	DTYPE abswx = d_abs2(wxr, wxi);
	DTYPE absY = d_abs2(Yr, Yi);
	if (abswx > 0 && absY > 0) {
		DTYPE Y_wxr = Yr - wxr;
		DTYPE Y_wxi = Yi - wxi;
		dLdPhi = (((Y_wxi * wxr) - (Y_wxr * wxi)) / absY); // magnitude of dLdPhi
		dLdMag = ((abswx + (Y_wxr * (wxr / abswx)) + (Y_wxi * (wxi / abswx))) / absY) * err;
		// rotate wx by complex conjugate of Y
		d_cmp_mult(wxr, wxi, Yr, -1.0f * Yi, wxr, wxi);
		dLdPhi = copysignf(dLdPhi, -1.0f * wxi) * err; // sign of dLdPhi, with error
		// gradient clipping here
		//assert(abs(dLdMag) < GRADIENT_CLIP);
		if (abs(dLdMag) > GRADIENT_CLIP) {
			dLdMag = copysignf(GRADIENT_CLIP, dLdMag);
		}
		if (abs(dLdPhi) > GRADIENT_CLIP) {
			dLdPhi = copysignf(GRADIENT_CLIP, dLdPhi);
		}
	}
	else { // ReLU discontinuity
		dLdMag = 0;
		dLdPhi = 0;
	}
}


//__device__ void get_dPhidMag_dPhidPhi(DTYPE Yr, DTYPE Yi, DTYPE wxr, DTYPE wxi, DTYPE err, DTYPE& dLdMag, DTYPE& dLdPhi) {
//	;
//}

__device__ int getInd(MatrixDim mdim, int row, int col, int aisle = 0) {
	return (aisle * mdim.astride) + (row * mdim.rstride) + col;
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
cudaError_t vecMatMultWithCuda(const CudaMatrix<DTYPE>& d_Ar, const CudaMatrix<DTYPE>& d_Ai, 
	const CudaMatrix<DTYPE>& d_Br, const CudaMatrix<DTYPE>& d_Bi, 
	CudaMatrix<DTYPE>& d_Cr, CudaMatrix<DTYPE>& d_Ci) {
	// flattened A must have same size as b row
	assert(d_Ar.mdim.getNumElems() == d_Br.mdim.rdim && d_Br.mdim.cdim == d_Cr.mdim.getNumElems()); // check mult compat
	assert(d_Ar.mdim == d_Ai.mdim); // real and imag must have same shape
	assert(d_Br.mdim == d_Bi.mdim);
	assert(d_Cr.mdim == d_Ci.mdim);
	cudaError_t cudaStatus = cudaSuccess;

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

	return cudaStatus;
}

__global__ void complexAddBiasKernel(CudaMatrixArg<DTYPE> d_ActR, CudaMatrixArg<DTYPE> d_ActI,
	CudaMatrixArg<DTYPE> d_BiasR, CudaMatrixArg<DTYPE> d_BiasI) {
	const int flatInd = threadIdx.x + (blockIdx.x * blockDim.x);
	if (flatInd < getNumElems(d_ActR.mdim)) {
		DTYPE pactR = getElemFlatten(d_ActR, flatInd);
		DTYPE pactI = getElemFlatten(d_ActI, flatInd);
		DTYPE bR = getElemFlatten(d_BiasR, flatInd);
		DTYPE bI = getElemFlatten(d_BiasI, flatInd);
		setElemFlatten(d_ActR, flatInd, pactR + bR);
		setElemFlatten(d_ActI, flatInd, pactI + bI);
	}
}

// helper function for MatMul
cudaError_t complexAddBiasWithCuda(CudaMatrix<DTYPE>& d_ActR, CudaMatrix<DTYPE>& d_ActI,
	const CudaMatrix<DTYPE>& d_BiasR, const CudaMatrix<DTYPE>& d_BiasI) {
	// flattened A must have same size as b row
	assert(d_ActR.mdim == d_ActI.mdim);
	assert(d_BiasR.mdim == d_BiasI.mdim);
	assert(d_ActR.mdim == d_BiasR.mdim);
	cudaError_t cudaStatus = cudaSuccess;

	int numIts = d_ActR.mdim.getNumElems() / VEC_SIZE;
	if (numIts * VEC_SIZE < d_ActR.mdim.getNumElems())
		++numIts;

	complexAddBiasKernel <<< numIts, VEC_SIZE >>> (d_ActR.getCudaMatrixArg(), d_ActI.getCudaMatrixArg(),
		d_BiasR.getCudaMatrixArg(), d_BiasI.getCudaMatrixArg());
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "complexAddBiasKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void setValueKernel(CudaMatrixArg<DTYPE> d_Mat, DTYPE value) {
	const int colVal = threadIdx.x + (blockIdx.x * blockDim.x);
	if (colVal < getNumElems(d_Mat.mdim))
		setElemFlatten(d_Mat, colVal, value);
}

cudaError_t setValueWithCuda(CudaMatrix<DTYPE>& d_Mat, DTYPE value) {
	int numVecs = d_Mat.mdim.getNumElems() / VEC_SIZE;
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

__global__ void complexConvolutionKernel(const CudaMatrixArg<DTYPE> d_prevActR, const CudaMatrixArg<DTYPE> d_prevActI,
	const CudaMatrixArg<DTYPE> d_convR, const CudaMatrixArg<DTYPE> d_convI, const ConvParams convParams, const int filterNum,
	CudaMatrixArg<DTYPE> d_nextActR, CudaMatrixArg<DTYPE> d_nextActI) {
	extern __shared__ DTYPE s[];
	const int sharedInputDim = blockDim.x + (2 * convParams.pad); // == 20 x 20 == 400
	const int numElemsInput = sharedInputDim * sharedInputDim * d_prevActR.mdim.adim;
	DTYPE* prevActR_s = s;
	DTYPE* prevActI_s = &(s[numElemsInput]);
	DTYPE* filterR_s = &(s[2*numElemsInput]);
	DTYPE* filterI_s = &(filterR_s[getNumElems(d_convR.mdim)]);
	// identify which block, which thread, and location in nextAct
	const int ownNextActRow = threadIdx.y + (blockIdx.y * blockDim.y);
	const int ownNextActCol = threadIdx.x + (blockIdx.x * blockDim.x);
	const int flatInd = (threadIdx.y * blockDim.x) + threadIdx.x;
	const int prevRowB = d_prevActR.mdim.rdim;
	const int prevColB = d_prevActR.mdim.cdim;
	const int pad = convParams.pad;
	const int chanStride = sharedInputDim * sharedInputDim;
	// <><><><><><><><><> Hardcoded for stride of 1 <><><><><><><>
	// fetch actual shared input if your ownNextActRow/col is in bounds.
	//for (int chan = 0; chan < d_prevActR.mdim.adim; ++chan) {
	//	if (ownNextActRow < prevRowB && ownNextActCol < prevColB) {
	//		prevActR_s[(chan * chanStride) + ((threadIdx.y + pad) * sharedInputDim) + threadIdx.x + pad] = getElem(d_prevActR, ownNextActRow, ownNextActCol, chan);
	//		prevActI_s[(chan * chanStride) + ((threadIdx.y + pad) * sharedInputDim) + threadIdx.x + pad] = getElem(d_prevActI, ownNextActRow, ownNextActCol, chan);
	//	}
	//	else {
	//		prevActR_s[(chan * chanStride) + ((threadIdx.y + pad) * sharedInputDim) + threadIdx.x + pad] = 0;
	//		prevActI_s[(chan * chanStride) + ((threadIdx.y + pad) * sharedInputDim) + threadIdx.x + pad] = 0;
	//	}
	//	if (flatInd < sharedInputDim) { // get edges
	//		for (int pOff = 1; pOff <= pad; ++pOff) {
	//			// top apron
	//			const int apronRow = blockIdx.y * blockDim.y - pOff;
	//			const int apronCol = blockIdx.x * blockDim.x - pad + flatInd;
	//			if (apronRow >= 0 && apronRow < prevRowB && apronCol >= 0 && apronCol < prevColB) {
	//				prevActR_s[(chan * chanStride) + ((pad - pOff) * sharedInputDim) + flatInd] = getElem(d_prevActR, apronRow, apronCol, chan);
	//				prevActI_s[(chan * chanStride) + ((pad - pOff) * sharedInputDim) + flatInd] = getElem(d_prevActI, apronRow, apronCol, chan);
	//			}
	//			else {
	//				prevActR_s[(chan * chanStride) + ((pad - pOff) * sharedInputDim) + flatInd] = 0;
	//				prevActI_s[(chan * chanStride) + ((pad - pOff) * sharedInputDim) + flatInd] = 0;
	//			}
	//		}
	//		for (int pOff = 1; pOff <= pad; ++pOff) {
	//			// bottom apron
	//			const int apronRow = (blockIdx.y + 1) * blockDim.y + pOff - 1;
	//			const int apronCol = (blockIdx.x * blockDim.x) - pad + flatInd;
	//			if (apronRow >= 0 && apronRow < prevRowB && apronCol >= 0 && apronCol < prevColB) {
	//				prevActR_s[(chan * chanStride) + ((pad + blockDim.y + pOff) * sharedInputDim) + flatInd] = getElem(d_prevActR, apronRow, apronCol, chan);
	//				prevActI_s[(chan * chanStride) + ((pad + blockDim.y + pOff) * sharedInputDim) + flatInd] = getElem(d_prevActI, apronRow, apronCol, chan);
	//			}
	//			else {
	//				prevActR_s[(chan * chanStride) + ((pad + blockDim.y + pOff) * sharedInputDim) + flatInd] = 0;
	//				prevActI_s[(chan * chanStride) + ((pad + blockDim.y + pOff) * sharedInputDim) + flatInd] = 0;
	//			}
	//		}
	//		for (int pOff = 1; pOff <= pad; ++pOff) {
	//			// left apron
	//			const int apronRow = blockIdx.y * blockDim.y - pad + flatInd;
	//			const int apronCol = blockIdx.x * blockDim.x - pOff;
	//			if (apronRow >= 0 && apronRow < prevRowB && apronCol >= 0 && apronCol < prevColB) {
	//				prevActR_s[(chan * chanStride) + ((flatInd) * sharedInputDim) + (pad - pOff)] = getElem(d_prevActR, apronRow, apronCol, chan);
	//				prevActI_s[(chan * chanStride) + ((flatInd) * sharedInputDim) + (pad - pOff)] = getElem(d_prevActI, apronRow, apronCol, chan);
	//			}
	//			else {
	//				prevActR_s[(chan * chanStride) + ((flatInd) * sharedInputDim) + (pad - pOff)] = 0;
	//				prevActI_s[(chan * chanStride) + ((flatInd) * sharedInputDim) + (pad - pOff)] = 0;
	//			}
	//		}
	//		for (int pOff = 1; pOff <= pad; ++pOff) {
	//			// right apron
	//			const int apronRow = blockIdx.y * blockDim.y - pad + flatInd;
	//			const int apronCol = (blockIdx.x + 1) * blockDim.x + pOff - 1;
	//			if (apronRow >= 0 && apronRow < prevRowB && apronCol >= 0 && apronCol < prevColB) {
	//				prevActR_s[(chan * chanStride) + ((flatInd)*sharedInputDim) + (pad + blockDim.x + pOff)] = getElem(d_prevActR, apronRow, apronCol, chan);
	//				prevActI_s[(chan * chanStride) + ((flatInd)*sharedInputDim) + (pad + blockDim.x + pOff)] = getElem(d_prevActI, apronRow, apronCol, chan);
	//			}
	//			else {
	//				prevActR_s[(chan * chanStride) + ((flatInd)*sharedInputDim) + (pad + blockDim.x + pOff)] = 0;
	//				prevActI_s[(chan * chanStride) + ((flatInd)*sharedInputDim) + (pad + blockDim.x + pOff)] = 0;
	//			}
	//		}
	//	}
	//}
	//

	// fetch the filter (assumes dimensions 1xFILTER_DIMxFILTER_DIM)
	if (threadIdx.x < convParams.filterDim.cdim && threadIdx.y < convParams.filterDim.rdim) {
		for (int chan = 0; chan < d_convR.mdim.adim; ++chan) {
			filterR_s[getInd(d_convR.mdim, threadIdx.y, threadIdx.x, chan)] = getElem(d_convR, threadIdx.y, threadIdx.x, chan);
			filterI_s[getInd(d_convI.mdim, threadIdx.y, threadIdx.x, chan)] = getElem(d_convI, threadIdx.y, threadIdx.x, chan);
		}
	}
	__syncthreads(); // shared input and weights have been fetched. compute the result
	if (ownNextActCol < d_nextActR.mdim.cdim && ownNextActRow < d_nextActR.mdim.rdim) { // only do this if result matters
		DTYPE dotValR = 0.0;
		DTYPE dotValI = 0.0;
		for (int f_chan = 0; f_chan < convParams.filterDim.adim; ++f_chan) {
			for (int f_row = 0; f_row < convParams.filterDim.rdim; ++f_row) {
				for (int f_col = 0; f_col < convParams.filterDim.cdim; ++f_col) {
					int s_row = threadIdx.y + f_row;
					int s_col = threadIdx.x + f_col;
					int tmpRow = ownNextActRow - pad + f_row;
					int tmpCol = ownNextActCol - pad + f_col;
					DTYPE weightR = filterR_s[getInd(d_convR.mdim, f_row, f_col, f_chan)];
					DTYPE weightI = filterI_s[getInd(d_convI.mdim, f_row, f_col, f_chan)];
					DTYPE actR = 0;
					DTYPE actI = 0;
					if (tmpRow >= 0 && tmpRow < d_prevActR.mdim.rdim && tmpCol >= 0 && tmpCol < d_prevActR.mdim.cdim) {
						actR = getElem(d_prevActR, tmpRow, tmpCol, f_chan); // prevActR_s[(f_chan * chanStride) + (s_row * sharedInputDim) + s_col];// from f_col to s_col
						actI = getElem(d_prevActI, tmpRow, tmpCol, f_chan); // prevActI_s[(f_chan * chanStride) + (s_row * sharedInputDim) + s_col];
					}
					DTYPE resR, resI;
					d_cmp_mult(actR, actI, weightR, weightI, resR, resI);
					dotValR += resR;
					dotValI += resI;
				}
			}
		}
		setElem(d_nextActR, ownNextActRow, ownNextActCol, dotValR, filterNum);
		setElem(d_nextActI, ownNextActRow, ownNextActCol, dotValI, filterNum); // set in correct filter position
	}
}

cudaError_t complexConvolutionWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI,
	CudaMatrix<DTYPE>* d_convR, CudaMatrix<DTYPE>* d_convI, const ConvParams& convParams,
	CudaMatrix<DTYPE>& d_nextActR, CudaMatrix<DTYPE>& d_nextActI) {
	assert(d_prevActR.mdim == d_prevActI.mdim && d_convR[0].mdim == d_convI[0].mdim && d_nextActR.mdim == d_nextActI.mdim);
	assert(d_convR[0].mdim.adim == d_prevActR.mdim.adim);

	dim3 bDim(BLOCK_SIZE, BLOCK_SIZE);
	unsigned int numRowIts = d_nextActR.mdim.rdim / BLOCK_SIZE;
	if (d_nextActR.mdim.rdim % BLOCK_SIZE != 0)
		++numRowIts;
	unsigned int numColIts = d_nextActR.mdim.cdim / BLOCK_SIZE;
	if (d_nextActR.mdim.cdim % BLOCK_SIZE != 0)
		++numColIts;
	dim3 gridDim(numColIts, numRowIts); // x, y

	cudaError_t cudaStatus(cudaSuccess);
	const int sharedSize = ((BLOCK_SIZE + (convParams.pad * 2)) *
		(BLOCK_SIZE + (convParams.pad * 2)) * d_prevActR.mdim.adim // number of elems of shared input
		+ d_convR[0].mdim.getNumElems()) * 2 * sizeof(DTYPE); // number of elems of convKernel in bytes, re, im
	// loop through the activation maps (filters)
	// call 2D grid of activation map
	// filter is shared memory
	// subset of input is shared memory
	for (int filterNum = 0; filterNum < convParams.numFilters; ++filterNum) {
		
		complexConvolutionKernel <<< gridDim, bDim, sharedSize >>> (d_prevActR.getCudaMatrixArg(), d_prevActI.getCudaMatrixArg(),
			d_convR[filterNum].getCudaMatrixArg(), d_convI[filterNum].getCudaMatrixArg(), convParams, filterNum,
			d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg());

		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "complexConvolutionKernel %d launch failed: %s\n", filterNum, cudaGetErrorString(cudaStatus));
		}

	}
	
	return cudaStatus;
}

__global__ void complexAveragePoolKernel(const CudaMatrixArg<DTYPE> d_prevActR, const CudaMatrixArg<DTYPE> d_prevActI,
	const ConvParams convParams, const size_t actMap, CudaMatrixArg<DTYPE> d_nextActR, CudaMatrixArg<DTYPE> d_nextActI) {
	// where am i in d_nextAct?
	int nextActRow = threadIdx.y + (blockIdx.y * blockDim.y);
	int nextActCol = threadIdx.x + (blockIdx.x * blockDim.x);

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
		DTYPE denom = 1.0 / ((DTYPE)(convParams.stride * convParams.stride));
		resR *= denom;
		resI *= denom;
		setElem(d_nextActR, nextActRow, nextActCol, resR, actMap);
		setElem(d_nextActI, nextActRow, nextActCol, resI, actMap);
	}
}

cudaError_t complexAveragePoolWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI, 
	const ConvParams& convParams, CudaMatrix<DTYPE>& d_nextActR, CudaMatrix<DTYPE>& d_nextActI) {
	assert(d_prevActR.mdim == d_prevActI.mdim);
	assert(d_nextActR.mdim == d_nextActI.mdim);
	assert(d_prevActR.mdim.adim == d_nextActR.mdim.adim); // same number of activation maps
	assert(d_prevActR.mdim.rdim / convParams.stride == d_nextActR.mdim.rdim); // downsample dim
	assert(d_prevActR.mdim.cdim / convParams.stride == d_nextActR.mdim.cdim); // downsample dim

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	size_t numRowIts = d_nextActR.mdim.rdim / BLOCK_SIZE;
	if (d_nextActR.mdim.rdim % BLOCK_SIZE != 0)
		++numRowIts;
	size_t numColIts = d_nextActR.mdim.cdim / BLOCK_SIZE;
	if (d_nextActR.mdim.cdim % BLOCK_SIZE != 0)
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

	return cudaStatus;
}


__global__ void complexBackpropKernel(const CudaMatrixArg<DTYPE> prevActR, const CudaMatrixArg<DTYPE> prevActI,
	CudaMatrixArg<DTYPE> prevError, CudaMatrixArg<DTYPE> weightsR, CudaMatrixArg<DTYPE> weightsI,
	const CudaMatrixArg<DTYPE> nextActR, const CudaMatrixArg<DTYPE> nextActI, const CudaMatrixArg<DTYPE> nextError, float lrnRate) {

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
	for (int nextError_it = 0; nextError_it < num_nextError_its; ++nextError_it) {
		int own_nextErr_col = nextError_it * VEC_SIZE + threadIdx.x;
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
				// apply LRN_RATE to dLdPhi right now
				dLdPhi *= lrnRate;
				DTYPE dLdPhiR, dLdPhiI;
				d_phi_to_comp(dLdPhi, dLdPhiR, dLdPhiI);
				// multiply into result
				// CHANGE - scale magnitude change by current magnitude
				DTYPE absw = d_abs2(wr, wi);
				if (absw > 0) {
					dLdMag *= d_abs2(xr, xi); // wr = (((dLdMag * (1.0f - ALPHA)) - ALPHA) * wr / absw * lrnRate) + wr;
					DTYPE absY = d_abs2(Yr, Yi);
					wr += ((dLdMag - ALPHA * absY * absw) * (wr / absw)) * lrnRate;
					wi += ((dLdMag - ALPHA * absY * absw) * (wi / absw)) * lrnRate;
					d_cmp_mult(wr, wi, dLdPhiR, dLdPhiI, wr, wi); // write back the rotation into the weights
					setElem(weightsR, prevError_col, wgt_col, wr);
					setElem(weightsI, prevError_col, wgt_col, wi);
				}
			}
		}
		__syncthreads();
	}
	if (prevError_col < getNumElems(prevError.mdim)) {
		setElemFlatten(prevError, prevError_col, prevError_val);
	}
}

__global__ void complexUpdateBiasKernel(const CudaMatrixArg<DTYPE> actR, const CudaMatrixArg<DTYPE> actI,
	const CudaMatrixArg<DTYPE> error, CudaMatrixArg<DTYPE> biasR, CudaMatrixArg<DTYPE> biasI, float lrnRate) {
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
		// apply LRN_RATE to dLdPhi; dLdMag later
		dLdPhi *= lrnRate;
		DTYPE dLdPhiR, dLdPhiI;
		d_phi_to_comp(dLdPhi, dLdPhiR, dLdPhiI);
		// multiply into result
		// CHANGE - scale magnitude change by current magnitude
		DTYPE absw = d_abs2(wr, wi);
		if (absw > 0) {
			// magnitude of input is 1
			DTYPE absY = d_abs2(Yr, Yi);
			wr += ((dLdMag - ALPHA * absY * absw) * (wr / absw)) * lrnRate;
			wi += ((dLdMag - ALPHA * absY * absw) * (wi / absw)) * lrnRate;
			d_cmp_mult(wr, wi, dLdPhiR, dLdPhiI, wr, wi); // write back the rotation into the weights
			setElemFlatten(biasR, col, wr);
			setElemFlatten(biasI, col, wi);
		}
		
	}
}


cudaError_t complexBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI, // if input layer, error can be used to create adv examples
	CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>& d_weightsR, CudaMatrix<DTYPE>& d_weightsI, CudaMatrix<DTYPE>& d_nextBiasR, CudaMatrix<DTYPE>& d_nextBiasI,
	const CudaMatrix<DTYPE>& d_nextActR, const CudaMatrix<DTYPE>& d_nextActI, const CudaMatrix<DTYPE>& d_nextError, float lrnRate) {
	
	// check that the dimensions fit
	assert(d_prevActR.mdim == d_prevActI.mdim && d_prevActR.mdim == d_prevError.mdim); // prev parallel
	assert(d_nextActR.mdim == d_nextActI.mdim && d_nextActR.mdim == d_nextError.mdim); // next parallel
	assert(d_nextBiasR.mdim == d_nextBiasI.mdim && d_nextBiasR.mdim == d_nextActR.mdim); // next bias parallel
	assert(d_weightsR.mdim == d_weightsI.mdim); // weights parallel
	assert(d_prevActR.mdim.getNumElems() == d_weightsR.mdim.rdim && d_weightsR.mdim.cdim == d_nextActR.mdim.getNumElems()); // transfer

	// compute next bias with the nextAct, nextBias, nextError
	unsigned int num_vecs = d_nextBiasR.mdim.getNumElems() / VEC_SIZE;
	if (d_nextBiasR.mdim.getNumElems() % VEC_SIZE != 0) {
		++num_vecs;
	}
	complexUpdateBiasKernel <<< num_vecs, VEC_SIZE >>> (d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg(),
		d_nextError.getCudaMatrixArg(), d_nextBiasR.getCudaMatrixArg(), d_nextBiasI.getCudaMatrixArg(), lrnRate);

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
	//const int ind = d_weightsR.mdim.getNumElems() / 2;
	//Matrix<DTYPE> weightBefore(d_weightsR);
	complexBackpropKernel <<< num_vecs, VEC_SIZE >>> (d_prevActR.getCudaMatrixArg(), d_prevActI.getCudaMatrixArg(),
		d_prevError.getCudaMatrixArg(), d_weightsR.getCudaMatrixArg(), d_weightsI.getCudaMatrixArg(),
		d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg(), d_nextError.getCudaMatrixArg(), lrnRate);
	
	//Matrix<DTYPE> weightAfter(d_weightsR);

	//printf("Weight Ratio: After/Before = %5.5f\n", weightAfter.getElemFlatten(ind) / weightBefore.getElemFlatten(ind) - 1.0f);

	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "complexBackpropKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void complexConvBackpropKernel(CudaMatrixArg<DTYPE> d_prevActR, CudaMatrixArg<DTYPE> d_prevActI, CudaMatrixArg<DTYPE> d_prevError,
	CudaMatrixArg<DTYPE> d_weightsR, CudaMatrixArg<DTYPE> d_weightsI,
	CudaMatrixArg<DTYPE> d_filterErrMag, CudaMatrixArg<DTYPE> d_filterErrPhi, const int actMap, const ConvParams convParams,
	CudaMatrixArg<DTYPE> d_nextActR, CudaMatrixArg<DTYPE> d_nextActI, CudaMatrixArg<DTYPE> d_nextError) {
	extern __shared__ DTYPE s[];
	DTYPE* wgtR_s = s;
	DTYPE* wgtI_s = &(s[getNumElems(d_weightsR.mdim)]);
	DTYPE* wgtErrMag_s = &(s[2 * getNumElems(d_weightsR.mdim)]);
	DTYPE* wgtErrPhi_s = &(s[3 * getNumElems(d_weightsR.mdim)]);
	// where am i in prevAct?
	int prevFlatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	int astride = d_prevActR.mdim.astride;
	int rstride = d_prevActR.mdim.rstride;
	int prevAisleInd = prevFlatInd / astride;
	int prevRowInd = (prevFlatInd - (prevAisleInd * astride)) / rstride;
	int prevColInd = prevFlatInd - (prevAisleInd * astride) - (prevRowInd * rstride);
	// fetch the shared filter data if your prevFlatInd < filter.getNumElems()
	if (threadIdx.x < getNumElems(d_weightsR.mdim)) {
		wgtR_s[threadIdx.x] = getElemFlatten(d_weightsR, threadIdx.x);
		wgtI_s[threadIdx.x] = getElemFlatten(d_weightsI, threadIdx.x);
		wgtErrMag_s[threadIdx.x] = getElemFlatten(d_filterErrMag, threadIdx.x);
		wgtErrPhi_s[threadIdx.x] = getElemFlatten(d_filterErrPhi, threadIdx.x);
	}
	__syncthreads();
	// if location in prevAct is valid
	if (prevFlatInd < getNumElems(d_prevActR.mdim)) {
		DTYPE eMagPrev = 0;
		DTYPE xR = getElemFlatten(d_prevActR, prevFlatInd);
		DTYPE xI = getElemFlatten(d_prevActI, prevFlatInd);
		// iterate through the filter to calculate the contributed error to prev the filter
		for (int fRow = 0; fRow < d_weightsR.mdim.rdim; ++fRow) {
			for (int fCol = 0; fCol < d_weightsR.mdim.cdim; ++fCol) {
				int nextRowInd = prevRowInd + (convParams.filterDim.rdim / 2) - fRow;
				int nextColInd = prevColInd + (convParams.filterDim.cdim / 2) - fCol;
				if (nextRowInd >= 0 && nextRowInd < d_nextActR.mdim.rdim && nextColInd >= 0 && nextColInd < d_nextActR.mdim.cdim) {
					// use aisle of prev to select filter aisle
					DTYPE wgtR = wgtR_s[getInd(d_weightsR.mdim, fRow, fCol, prevAisleInd)];
					DTYPE wgtI = wgtI_s[getInd(d_weightsI.mdim, fRow, fCol, prevAisleInd)];
					DTYPE err = getElem(d_nextError, nextRowInd, nextColInd, actMap); // error aisle corresponds to actMap
					DTYPE YR = getElem(d_nextActR, nextRowInd, nextColInd, actMap);
					DTYPE YI = getElem(d_nextActI, nextRowInd, nextColInd, actMap);
					// calculate gradient and atomic add to shared ErrMag, ErrPhi, add to local eMagPrev
					DTYPE wxr, wxi;
					d_cmp_mult(xR, xI, wgtR, wgtI, wxr, wxi);
					DTYPE dLdMag, dLdPhi;
					get_dLdMag_dLdPhi(YR, YI, wxr, wxi, err, dLdMag, dLdPhi);
					eMagPrev += (d_abs2(wgtR, wgtI) * dLdMag);
					// atomic add the other things
					dLdMag *= d_abs2(xR, xI);
					atomicAdd(&(wgtErrMag_s[getInd(d_filterErrMag.mdim, fRow, fCol, prevAisleInd)]), dLdMag);
					atomicAdd(&(wgtErrPhi_s[getInd(d_filterErrPhi.mdim, fRow, fCol, prevAisleInd)]), dLdPhi);
				}
			}
		}
		eMagPrev += getElemFlatten(d_prevError, prevFlatInd);
		setElemFlatten(d_prevError, prevFlatInd, eMagPrev);
	}
	__syncthreads();
	if (threadIdx.x < getNumElems(d_weightsR.mdim)) {
		setElemFlatten(d_filterErrMag, threadIdx.x, wgtErrMag_s[threadIdx.x]);
		setElemFlatten(d_filterErrPhi, threadIdx.x, wgtErrPhi_s[threadIdx.x]);
	}
}

__global__ void complexUpdateKernelWeights(CudaMatrixArg<DTYPE> d_weightsR, CudaMatrixArg<DTYPE> d_weightsI,
	CudaMatrixArg<DTYPE> d_filterErrMag, CudaMatrixArg<DTYPE> d_filterErrPhi, const float lrnRate, const unsigned int receptiveFieldSize) {

	DTYPE wR = getElem(d_weightsR, threadIdx.y, threadIdx.x, threadIdx.z);
	DTYPE wI = getElem(d_weightsI, threadIdx.y, threadIdx.x, threadIdx.z);
	DTYPE eMag = getElem(d_filterErrMag, threadIdx.y, threadIdx.x, threadIdx.z); 
	DTYPE ePhi = getElem(d_filterErrPhi, threadIdx.y, threadIdx.x, threadIdx.z) * lrnRate; // learn rate is applied to ePhi
	DTYPE ePhiR, ePhiI;
	d_phi_to_comp(ePhi, ePhiR, ePhiI);
	// stretch weights by dMag
	// CHANGE - scale magnitude change by current magnitude
	DTYPE absw = d_abs2(wR, wI);
	if (absw > 0.0f) {
		eMag /= ((DTYPE)receptiveFieldSize);
		wR += ((eMag - ALPHA * absw) * (wR / absw)) * lrnRate;
		wR += ((eMag - ALPHA * absw) * (wI / absw)) * lrnRate;
		// rotate weights by dPhi
		d_cmp_mult(wR, wI, ePhiR, ePhiI, wR, wI);
		// write back
		setElem(d_weightsR, threadIdx.y, threadIdx.x, wR, threadIdx.z);
		setElem(d_weightsI, threadIdx.y, threadIdx.x, wI, threadIdx.z);
	}
}

cudaError_t complexConvBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI,
	CudaMatrix<DTYPE>& d_prevError, CudaMatrix<DTYPE>* d_weightsR, CudaMatrix<DTYPE>* d_weightsI, const ConvParams& convParams,
	const CudaMatrix<DTYPE>& d_nextActR, const CudaMatrix<DTYPE> d_nextActI, const CudaMatrix<DTYPE>& d_nextError, float lrnRate) {
	// check that the dimensions fit
	assert(d_prevActR.mdim == d_prevActI.mdim && d_prevActR.mdim == d_prevError.mdim); // prev parallel
	assert(d_nextActR.mdim == d_nextActI.mdim && d_nextActR.mdim == d_nextError.mdim); // next parallel
	assert(d_weightsR[0].mdim == d_weightsI[0].mdim); // weights parallel

	cudaError_t cudaStatus(cudaSuccess);

	const int linearBlock = BLOCK_SIZE * BLOCK_SIZE;
	int numLinearBlocks = d_prevActR.mdim.getNumElems() / linearBlock;
	if (d_prevActR.mdim.getNumElems() % linearBlock != 0) {
		++numLinearBlocks;
	}

	CudaMatrix<DTYPE> d_filterErrMag(convParams.filterDim);
	CudaMatrix<DTYPE> d_filterErrPhi(convParams.filterDim);

	const int sharedSize = 4 * d_weightsR[0].mdim.size;
	assert(linearBlock > d_weightsR[0].mdim.getNumElems()); // shared memory filled based on threadId
	
	// dynamic shared memory is cumulative mag/phi error for the filter
	for (int actMap = 0; actMap < convParams.numFilters; ++actMap) {
		// global scratchpad memory for filter error
		setValueWithCuda(d_filterErrMag, 0);
		setValueWithCuda(d_filterErrPhi, 0);

		complexConvBackpropKernel <<< numLinearBlocks, linearBlock, sharedSize >>>
			(d_prevActR.getCudaMatrixArg(),	d_prevActI.getCudaMatrixArg(), d_prevError.getCudaMatrixArg(), 
				d_weightsR[actMap].getCudaMatrixArg(), d_weightsI[actMap].getCudaMatrixArg(), 
				d_filterErrMag.getCudaMatrixArg(), d_filterErrPhi.getCudaMatrixArg(), actMap, convParams, 
				d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg(), d_nextError.getCudaMatrixArg());

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
		complexUpdateKernelWeights <<< 1, dim3(fDim.cdim, fDim.rdim, fDim.adim) >>> (d_weightsR[actMap].getCudaMatrixArg(),
			d_weightsI[actMap].getCudaMatrixArg(), d_filterErrMag.getCudaMatrixArg(), d_filterErrPhi.getCudaMatrixArg(), lrnRate, d_prevActR.mdim.rdim * d_prevActR.mdim.cdim);
		
		cudaStatus = cudaGetLastError();

		if (cudaStatus != cudaSuccess) {
			// Check for any errors launching the kernel
			fprintf(stderr, "complexUpdateKernelWeights launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
	}

	return cudaStatus;
}

__global__ void complexAvgPoolBackpropKernel(CudaMatrixArg<DTYPE> d_prevActR, CudaMatrixArg<DTYPE> d_prevActI,
	CudaMatrixArg<DTYPE> d_prevError, const ConvParams convParams, CudaMatrixArg<DTYPE> d_nextActR, 
	CudaMatrixArg<DTYPE> d_nextActI, CudaMatrixArg<DTYPE> d_nextError) {
	// where am i in prevError?
	int prevFlatInd = threadIdx.x + (blockDim.x * blockIdx.x);
	int astride = d_prevError.mdim.astride;
	int rstride = d_prevError.mdim.rstride;
	int prevAisleInd = prevFlatInd / astride;
	int prevRowInd = (prevFlatInd - (prevAisleInd * astride)) / rstride;
	int prevColInd = prevFlatInd - (prevAisleInd * astride) - (prevRowInd * rstride);
	// could use shared memory of nextAct for each of the receptive fields but will save that for later
	int nextRowInd = prevRowInd / convParams.stride;
	int nextColInd = prevColInd / convParams.stride;
	// calculate the derivative of the Loss wrt magnitude for these constant weights
	if (prevAisleInd < d_prevError.mdim.adim && prevRowInd < d_prevError.mdim.rdim && prevColInd < d_prevError.mdim.cdim) { // if fit in prevAct, should fit in nextAct
		DTYPE nErr = getElem(d_nextError, nextRowInd, nextColInd, prevAisleInd);
		DTYPE dLdMag, dLdPhi;
		DTYPE wri = 1.0f / ((DTYPE)(convParams.stride * convParams.stride));
		DTYPE wxr = getElem(d_prevActR, prevRowInd, prevColInd, prevAisleInd)* wri;
		DTYPE wxi = getElem(d_prevActI, prevRowInd, prevColInd, prevAisleInd)* wri;
		DTYPE Yr = getElem(d_nextActR, nextRowInd, nextColInd, prevAisleInd);
		DTYPE Yi = getElem(d_nextActI, nextRowInd, nextColInd, prevAisleInd);
		get_dLdMag_dLdPhi(Yr, Yi, wxr, wxi, nErr, dLdMag, dLdPhi);

		// multiply dLdMag by magnitude of weight to get projected mag error
		dLdMag *= wri;
		setElem(d_prevError, prevRowInd, prevColInd, dLdMag, prevAisleInd);
	}
}

cudaError_t complexAvgPoolBackpropWithCuda(const CudaMatrix<DTYPE>& d_prevActR, const CudaMatrix<DTYPE>& d_prevActI, CudaMatrix<DTYPE>& d_prevError,
	const ConvParams& convParams, const CudaMatrix<DTYPE>& d_nextActR, const CudaMatrix<DTYPE> d_nextActI, const CudaMatrix<DTYPE>& d_nextError) {
	assert(d_prevActR.mdim == d_prevActI.mdim && d_prevActR.mdim == d_prevError.mdim); // prev parallel
	assert(d_nextActR.mdim == d_nextActI.mdim && d_nextActR.mdim == d_nextError.mdim); // next parallel
	assert(d_prevActR.mdim.getNumElems() / (convParams.stride * convParams.stride) == d_nextActR.mdim.getNumElems()); // downsample
	assert(d_prevActR.mdim.adim == d_nextActR.mdim.adim && d_prevActR.mdim.adim == convParams.numFilters); // preserve number of actMaps

	// loop through aisle layer and launch blocks
	const int linearBlock = BLOCK_SIZE * BLOCK_SIZE;
	int numLinearBlocks = d_prevError.mdim.getNumElems() / linearBlock;
	if (d_prevError.mdim.getNumElems() % linearBlock != 0) {
		++numLinearBlocks;
	}

	cudaError_t cudaStatus(cudaSuccess);

	complexAvgPoolBackpropKernel <<< numLinearBlocks, linearBlock >>> (d_prevActR.getCudaMatrixArg(), d_prevActI.getCudaMatrixArg(),
		d_prevError.getCudaMatrixArg(), convParams, d_nextActR.getCudaMatrixArg(), d_nextActI.getCudaMatrixArg(),
		d_nextError.getCudaMatrixArg());
	Matrix<DTYPE> t_nextError(d_nextError);
	/*printf("AvgPool NextError: ");
	for (int actMap = 0; actMap < d_nextError.mdim.adim; ++actMap) {
		printf("%d %5.5f \t", actMap, t_nextError.getElem(7, 7, actMap));
	}
	printf("\n");*/
	cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		// Check for any errors launching the kernel
		fprintf(stderr, "complexAvgPoolBackprop launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaSuccess;
}