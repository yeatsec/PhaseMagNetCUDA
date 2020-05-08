

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

	size_t num_a_its = Ar.mdim.cdim / VEC_SIZE;
	if (Ar.mdim.cdim % VEC_SIZE != 0)
		++num_a_its;

	__shared__ DTYPE Ar_s[VEC_SIZE];
	__shared__ DTYPE Ai_s[VEC_SIZE];
	for (size_t a_it = 0; a_it < num_a_its; ++a_it) { // select block of A to do 
		size_t A_col = a_it * VEC_SIZE + threadIdx.x;
		if (A_col < Ar.mdim.cdim) { // don't index A_col if past end
			Ar_s[threadIdx.x] = getElem(Ar, 0, A_col);
			Ai_s[threadIdx.x] = getElem(Ai, 0, A_col);
		}
		__syncthreads(); // all threads must load in A before we can use it to compute
		if (col < Cr.mdim.cdim) { // don't continue if you're past the end of C
			for (size_t dot_it = 0; dot_it < VEC_SIZE; ++dot_it) {
				size_t b_row_ind = a_it * VEC_SIZE + dot_it;
				if (b_row_ind < Ar.mdim.cdim) { // don't index into b past its row dimension. also implies As[dot_it] will not be accessed
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

	setElem(Cr, 0, col, Cvalue_r);
	setElem(Ci, 0, col, Cvalue_i);
}

// helper function for MatMul
cudaError_t vecMatMultWithCuda(const Matrix<DTYPE>& Ar, const Matrix<DTYPE>& Ai, 
	const Matrix<DTYPE>& Br, const Matrix<DTYPE>& Bi, 
	Matrix<DTYPE>& Cr, Matrix<DTYPE>& Ci) {
	assert(Ar.mdim.cdim == Br.mdim.rdim && Ar.mdim.rdim == Cr.mdim.rdim && Br.mdim.cdim == Cr.mdim.cdim); // check mult compat
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

	// invoke kernel
	unsigned int num_vecs = d_Cr.mdim.cdim / VEC_SIZE;
	if (d_Cr.mdim.cdim % VEC_SIZE != 0)
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

__device__ void get_dLdMag_dLdPhi(DTYPE Yr, DTYPE Yi, DTYPE wxr, DTYPE wxi, DTYPE err, DTYPE& dLdMag, DTYPE& dLdPhi) {
	DTYPE dPhi = atan2f(wxi, wxr) - atan2f(Yi - wxi, Yr - wxr); // dPhi = ang(wx) - ang(Y-wx)
	DTYPE abswx = d_abs2(wxr, wxi);
	DTYPE absY = d_abs2(Yr, Yi);
	DTYPE abswx_Y = d_abs2(Yr - wxr, Yi - wxi);
	dLdMag = ((abswx + (abswx_Y * cosf(dPhi))) / absY) * err;
	dLdPhi = (-1.0f) * ((abswx * abswx_Y * sinf(dPhi)) / absY) * err; // radians
}

__global__ void complexBackpropKernel(const CudaMatrixArg<DTYPE> prevActR, const CudaMatrixArg<DTYPE> prevActI,
	CudaMatrixArg<DTYPE> prevError, CudaMatrixArg<DTYPE> weightsR, CudaMatrixArg<DTYPE> weightsI,
	const CudaMatrixArg<DTYPE> nextActR, const CudaMatrixArg<DTYPE> nextActI, const CudaMatrixArg<DTYPE> nextError) {

	unsigned int prevError_col = threadIdx.x + (blockIdx.x * VEC_SIZE); // column of prevError "owned" by the thread
	unsigned int num_nextError_its = nextError.mdim.cdim / VEC_SIZE;
	DTYPE prevError_val = 0.0;
	if (nextError.mdim.cdim % VEC_SIZE != 0) {
		++num_nextError_its;
	}
	// no need to reallocate shared each iteration
	__shared__ DTYPE nextError_s[VEC_SIZE];
	__shared__ DTYPE nextActR_s[VEC_SIZE];
	__shared__ DTYPE nextActI_s[VEC_SIZE];
	for (unsigned int nextError_it = 0; nextError_it < num_nextError_its; ++nextError_it) {
		unsigned int own_nextErr_col = nextError_it * VEC_SIZE + threadIdx.x;
		if (own_nextErr_col < nextError.mdim.cdim) {
			nextError_s[threadIdx.x] = getElem(nextError, 0, own_nextErr_col);
			nextActR_s[threadIdx.x] = getElem(nextActR, 0, own_nextErr_col);
			nextActI_s[threadIdx.x] = getElem(nextActI, 0, own_nextErr_col);
		}
		__syncthreads();
		// iterate through weights along row axis (nextError dim) to MAC into own prevError value with w(prevError_col, ...)*nextError(...)
		for (unsigned int dot_it = 0; dot_it < VEC_SIZE; ++dot_it) {
			unsigned int wgt_col = nextError_it * VEC_SIZE + dot_it;
			if (prevError_col < prevError.mdim.cdim && wgt_col < nextError.mdim.cdim) {
				DTYPE wr = getElem(weightsR, prevError_col, wgt_col);
				DTYPE wi = getElem(weightsI, prevError_col, wgt_col);
				DTYPE xr = getElem(prevActR, 0, prevError_col);
				DTYPE xi = getElem(prevActI, 0, prevError_col);
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
	if (prevError_col < prevError.mdim.cdim) {
		setElem(prevError, 0, prevError_col, prevError_val);
	}
}

__global__ void complexUpdateBiasKernel(const CudaMatrixArg<DTYPE> actR, const CudaMatrixArg<DTYPE> actI,
	const CudaMatrixArg<DTYPE> error, CudaMatrixArg<DTYPE> biasR, CudaMatrixArg<DTYPE> biasI) {
	// each kernel is called on a subvector of activations. update bias according to its contribution to the activation
	// each thread is responsible for one activation + bias + error combination
	unsigned int col = threadIdx.x + (blockIdx.x * VEC_SIZE); // a & b * err combination
	if (col < biasR.mdim.cdim) {
		DTYPE Yr = getElem(actR, 0, col);
		DTYPE Yi = getElem(actI, 0, col);
		DTYPE wr = getElem(biasR, 0, col);
		DTYPE wi = getElem(biasI, 0, col);
		DTYPE err = getElem(error, 0, col);
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
		setElem(biasR, 0, col, wr);
		setElem(biasI, 0, col, wi);
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
	assert(prevActR.mdim.cdim == weightsR.mdim.rdim && weightsR.mdim.cdim == nextActR.mdim.cdim); // transfer
	assert(prevActR.mdim.rdim == nextActR.mdim.rdim); // mult compat

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
	unsigned int num_vecs = d_nextBiasR.mdim.cdim / VEC_SIZE;
	if (d_nextBiasR.mdim.cdim % VEC_SIZE != 0) {
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
	num_vecs = d_prevError.mdim.cdim / VEC_SIZE;
	if (d_prevError.mdim.cdim % VEC_SIZE != 0) {
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