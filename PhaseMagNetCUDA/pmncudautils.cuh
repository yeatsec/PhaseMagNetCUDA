

#ifndef PMNCUDAUTILS_CUH
#define PMNCUDAUTILS_CUH

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BLOCK_SIZE 16
#define VEC_SIZE 32

// For weight initialization https://stackoverflow.com/questions/686353/random-float-number-generation


#define PI 3.14159265f

enum class LayerType { input, fc, conv, maxpool, avgpool, phasorconv};
enum class ActivationType { relu, sigmoid, softmax };


typedef float DTYPE; // data type for neural activation and weights
typedef unsigned char uchar;


using scalarActFunc = DTYPE(*) (DTYPE);



// forward declarations
template <typename T>
struct Matrix;
template <typename T>
struct CudaMatrix;

// add adim, remove stride
struct MatrixDim {
	unsigned int adim, rdim, cdim, size, astride, rstride;
	MatrixDim(const unsigned int rdim, const unsigned int cdim, const unsigned int size_of_data_type, const unsigned int adim = 1) :
		adim(adim), rdim(rdim), cdim(cdim), size(adim * rdim * cdim * size_of_data_type), astride(rdim*cdim), rstride(cdim) {}
	MatrixDim() : adim(1), rdim(1), cdim(1), size(1), astride(1), rstride(1) {};
	bool operator==(const MatrixDim& other) const {
		return (adim == other.adim && rdim == other.rdim && cdim == other.cdim && size == other.size);
	}
	unsigned int getNumElems() const {
		return adim * rdim * cdim;
	}
	MatrixDim transpose(void) {
		MatrixDim temp;
		temp.adim = adim;
		temp.rdim = cdim;
		temp.cdim = rdim;
		temp.size = size;
		temp.astride = astride;
		temp.rstride = temp.cdim;
		return temp;
	}
};

struct ConvParams {
	MatrixDim filterDim;
	unsigned int stride, pad, numFilters;
	ConvParams() : filterDim(), stride(0), pad(0), numFilters(1) {}
	/*
		Conv - set stride = 0, pad = 0, filterDim to 5x5, numFilters configurable
		Avgpool - stride configurable, pad = 0, filterDim.rdim,cdim = stride, filterDim.adim = 1, numFilters must equal that of prev
	*/
	MatrixDim getNextActDim(const MatrixDim & prevActDim, const unsigned int sizeOfDtype) {
		assert(prevActDim.adim == filterDim.adim);
		MatrixDim outDim;
		outDim.adim = numFilters;
		outDim.rdim = 1 + (prevActDim.rdim + 2 * pad - filterDim.rdim) / stride;
		outDim.cdim = 1 + (prevActDim.cdim + 2 * pad - filterDim.cdim) / stride;
		outDim.astride = outDim.rdim * outDim.cdim;
		outDim.rstride = outDim.cdim;
		outDim.size = outDim.adim * outDim.rdim * outDim.cdim * sizeOfDtype;
		return outDim;
	}
};

template <typename T>
struct Matrix {
	MatrixDim mdim;
	T* data;
	Matrix() {
		mdim.adim = 0;
		mdim.rdim = 0;
		mdim.cdim = 0;
		mdim.size = 0;
		mdim.astride = 0;
		mdim.rstride = 0;
		data = nullptr;
	}
	Matrix(const MatrixDim& mdim) : mdim(mdim) {
		data = new T[getNumElems()];
	}
	Matrix(const Matrix<T>& toCopy) : mdim(toCopy.mdim) {
		size_t numElems = getNumElems();
		data = new T[numElems];
		std::memcpy(data, toCopy.data, mdim.size);
	}
	Matrix(const CudaMatrix<T>& toCopy) : mdim(toCopy.mdim) {
		size_t numElems = getNumElems();
		data = new T[numElems];
		cudaError_t err = cudaMemcpy(data, toCopy.data, mdim.size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy Matrix(CudaMatrix) failed! %d \n", err);
		}
	}
	Matrix<T>& operator=(Matrix<T> other) {
		if (&other != this) {
			std::swap(mdim, other.mdim);
			std::swap(data, other.data);
		}
		return *this;
	}
	~Matrix() {
		delete[] data;
	}
	unsigned int getNumElems(void) const {
		return mdim.getNumElems();
	}
	void fill(const T& value) {
		for (size_t i = 0; i < getNumElems(); ++i)
			data[i] = value;
	}
	void fillFromCuda(const CudaMatrix<T>& copyFrom) {
		assert(copyFrom.mdim == mdim);
		cudaError_t err = cudaMemcpy(data, copyFrom.data, mdim.size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy Matrix.fillFromCuda failed! %d \n", err);
		}
	}
	void fillFromMatrix(const Matrix<T>& copyFrom) {
		assert(copyFrom.mdim == mdim);
		std::memcpy(data, copyFrom.data, mdim.size);
	}
	void fillFromUbyte(const uchar* const ucharptr) {
		for (size_t i = 0; i < getNumElems(); ++i) {
			data[i] = (DTYPE)(ucharptr[i]) / 255.0f;
		}
	}
	void dumpToUbyte(uchar* ucharptr) {
		for (unsigned int i = 0; i < getNumElems(); ++i) {
			ucharptr[i] = ((uchar)(data[i] * 255.0f));
		}
	}
	void fillRandom(T minval, T maxval) {
		for (size_t i = 0; i < getNumElems(); ++i)
			data[i] = minval + static_cast <T> (rand()) / (static_cast <T> (RAND_MAX / (maxval-minval)));
	}
	Matrix<T> transpose() {
		assert(mdim.adim == 1);
		Matrix<T> temp(mdim);
		std::swap(temp.mdim.rdim, temp.mdim.cdim);
		temp.mdim.rstride = temp.mdim.cdim;
		for (size_t tr = 0; tr < temp.mdim.rdim; ++tr) {
			for (size_t tc = 0; tc < temp.mdim.cdim; ++tc) {
				temp.data[tr * temp.mdim.cdim + tc] = data[tc * mdim.cdim + tr];
			}
		}
		return temp;
	}
	void forEach(void func(T*)) {
		for (size_t i = 0; i < getNumElems(); ++i)
			func(&(data[i]));
	}
	static Matrix<T> pointwiseOp(const Matrix<T>& A, const Matrix<T>& B, T func(const T, const T)) {
		assert(A.mdim == B.mdim); // A and B must have same dimensions
		Matrix<T> temp(A.mdim);
		for (size_t i = 0; i < A.getNumElems(); ++i) {
			temp.data[i] = func(A.data[i], B.data[i]);
		}
		return temp;
	}
	void pointwiseOp(const Matrix<T>& other, T func(const T, const T)) {
		assert(mdim == other.mdim);
		for (size_t i = 0; i < getNumElems(); ++i) {
			data[i] = func(data[i], other.data[i]);
		}
	}
	void addMe(const Matrix<T>& addFrom) {
		assert(mdim == addFrom.mdim);
		for (size_t i = 0; i < getNumElems(); ++i) {
			data[i] += addFrom.data[i];
		}
	}
	unsigned int getInd(const unsigned int& row, const unsigned int& col, const unsigned int& aisle = 0) const {
		return (aisle * mdim.rdim * mdim.cdim) + (row * mdim.cdim) + col;
	}
	const T& getElem(const unsigned int& row, const unsigned int& col, const unsigned int& aisle = 0) const {
		return data[getInd(row, col, aisle)];
	}
	void setElem(const unsigned int& row, const unsigned int& col, const T& value, const unsigned int& aisle = 0) {
		data[getInd(row, col, aisle)] = value; // all host side, should be cdim because it is typed
	}
	const T& getElemFlatten(const unsigned int& ind) const {
		assert(ind < getNumElems());
		return data[ind];
	}
	void setElemFlatten(const unsigned int& ind, const T& value) {
		data[ind] = value;
	}
};

template <typename T>
struct CudaMatrixArg {
	MatrixDim mdim;
	T* data;
	CudaMatrixArg() : mdim(), data(nullptr) {};
};

template <typename T>
struct CudaMatrix {
	MatrixDim mdim;
	T* data;
	CudaMatrix() {
		mdim.adim = 0;
		mdim.rdim = 0;
		mdim.cdim = 0;
		mdim.size = 0;
		mdim.astride = 0;
		mdim.rstride = 0;
		data = nullptr;
	}
	CudaMatrix(const MatrixDim& mdim) : mdim(mdim) {
		cudaError_t err = cudaMalloc(&data, mdim.size);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc CudaMatrix(mdim) failed!\n");
		}
	}
	CudaMatrix(const Matrix<T>& hostMat) : mdim(hostMat.mdim) { // ctor from Matrix object
		cudaError_t err = cudaMalloc(&data, mdim.size);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc CudaMatrix(Matrix) failed! %d \n", err);
		}
		err = cudaMemcpy(data, hostMat.data, mdim.size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy CudaMatrix(Matrix) failed! %d \n", err);
		}
	}
	CudaMatrix(const CudaMatrix<T>& toCopy) : mdim(toCopy.mdim) {
		cudaError_t err = cudaMalloc(&data, mdim.size);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc CudaMatrix failed! %d \n", err);
		}
		err = cudaMemcpy(data, toCopy.data, mdim.size, cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy CudaMatrix failed! %d \n", err);
		}
	}
	CudaMatrix<T>& operator=(CudaMatrix<T> other) {
		if (&other != this) {
			std::swap(mdim, other.mdim);
			std::swap(data, other.data);
		}
		return *this;
	}
	~CudaMatrix() {
		cudaFree(data);
		data = nullptr;
	}
	CudaMatrixArg<T> getCudaMatrixArg() const { // this is a bit misleading
		CudaMatrixArg<T> temp;
		temp.mdim = mdim;
		temp.data = data;
		return temp;
	}
	void fillFromMatrix(Matrix<DTYPE>& hostMat) {
		assert(mdim == hostMat.mdim);
		cudaError_t err = cudaMemcpy(data, hostMat.data, mdim.size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy CudaMatrix::fillFromMatrix failed! %d \n", err);
		}
	}
};

struct LayerParams {
	LayerType layType;
	ActivationType actType;
	MatrixDim matDim;
	ConvParams convParams;
	LayerParams(const LayerType lt, const ActivationType at, const MatrixDim dim, const ConvParams convParams) :
		layType(lt),
		actType(at),
		matDim(dim),
		convParams(convParams) {}
	LayerParams(const LayerType lt, const ActivationType at, const MatrixDim dim) :
		layType(lt),
		actType(at),
		matDim(dim),
		convParams() {}
};

struct Layer {
	LayerParams layParams;
	CudaMatrix<DTYPE> layerData; // default to n-vector, (1, N)
	CudaMatrix<DTYPE> layerDataAng;
	CudaMatrix<DTYPE> errorData;
	CudaMatrix<DTYPE> bias;
	curandState* layerRNG;
	CudaMatrix<DTYPE>* weightsPrevR; // list of filters if conv
	CudaMatrix<DTYPE>* weightsPrevI; // list of filters if conv
	CudaMatrix<DTYPE>* weightsNextR; // same as prev
	CudaMatrix<DTYPE>* weightsNextI;
	Layer(const LayerParams& lp) :
		layParams(lp),
		layerData(lp.matDim),
		layerDataAng(lp.matDim),
		errorData(lp.matDim),
		bias(lp.matDim),
		layerRNG(0),
		weightsPrevR(nullptr),
		weightsPrevI(nullptr),
		weightsNextR(nullptr),
		weightsNextI(nullptr)
	{
		// account for layer
		Matrix<DTYPE> temp(bias.mdim);
		temp.fill(0.0); // He initialization
		bias.fillFromMatrix(temp);
	}
	Layer(const Layer& toCopy) :
		layParams(toCopy.layParams),
		layerData(toCopy.layerData),
		layerDataAng(toCopy.layerDataAng),
		errorData(toCopy.errorData),
		bias(toCopy.bias),
		weightsPrevR(toCopy.weightsPrevR), // shallow copy
		weightsPrevI(toCopy.weightsPrevI),
		weightsNextR(toCopy.weightsNextR),
		weightsNextI(toCopy.weightsNextI)
	{
		if (toCopy.layerRNG != 0) {
			cudaError_t cudaStatus = cudaMalloc(&layerRNG, sizeof(curandState) * layerData.mdim.getNumElems());
			if (cudaStatus != cudaSuccess) {
				printf("layer curand allocate copyctor failed! %s\n", cudaGetErrorString(cudaStatus));
			}
			cudaStatus = cudaMemcpy(layerRNG, toCopy.layerRNG, sizeof(curandState) * layerData.mdim.getNumElems(), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) {
				printf("layer curand Memcpy copyctor failed! %s\n", cudaGetErrorString(cudaStatus));
			}
		}
		else {
			layerRNG = 0;
		}
	}
	Layer& operator=(Layer other) {
		if (&other != this) {
			std::swap(layParams, other.layParams);
			std::swap(layerData, other.layerData);
			std::swap(layerDataAng, other.layerDataAng);
			std::swap(errorData, other.errorData);
			std::swap(bias, other.bias);
			weightsPrevR = other.weightsPrevR; // shallow copy
			weightsPrevI = other.weightsPrevI;
			weightsNextR = other.weightsNextR;
			weightsNextI = other.weightsNextI;
		}
		return *this;
	}
	~Layer() {
		// compiler will invoke dtor on individual matrices
		// delete weightsPrev;
		// weightsNext is not "owned" by this layer
		cudaFree(layerRNG);
		weightsPrevR = nullptr;
		weightsPrevI = nullptr;
		weightsNextR = nullptr;
		weightsNextI = nullptr;
	}
	void freeWeightsPrev(void) {
		delete[] weightsPrevR;
		delete[] weightsPrevI;
		weightsPrevR = nullptr;
		weightsPrevI = nullptr;
	}
	void initializeRNG(unsigned long long seed) {
		assert(layerRNG == 0); // no reassigning layerRNG
		cudaError_t cudaStatus = cudaMalloc(&layerRNG, sizeof(curandState) * layerData.mdim.getNumElems());
		if (cudaStatus != cudaSuccess) {
			printf("layer curand allocate failed! %s\n", cudaGetErrorString(cudaStatus));
		}
	}
	void initializeWeightsPrev(const MatrixDim& matDim, const size_t numSets = 1) {
		if (numSets == 0)
			return; /* --- for MaxPool, AvgPool --- */
		weightsPrevR = new CudaMatrix<DTYPE>[numSets];
		weightsPrevI = new CudaMatrix<DTYPE>[numSets];
		Matrix<DTYPE> tempR(matDim);
		Matrix<DTYPE> tempI(matDim);
		for (unsigned int s = 0; s < numSets; ++s) {
			weightsPrevR[s] = CudaMatrix<DTYPE>(matDim);
			weightsPrevI[s] = CudaMatrix<DTYPE>(matDim);
			DTYPE denom, r1, sign;
			if (numSets > 1) { // indicates convolution
				denom = ((DTYPE)(matDim.getNumElems()));
				for (unsigned int i = 0; i < matDim.getNumElems(); ++i) {
					// random number generator to initialize weights
					DTYPE ang = 2* PI * ((static_cast <DTYPE> (rand())) / (static_cast<DTYPE> (RAND_MAX)));
					DTYPE mag = ((static_cast <DTYPE> (rand())) / (static_cast<DTYPE> (RAND_MAX)));
					tempR.data[i] = mag * sqrtf(2.0f / denom) * cosf(ang);
					tempI.data[i] = mag * sqrtf(2.0f / denom) * sinf(ang);
				}
			}
			else { // indicates FC
				denom = ((DTYPE) (matDim.rdim));
				for (unsigned int i = 0; i < matDim.getNumElems(); ++i) {
					// random number generator to initialize weights
					r1 = ((static_cast <DTYPE> (rand())) / (static_cast<DTYPE> (RAND_MAX)));
					sign = (rand() > RAND_MAX / 2) ? 1.0f : -1.0f;
					tempR.data[i] = sign * r1 * sqrtf(2.0f / denom);
					sign = (rand() > RAND_MAX / 2) ? 1.0f : -1.0f;
					r1 = ((static_cast <DTYPE> (rand())) / (static_cast<DTYPE> (RAND_MAX)));
					tempI.data[i] = sign * r1 * sqrtf(2.0f / denom);
				}
			}
			weightsPrevR[s].fillFromMatrix(tempR);
			weightsPrevI[s].fillFromMatrix(tempI);
		}
	}
	void linkWeightsNext(const Layer* const layptr) {
		weightsNextR = layptr->weightsPrevR;
		weightsNextI = layptr->weightsPrevI;
	}
	CudaMatrix<DTYPE>* getWeightsPrevR() const {
		return weightsPrevR;
	}
	CudaMatrix<DTYPE>* getWeightsPrevI() const {
		return weightsPrevI;
	}
	CudaMatrix<DTYPE>* getWeightsNextR() const {
		return weightsNextR;
	}
	CudaMatrix<DTYPE>* getWeightsNextI() const {
		return weightsNextI;
	}
};

template <typename T>
__device__ const T getElem(const CudaMatrixArg<T>& mat, const size_t& row, const size_t& col, const size_t& aisle = 0) {
	int ind = (aisle * mat.mdim.astride) + (row * mat.mdim.rstride) + col;
	return mat.data[ind];
}

template <typename T>
__device__ void setElem(CudaMatrixArg<T>& mat, const size_t& row, const size_t& col, const T& value, const size_t& aisle = 0) {
	int ind = (aisle * mat.mdim.astride) + (row * mat.mdim.rstride) + col;
	mat.data[ind] = value;
}

template <typename T>
__device__ const T getElemFlatten(const CudaMatrixArg<T>& mat, const size_t& col) {
	return mat.data[col];
}

template <typename T>
__device__ void setElemFlatten(CudaMatrixArg<T>& mat, const size_t& col, const T& value) {
	mat.data[col] = value;
}

#endif //PMNCUDAUTILS_CUH