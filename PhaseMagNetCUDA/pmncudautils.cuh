
#ifndef PMNCUDAUTILS_CUH
#define PMNCUDAUTILS_CUH

#include <algorithm>
#include <assert.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
#define VEC_SIZE 32
#define PAD 2
#define FILTER_DIM 5

#define LRN_RATE 0.0001f

// For weight initialization https://stackoverflow.com/questions/686353/random-float-number-generation
#define WGT_HIGH 0.01f
#define WGT_LOW 0.0f

#define PI 3.14159265f

class NotImplementedException : public std::logic_error
{
public:
	virtual char const* what() const {
		return "Function not yet implemented";
	}
};

enum class LayerType { input, fc, conv, maxpool, avgpool };
enum class ActivationType { relu, softmax };

typedef float DTYPE; // data type for neural activation and weights
typedef unsigned char uchar;

// forward declarations
template <typename T>
struct Matrix;
template <typename T>
struct CudaMatrix;

// add adim, remove stride
struct MatrixDim {
	size_t adim, rdim, cdim, size, astride, rstride;
	MatrixDim(const size_t rdim, const size_t cdim, const size_t size_of_data_type, const size_t adim = 1) :
		adim(adim), rdim(rdim), cdim(cdim), size(adim * rdim * cdim * size_of_data_type), astride(rdim*cdim), rstride(cdim) {}
	MatrixDim() : adim(1), rdim(1), cdim(1), size(1), astride(1), rstride(1) {};
	bool operator==(const MatrixDim& other) const {
		return (adim == other.adim && rdim == other.rdim && cdim == other.cdim && size == other.size);
	}
	const size_t getNumElems() const {
		return adim * rdim * cdim;
	}
};

struct ConvParams {
	MatrixDim filterDim;
	size_t stride, pad, numFilters;
	/*
		Conv - set stride = 0, pad = 0, filterDim to 5x5, numFilters configurable
		Avgpool - stride configurable, pad = 0, filterDim.rdim,cdim = stride, filterDim.adim = 1, numFilters must equal that of prev
	*/
	MatrixDim getNextActDim(const MatrixDim & prevActDim, const size_t sizeOfDtype) {
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
	const size_t getNumElems(void) const {
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
	void addMe(const Matrix<T>& addFrom) {
		assert(mdim == addFrom.mdim);
		for (size_t i = 0; i < getNumElems(); ++i) {
			data[i] += addFrom.data[i];
		}
	}
	const size_t getInd(const size_t& row, const size_t& col, const size_t& aisle = 0) const {
		return (aisle * mdim.rdim * mdim.cdim) + (row * mdim.cdim) + col;
	}
	const T& getElem(const size_t& row, const size_t& col, const size_t& aisle = 0) const {
		return data[getInd(row, col, aisle)];
	}
	void setElem(const size_t& row, const size_t& col, const T& value, const size_t& aisle = 0) {
		data[getInd(row, col, aisle)] = value; // all host side, should be cdim because it is typed
	}
	const T& getElemFlatten(const size_t& ind) const {
		assert(ind < getNumElems());
		return data[ind];
	}
	void setElemFlatten(const size_t& ind, const T& value) {
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
		std::swap(mdim, other.mdim);
		std::swap(data, other.data);
		assert(false);
		return *this;
	}
	~CudaMatrix() {
		cudaFree(data);
		data = nullptr;
	}
	CudaMatrixArg<T> getCudaMatrixArg() {
		CudaMatrixArg<T> temp;
		temp.mdim = mdim;
		temp.data = data;
		return temp;
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
	Matrix<DTYPE> layerDataR; // default to n-vector, (1, N)
	Matrix<DTYPE> layerDataI;
	Matrix<DTYPE> errorData;
	Matrix<DTYPE> biasR;
	Matrix<DTYPE> biasI;
	Matrix<DTYPE>* weightsPrevR; // list of filters if conv
	Matrix<DTYPE>* weightsPrevI; // list of filters if conv
	Matrix<DTYPE>* weightsNextR; // same as prev
	Matrix<DTYPE>* weightsNextI;
	Layer(const LayerParams& lp) :
		layParams(lp),
		layerDataR(lp.matDim),
		layerDataI(lp.matDim),
		errorData(lp.matDim),
		biasR(lp.matDim),
		biasI(lp.matDim),
		weightsPrevR(nullptr),
		weightsPrevI(nullptr),
		weightsNextR(nullptr),
		weightsNextI(nullptr)
	{
		// account for layer
		biasR.fillRandom(-0.01, 0.01);
		biasI.fillRandom(-0.01, 0.01);
	}
	Layer(const Layer& toCopy) :
		layParams(toCopy.layParams),
		layerDataR(toCopy.layerDataR),
		layerDataI(toCopy.layerDataI),
		errorData(toCopy.errorData),
		biasR(toCopy.biasR),
		biasI(toCopy.biasI),
		weightsPrevR(toCopy.weightsPrevR), // shallow copy
		weightsPrevI(toCopy.weightsPrevI),
		weightsNextR(toCopy.weightsNextR),
		weightsNextI(toCopy.weightsNextI)
	{
		
	}
	Layer& operator=(Layer other) {
		if (&other != this) {
			std::swap(layParams, other.layParams);
			std::swap(layerDataR, other.layerDataR);
			std::swap(layerDataI, other.layerDataI);
			std::swap(errorData, other.errorData);
			std::swap(biasR, other.biasR);
			std::swap(biasI, other.biasI);
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
	void initializeWeightsPrev(const MatrixDim& matDim, const size_t numSets = 1) {
		if (numSets == 0)
			return; /* --- for MaxPool, AvgPool --- */
		weightsPrevR = new Matrix<DTYPE>[numSets];
		weightsPrevI = new Matrix<DTYPE>[numSets];
		for (unsigned int s = 0; s < numSets; ++s) {
			weightsPrevR[s] = Matrix<DTYPE>(matDim);
			weightsPrevI[s] = Matrix<DTYPE>(matDim);
			for (unsigned int i = 0; i < matDim.getNumElems(); ++i) {
				// random number generator to initialize weights
				DTYPE max_weight = 2.0 / ((DTYPE)matDim.rdim);
				weightsPrevR[s].data[i] = static_cast <DTYPE> (rand()) / (static_cast <DTYPE> (RAND_MAX / max_weight));
				weightsPrevI[s].data[i] = static_cast <DTYPE> (rand()) / (static_cast <DTYPE> (RAND_MAX / max_weight));
			}
		}
	}
	void linkWeightsNext(const Layer* const layptr) {
		weightsNextR = layptr->weightsPrevR;
		weightsNextI = layptr->weightsPrevI;
	}
	Matrix<DTYPE>* getWeightsPrevR() const {
		return weightsPrevR;
	}
	Matrix<DTYPE>* getWeightsPrevI() const {
		return weightsPrevI;
	}
	Matrix<DTYPE>* getWeightsNextR() const {
		return weightsNextR;
	}
	Matrix<DTYPE>* getWeightsNextI() const {
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