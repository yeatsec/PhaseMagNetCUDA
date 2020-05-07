
#ifndef PMNCUDAUTILS_CUH
#define PMNCUDAUTILS_CUH

#include <algorithm>
#include <assert.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
#define VEC_SIZE 256

#define LRN_RATE 0.005

// For weight initialization https://stackoverflow.com/questions/686353/random-float-number-generation
#define WGT_HIGH 0.01
#define WGT_LOW 0.0

enum class LayerType { input, fc, conv, maxpool };
enum class ActivationType { relu, softmax };

typedef float DTYPE; // data type for neural activation and weights
typedef unsigned char uchar;

// forward declarations
template <typename T>
struct Matrix;
template <typename T>
struct CudaMatrix;

struct MatrixDim {
	size_t rdim, cdim, size, stride;
	MatrixDim(const size_t rdim, const size_t cdim, const size_t size_of_data_type) :
		rdim(rdim), cdim(cdim), size(rdim * cdim * size_of_data_type), stride(cdim) {}
	MatrixDim() : rdim(1), cdim(1), size(1), stride(1) {};
	bool operator==(const MatrixDim& other) const {
		return (rdim == other.rdim && cdim == other.cdim && size == other.size && stride == other.stride);
	}
};

template <typename T>
struct Matrix {
	MatrixDim mdim;
	T* data;
	Matrix() {
		mdim.rdim = 0;
		mdim.cdim = 0;
		mdim.size = 0;
		mdim.stride = 0;
		data = nullptr;
	}
	Matrix(const MatrixDim& mdim) : mdim(mdim) {
		data = new T[mdim.rdim*mdim.cdim];
	}
	Matrix(const Matrix<T>& toCopy) : mdim(toCopy.mdim) {
		size_t numElems = mdim.rdim * mdim.cdim;
		data = new T[numElems];
		std::memcpy(data, toCopy.data, mdim.size);
	}
	Matrix<T>& operator=(Matrix<T> other) {
		std::swap(mdim, other.mdim);
		std::swap(data, other.data);
		return *this;
	}
	~Matrix() {
		delete[] data;
	}
	void fill(const T& value) {
		for (size_t r = 0; r < mdim.rdim; ++r) {
			for (size_t c = 0; c < mdim.cdim; ++c) {
				data[r*mdim.cdim + c] = value;
			}
		}
	}
	void fillFromCuda(const CudaMatrix<T>& copyFrom) {
		assert(copyFrom.mdim == mdim);
		cudaError_t err = cudaMemcpy(data, copyFrom.data, mdim.size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy Matrix.fillFromCuda failed!\n");
		}
	}
	void fillFromMatrix(const Matrix<T>& copyFrom) {
		assert(copyFrom.mdim == mdim);
		std::memcpy(data, copyFrom.data, mdim.size);
	}
	void fillFromUbyte(const uchar* const ucharptr) {
		for (size_t i = 0; i < mdim.cdim * mdim.rdim; ++i) {
			data[i] = (DTYPE)(ucharptr[i]) / 255.0;
		}
	}
	Matrix<T> transpose() {
		Matrix<T> temp(mdim);
		std::swap(temp.mdim.rdim, temp.mdim.cdim);
		temp.mdim.stride = temp.mdim.cdim;
		for (size_t tr = 0; tr < temp.mdim.rdim; ++tr) {
			for (size_t tc = 0; tc < temp.mdim.cdim; ++tc) {
				temp.data[tr * temp.mdim.cdim + tc] = data[tc * mdim.cdim + tr];
			}
		}
		return temp;
	}
	void forEach(void func(T*)) {
		for (size_t i = 0; i < mdim.rdim * mdim.cdim; ++i)
			func(&(data[i]));
	}
	static Matrix<T> pointwiseOp(const Matrix<T>& A, const Matrix<T>& B, T func(const T, const T)) {
		assert(A.mdim == B.mdim); // A and B must have same dimensions
		Matrix<T> temp(A.mdim);
		for (size_t i = 0; i < temp.mdim.rdim * temp.mdim.cdim; ++i) {
			temp.data[i] = func(A.data[i], B.data[i]);
		}
		return temp;
	}
	void addMe(const Matrix<T>& addFrom) {
		assert(mdim == addFrom.mdim);
		for (size_t i = 0; i < mdim.rdim * mdim.cdim; ++i) {
			data[i] += addFrom.data[i];
		}
	}
	const T& getElem(const size_t& row, const size_t& col) {
		return data[row * mdim.cdim + col];
	}
	void setElem(const size_t& row, const size_t& col, const T& value) {
		data[row * mdim.cdim + col] = value; // all host side, should be cdim because it is typed
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
		mdim.rdim = 0;
		mdim.cdim = 0;
		mdim.size = 0;
		mdim.stride = 0;
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
			fprintf(stderr, "cudaMalloc CudaMatrix(Matrix) failed!\n");
		}
		err = cudaMemcpy(data, hostMat.data, mdim.size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy CudaMatrix(Matrix) failed!\n");
		}
	}
	CudaMatrix(const CudaMatrix<T>& toCopy) : mdim(toCopy.mdim) {
		cudaError_t err = cudaMalloc(&data, mdim.size);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMalloc CudaMatrix failed!\n");
		}
		err = cudaMemcpy(data, toCopy.data, mdim.size, cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy CudaMatrix failed!\n");
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
	LayerParams(const LayerType lt, const ActivationType at, const MatrixDim dim) :
		layType(lt),
		actType(at),
		matDim(dim) {}
};

struct Layer {
	LayerParams layParams;
	Matrix<DTYPE> layerData; // default to n-vector, (1, N)
	Matrix<DTYPE> errorData;
	Matrix<DTYPE> bias;
	Matrix<DTYPE>* weightsPrev;
	Matrix<DTYPE>* weightsNext;
	Layer(const LayerParams& lp) :
		layParams(lp),
		layerData(lp.matDim),
		errorData(lp.matDim),
		bias(lp.matDim),
		weightsPrev(nullptr),
		weightsNext(nullptr)
	{
		bias.fill(0);
	}
	Layer(const Layer& toCopy) :
		layParams(toCopy.layParams),
		layerData(toCopy.layerData),
		errorData(toCopy.errorData),
		bias(toCopy.bias),
		weightsPrev(toCopy.weightsPrev), // shallow copy
		weightsNext(toCopy.weightsNext)
	{}
	Layer& operator=(Layer other) {
		std::swap(layParams, other.layParams);
		std::swap(layerData, other.layerData);
		std::swap(errorData, other.errorData);
		std::swap(bias, other.bias);
		return *this;
	}
	~Layer() {
		// delete weightsPrev;
		// weightsNext is not "owned" by this layer
		weightsPrev = nullptr;
		weightsNext = nullptr;
	}
	void freeWeightsPrev(void) {
		delete weightsPrev;
		weightsPrev = nullptr;
	}
	void initializeWeightsPrev(const MatrixDim& matDim) {
		weightsPrev = new Matrix<DTYPE>(matDim);
		for (unsigned int i = 0; i < matDim.cdim*matDim.rdim; ++i) {
			// random number generator to initialize weights
			DTYPE max_weight = 2.0 / ((DTYPE)matDim.rdim);
			(weightsPrev->data)[i] = static_cast <DTYPE> (rand()) / (static_cast <DTYPE> (RAND_MAX / max_weight));
		}
	}
	void linkWeightsNext(Matrix<DTYPE>* const wgtptr) {
		weightsNext = wgtptr;
	}
	Matrix<DTYPE>* getWeightsPrev() const {
		return weightsPrev;
	}
	Matrix<DTYPE>* getWeightsNext() const {
		return weightsNext;
	}
};

template <typename T>
__device__ const T getElem(const CudaMatrixArg<T>& mat, const size_t& row, const size_t& col) {
	return mat.data[row * mat.mdim.cdim + col];
}

template <typename T>
__device__ void setElem(CudaMatrixArg<T>& mat, const size_t& row, const size_t& col, const T& value) {
	mat.data[row * mat.mdim.cdim + col] = value;
}

#endif //PMNCUDAUTILS_CUH