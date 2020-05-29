#ifndef PHASEMAGNETCUDA_CUH
#define PHASEMAGNETCUDA_CUH

#include <stdio.h>
#include <stdint.h>
#include <cstring>

#include "pmncudautils.cuh"
#include "LinkedList.cuh"
#include "read_dataset.cuh"


class PhaseMagNetCUDA {
private:
	LinkedList<Layer> layers;
	bool initialized;
	void setInput(const uchar* const ucharptr);
	Matrix<DTYPE> getOutput(void) const;
	void forwardPropagate(float dropout = 0.0f);
	void backwardPropagate(const Matrix<DTYPE>& expected, float lrnRate, bool adjInput = false, float eps = 0.0f, bool targeted = false);
	void resetState(void);
public:
	PhaseMagNetCUDA(void);
	void addLayer(const LayerParams& lp);
	void initialize(bool fromFile = false);
	void free(void);
	void train(const unsigned int num_examples, uchar** inputData, uchar* labels, float lrnRate = 0.01, bool verbose = false, float dropout = 0.0f);
	void genAdv(const std::string name, const unsigned int num_examples, const unsigned int img_rows, const unsigned int img_cols, uchar** inputData, uchar* labels, float epsilon, int numIts, bool targeted = false, bool randomStart = false, bool verbose = false);
	Matrix<DTYPE> predict(const unsigned int num_examples, uchar** inputData, bool verbose = false);
	float evaluate(const unsigned int num_examples, uchar** inputData, uchar* labels, bool verbose = false);
	void save(const std::string& path);
	void load(const std::string& path);
};

#endif // PHASEMAGNETCUD_CUH