#ifndef PHASEMAGNETCUDA_CUH
#define PHASEMAGNETCUDA_CUH

#include <stdio.h>
#include <stdint.h>
#include <cstring>

#include "pmncudautils.cuh"
#include "LinkedList.cuh"

typedef unsigned char uchar;

class PhaseMagNetCUDA {
private:
	LinkedList<Layer> layers;
	bool initialized;
	void setInput(const uchar* const ucharptr);
	Matrix<DTYPE> getOutput(void) const;
	void forwardPropagate(void);
	void backwardPropagate(const Matrix<DTYPE>& expected, float lrnRate);
	void resetState(void);
public:
	PhaseMagNetCUDA(void);
	void addLayer(const LayerParams& lp);
	void initialize(bool fromFile = false);
	void free(void);
	void train(const size_t num_examples, uchar** inputData, uchar* labels, float lrnRate = 0.01, bool verbose = false);
	Matrix<DTYPE> predict(const size_t num_examples, uchar** inputData, bool verbose = false);
	float evaluate(const size_t num_examples, uchar** inputData, uchar* labels, bool verbose = false);
	void save(const std::string& path);
	void load(const std::string& path);
};

#endif // PHASEMAGNETCUD_CUH