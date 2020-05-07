#ifndef PHASEMAGNETCUDA_CUH
#define PHASEMAGNETCUDA_CUH

#include <stdio.h>
#include <stdint.h>
#include <cstring>

#include "pmncudautils.cuh"
#include "LinkedList.cuh"

class PhaseMagNetCUDA {
private:
	LinkedList<Layer> layers;
public:
	bool initialized;
	void setInput(const Matrix<DTYPE>& input);
	Matrix<DTYPE> getOutput(void) const;
	void forwardPropagate(void);
	void backwardPropagate(const Matrix<DTYPE>& expected);
	void resetState(void);
//public:
	PhaseMagNetCUDA(void);
	void addLayer(const LayerParams& lp);
	void initialize(bool fromFile = false);
	void free(void);
	void train(const size_t num_examples, uchar** inputData, uchar* labels, bool verbose = false);
	Matrix<DTYPE> predict(const size_t num_examples, uchar** inputData);
	float evaluate(const size_t num_examples, uchar** inputData, uchar* labels);
	void save(const std::string& path);
	void load(const std::string& path);
};

#endif // PHASEMAGNETCUD_CUH