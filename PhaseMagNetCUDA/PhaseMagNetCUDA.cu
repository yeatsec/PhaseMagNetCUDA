#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>

#include "PhaseMagNetCUDA.cuh"
#include "cudafuncs.cuh"


/*
	Convention: Activation A<1, N>, Weights <N, M>, Activation <1, M>
*/

PhaseMagNetCUDA::PhaseMagNetCUDA() :
	layers(),
	initialized(false) {}

void PhaseMagNetCUDA::initialize(bool fromFile) {
	assert(!(initialized || layers.isEmpty()));
	// do this to seed the random
	// srand(static_cast <unsigned> (time(0)));
	srand(0);
	// iterate through the layers list and initialize weights
	auto initfunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) { // only do this for weight "owners"
			Layer* elemPtr = (ptr->getElemPtr()); // reference to this element
			LinkedListNode<Layer>* prevNodePtr = ptr->getPrev();
			Layer* prevPtr = prevNodePtr->getElemPtr();
			// rdim=cdim of prev layer, cdim=rdim of this layer
			MatrixDim matDim;
			matDim.rdim = prevPtr->layParams.matDim.cdim;
			matDim.cdim = elemPtr->layParams.matDim.cdim;
			matDim.size = matDim.rdim * matDim.cdim * sizeof(DTYPE);
			matDim.stride = matDim.cdim;
			elemPtr->initializeWeightsPrev(matDim); // weightsprev pointer set
			printf("linked\n");
			prevPtr->linkWeightsNext(elemPtr->getWeightsPrev()); // link ptr
		}
	};
	auto initfromfilefunc = [](LinkedListNode<Layer>* ptr) { // weights already allocated
		if (ptr->hasPrev()) {
			Layer* elemPtr = (ptr->getElemPtr()); // reference to this element
			LinkedListNode<Layer>* prevNodePtr = ptr->getPrev();
			Layer* prevPtr = prevNodePtr->getElemPtr();
			printf("linked\n");
			prevPtr->linkWeightsNext(elemPtr->getWeightsPrev()); // link ptr
		}
	};
	if (fromFile) {
		layers.forEach(initfromfilefunc);
	} else {
		layers.forEach(initfunc);
	}
	initialized = true;
}

void PhaseMagNetCUDA::free() {
	auto freefunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) {
			Layer* elemPtr = ptr->getElemPtr();
			elemPtr->freeWeightsPrev();
		}
	};
}

void PhaseMagNetCUDA::addLayer(const LayerParams& lp) {
	assert(!initialized);
	Layer lay(lp);
	layers.append(lay);
}

void PhaseMagNetCUDA::train(const size_t num_examples, uchar** inputData, uchar* labels, bool verbose) {
	assert(initialized);
	Matrix<DTYPE> inputdata(layers.getHead()->getElemPtr()->layParams.matDim);
	Matrix<DTYPE> ex(layers.getTail()->getElemPtr()->layParams.matDim);
	ex.fill(0.0);
	for (size_t i = 0; i < num_examples; ++i) {
		ex.setElem(0, labels[i], 1.0);
		layers.getHead()->getElemPtr()->layerData.fillFromUbyte(inputData[i]);
		resetState(); // doesn't overwrite input
		forwardPropagate();
		backwardPropagate(ex);
		ex.setElem(0, labels[i], 0.0); // prep for next example
		if (verbose) {
			printf("Training Progress: %5.2f\t\r", 100.0 * ((float)i) / ((float)num_examples));
		}
		Matrix<DTYPE>& output = getOutput();
		if (isnan(output.getElem(0, 0))) {
			throw std::runtime_error("isNaN\n");
		}
	}
}

Matrix<float> PhaseMagNetCUDA::predict(const size_t num_examples, uchar** inputData) {
	assert(initialized);
	MatrixDim mdim(num_examples, layers.getTail()->getElem().layParams.matDim.cdim, sizeof(float)); // hardcode float here
	Matrix<DTYPE> predictions(mdim);
	for (size_t i = 0; i < num_examples; ++i) {
		layers.getHead()->getElemPtr()->layerData.fillFromUbyte(inputData[i]);
		forwardPropagate();
		Matrix<DTYPE>& output = getOutput();
		for (size_t j = 0; j < predictions.mdim.cdim; ++j) {
			predictions.setElem(i, j, output.data[j]);
		}
	}
	return predictions;
}

float PhaseMagNetCUDA::evaluate(const size_t num_examples, uchar** inputData, uchar* labels) {
	assert(initialized);
	Matrix<float>& predictions = predict(num_examples, inputData);
	size_t num_correct = 0;
	for (size_t i = 0; i < num_examples; ++i) {
		float maxval = 0;
		int maxind = 0;
		for (size_t j = 0; j < predictions.mdim.cdim; ++j) {
			float val = predictions.getElem(i, j);
			if (val > maxval) {
				maxind = j;
				maxval = val;
			}
		}
		if (maxind == (int)labels[i])
			++num_correct;
	}
	return ((float) num_correct) / ((float) num_examples);
}

void PhaseMagNetCUDA::setInput(const Matrix<DTYPE>& input) {
	Layer* layPtr = layers.getHead()->getElemPtr();
	layPtr->layerData.fillFromMatrix(input);
}

Matrix<DTYPE> PhaseMagNetCUDA::getOutput() const {
	Layer* layPtr = layers.getTail()->getElemPtr();
	Matrix<DTYPE> out(layPtr->layerData);
	// iterate through output and calculate softmax
	DTYPE agg = 0.0;
	for (size_t i = 0; i < out.mdim.cdim; ++i) {
		DTYPE expval = exp(out.getElem(0, i));
		out.setElem(0, i, expval);
		agg += expval;
	}
	for (size_t i = 0; i < out.mdim.cdim; ++i) {
		out.setElem(0, i, out.getElem(0, i) / agg);
	}
	return out;
}

void PhaseMagNetCUDA::forwardPropagate(void) {
	// walk Layer list and update layer state incrementally
	// call matmul kernels in loop
	auto propagatefunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasNext()) { // only propagate if not last layer
			Layer* prevLayerPtr = ptr->getElemPtr();
			Layer* nextLayerPtr = (ptr->getNext())->getElemPtr();
			vecMatMultWithCuda(prevLayerPtr->layerData, *(prevLayerPtr->getWeightsNext()), nextLayerPtr->layerData); // writes results in nextLayerRef.layerdata
			nextLayerPtr->layerData.addMe(nextLayerPtr->bias); // add the bias to result
			auto relufunc = [](DTYPE* actptr) {
				*actptr = ((*actptr) < 0.0) ? 0.0 : *actptr;
			};
			nextLayerPtr->layerData.forEach(relufunc);
		}
	};
	layers.forEach(propagatefunc);
}

void PhaseMagNetCUDA::backwardPropagate(const Matrix<DTYPE>& expected) {
	// calculate error at the output
	auto subtract = [](DTYPE e, DTYPE o) {return e - o; };
	Matrix<DTYPE> err = Matrix<DTYPE>::pointwiseOp(expected, getOutput(), subtract);
	((layers.getTail()->getElemPtr())->errorData).fillFromMatrix(err);
	// walk Layer list in reverse and update layer error incrementally, adjust "next" weights
	// call matmul kernels on transpose weights
	// call weightUpdate kernels
	auto backpropagatefunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) {
			Layer* prevLayerPtr = (ptr->getPrev())->getElemPtr();
			Layer* nextLayerPtr = ptr->getElemPtr(); // iterating backwards, so this appears backwards wrt forwardpropagate
			auto reluderivfunc = [](DTYPE act, DTYPE err) {
				if (act > 0.0) {
					return err;
				}
				else {
					return (DTYPE) 0;
				}
			};
			Matrix<DTYPE> errToPropBack = Matrix<DTYPE>::pointwiseOp(nextLayerPtr->layerData, nextLayerPtr->errorData, reluderivfunc);
			// propagate error backwards
			vecMatMultWithCuda(errToPropBack, (prevLayerPtr->getWeightsNext())->transpose(), prevLayerPtr->errorData);
			// update weights
			updateWeightsWithCuda(prevLayerPtr->layerData, *(prevLayerPtr->getWeightsNext()), nextLayerPtr->errorData);
			MatrixDim biasDim(1, 1, sizeof(DTYPE));
			Matrix<DTYPE> biasMat(biasDim);
			biasMat.data[0] = 1.0;
			updateWeightsWithCuda(biasMat, nextLayerPtr->bias, nextLayerPtr->errorData);
		}
	};
	layers.forEachReverse(backpropagatefunc);
}

void PhaseMagNetCUDA::resetState(void) {
	// use CUDA to set everything to zero?
	auto resetfunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) { // input doesn't need to be reset; will be overwritten
			Layer* elemPtr = ptr->getElemPtr(); // reference to this element
			elemPtr->layerData.fill(0);
			elemPtr->errorData.fill(0);
		}
	};
	layers.forEach(resetfunc);
}

// thanks - https://stackoverflow.com/questions/22899595/saving-a-2d-array-to-a-file-c/22899753#22899753

template <typename T>
void writeMatrix(std::ostream& os, const Matrix<T>& mat)
{
	os << "matrix " << mat.mdim.rdim << " " << mat.mdim.cdim << " " << mat.mdim.size << " " << mat.mdim.stride << " \n";
	for (int i = 0; i < mat.mdim.cdim * mat.mdim.rdim; ++i)
	{
		os << mat.data[i] << " ";
	}
	os << "\n";
}

template <typename T>
Matrix<T> readMatrix(std::ifstream& is) {
	char output[10];
	is >> output;
	assert(std::strcmp(output, "matrix") == 0);
	size_t rdim, cdim, size, stride;
	is >> rdim >> cdim >> size >> stride;
	MatrixDim mdim;
	mdim.rdim = rdim;
	mdim.cdim = cdim;
	mdim.size = size;
	mdim.stride = stride;
	Matrix<T> temp(mdim);
	for (size_t i = 0; i < rdim * cdim; ++i) {
		is >> temp.data[i];
	}
	return temp;
}
//layparams
//LayerType layType;
//ActivationType actType;
//MatrixDim matDim;

//Matrix layerData
//Matrix errorData
//Matrix bias
//Matrix* weightsPrev
//Matrix* weightsNext

void writeLayer(std::ostream& os, const Layer* layptr) {
	MatrixDim mdim(layptr->layParams.matDim); // copy
	os << "layer " << static_cast<int>(layptr->layParams.layType) << " " << static_cast<int>(layptr->layParams.actType) << " "
		<< mdim.rdim << " " << mdim.cdim << " " << mdim.size << " " << mdim.stride << " \n";
	switch (layptr->layParams.layType) {
	case LayerType::input:
		break;
	default: // has biases and weightsPrev
		writeMatrix(os, layptr->bias);
		writeMatrix(os, *(layptr->weightsPrev));
		break;
	}
}

Layer readLayer(std::ifstream& istrm) {
	char output[10];
	istrm >> output;
	assert(std::strcmp(output, "layer") == 0); // to read a layer, gotta have "layer" first
	int layType, actType;
	istrm >> layType >> actType;
	size_t rdim, cdim, size, stride;
	istrm >> rdim >> cdim >> size >> stride;
	MatrixDim mdim;
	mdim.rdim = rdim;
	mdim.cdim = cdim;
	mdim.size = size;
	mdim.stride = stride;
	LayerParams lp(static_cast<LayerType>(layType), static_cast<ActivationType>(actType), mdim);
	Layer lay(lp);
	switch (lp.layType) {
	case LayerType::input:
		// weightsPrev will remain nullptr and bias will be 0; move on
		break;
	default:
		lay.bias.fillFromMatrix(readMatrix<DTYPE>(istrm));
		Matrix<DTYPE>& weightsPrevRef = readMatrix<DTYPE>(istrm);
		lay.weightsPrev = new Matrix<DTYPE>(weightsPrevRef); // allocated and copied
		break;
	}
	return lay;
	
}


void PhaseMagNetCUDA::save(const std::string& path) {
	assert(initialized);

	std::fstream of(path, std::ios::out); // do not want to append to existing file. path should be to a new file
	if (of.is_open()) {
		of << layers.getSize() << "\n";
		// walk layers and save each layer
		for (LinkedListNode<Layer>* ptr = layers.getHead(); ptr->hasNext(); ptr = ptr->getNext()) // will end at tail
			writeLayer(of, ptr->getElemPtr());
		writeLayer(of, (layers.getTail())->getElemPtr());
		of.close();
	}

}

void PhaseMagNetCUDA::load(const std::string& path) {
	assert(layers.isEmpty() && !initialized);
	std::ifstream ifile(path, std::ios::in); // do not want to append to existing file. path should be to a new file
	if (ifile.is_open()) {
		// walk layers and save each layer
		size_t num_layers;
		ifile >> num_layers;
		for (size_t i = 0; i < num_layers; ++i) {
			layers.append(readLayer(ifile));
		}
		initialize(true);
		ifile.close();
	}
}
