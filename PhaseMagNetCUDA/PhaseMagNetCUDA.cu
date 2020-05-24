#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>

#include "PhaseMagNetCUDA.cuh"
#include "cudafuncs.cuh"

// #define PRINT_WEIGHT
// #define PRINT_ACT
// #define PRINT_MAG
// #define PRINT_ERROR


/*
	Convention: Activation A<1, N>, Weights <N, M>, Activation <1, M>
*/

DTYPE abs2(DTYPE r, DTYPE i) {
	return sqrt((r * r) + (i * i));
}

PhaseMagNetCUDA::PhaseMagNetCUDA() :
	layers(),
	initialized(false) {}

void PhaseMagNetCUDA::initialize(bool fromFile) {
	assert(!(initialized || layers.isEmpty()));
	// do this to seed the random
	// srand(static_cast <unsigned> (time(0)));
	srand(3);
	// iterate through the layers list and initialize weights
	auto initfunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) { // only do this for weight "owners"
			Layer* elemPtr = (ptr->getElemPtr()); // reference to this element
			LinkedListNode<Layer>* prevNodePtr = ptr->getPrev();
			Layer* prevPtr = prevNodePtr->getElemPtr();
			MatrixDim matDim;
			size_t numSets = 1;
			switch (elemPtr->layParams.layType) {
			case LayerType::maxpool: // use convparams stride and filterDim, pad must be zero
			case LayerType::avgpool:
			case LayerType::conv: // ensure that the dimensions match up
				// expect that the dimensions for both the conv filters and activation maps are set
				assert(elemPtr->layParams.convParams.getNextActDim(prevPtr->layParams.matDim, 
					sizeof(DTYPE)) == elemPtr->layParams.matDim);
				matDim = elemPtr->layParams.convParams.filterDim;
				numSets = elemPtr->layParams.convParams.numFilters;
				break;
			default:
				matDim.adim = 1;
				matDim.rdim = prevPtr->layParams.matDim.getNumElems();
				matDim.cdim = elemPtr->layParams.matDim.getNumElems();
				matDim.size = matDim.rdim * matDim.cdim * sizeof(DTYPE);
				matDim.astride = matDim.rdim * matDim.cdim;
				matDim.rstride = matDim.cdim;
				break;
			}
			// currently hardcoded for fully connected
			elemPtr->initializeWeightsPrev(matDim, numSets); // weightsprev pointer set
			printf("linked\n");
			prevPtr->linkWeightsNext(elemPtr); // link ptr
		}
	};
	auto initfromfilefunc = [](LinkedListNode<Layer>* ptr) { // weights already allocated
		if (ptr->hasPrev()) {
			Layer* elemPtr = (ptr->getElemPtr()); // reference to this element
			LinkedListNode<Layer>* prevNodePtr = ptr->getPrev();
			Layer* prevPtr = prevNodePtr->getElemPtr();
			printf("linked\n");
			prevPtr->linkWeightsNext(elemPtr); // link ptr
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
		if (ptr->hasNext()) {
			Layer* elemPtr = ptr->getElemPtr();
			elemPtr->weightsNextR = nullptr;
			elemPtr->weightsNextI = nullptr;
		}
	};
	layers.forEach(freefunc);
}

void PhaseMagNetCUDA::addLayer(const LayerParams& lp) {
	assert(!initialized);
	Layer lay(lp);
	layers.append(lay);
}

void PhaseMagNetCUDA::train(const size_t num_examples, uchar** inputData, uchar* labels, float lrnRate, bool verbose) {
	assert(initialized);
	Matrix<DTYPE> ex(layers.getTail()->getElemPtr()->layParams.matDim);
	ex.fill(0.0);
	for (size_t i = 0; i < num_examples; ++i) {
		ex.setElem(0, labels[i], 1.0);
		setInput(inputData[i]);
		resetState(); // doesn't overwrite input
		forwardPropagate();
		backwardPropagate(ex, lrnRate);
		//Matrix<DTYPE>& prmat = layers.getHead()->getNext()->getElem().errorData;
		//printf("DEBUG %1.8f\n", prmat.getElemFlatten(200/*prmat.mdim.getNumElems()/2*/));
		ex.setElem(0, labels[i], 0.0); // prep for next example
		if (verbose) {
			printf("Training Progress: %5.2f\t\r", 100.0 * ((float)i+1) / ((float)num_examples));
		}
		Matrix<DTYPE>& output = getOutput();
		if (isnan(output.getElem(0, 0))) {
			printf("\nisNaN output\n");
			throw std::runtime_error("output isNaN\n");
		}
	}
	printf("\n");
}

Matrix<float> PhaseMagNetCUDA::predict(const size_t num_examples, uchar** inputData, bool verbose) {
	assert(initialized);
	MatrixDim mdim(num_examples, layers.getTail()->getElem().layParams.matDim.cdim, sizeof(float)); // hardcode float here
	Matrix<DTYPE> predictions(mdim);
	for (size_t i = 0; i < num_examples; ++i) {
		setInput(inputData[i]);
		forwardPropagate();
		Matrix<DTYPE>& output = getOutput();
		//printf("DEBUG %5.3f\n", layers.getHead()->getNext()->getElem().layerDataR.getElem(14, 14));
		//printf("Output:\t");
		for (size_t j = 0; j < predictions.mdim.cdim; ++j) {
			predictions.setElem(i, j, output.data[j]);
			//printf("%3.3f\t", output.data[j]);
		}
		//printf("\n");
		if (verbose) {
			printf("Prediction Progress: %5.2f\t\r", 100.0 * ((float)(i+1)) / ((float)num_examples));
		}
	}
	if (verbose)
		printf("\n");
	return predictions;
}

float PhaseMagNetCUDA::evaluate(const size_t num_examples, uchar** inputData, uchar* labels, bool verbose) {
	assert(initialized);
	Matrix<float>& predictions = predict(num_examples, inputData, verbose);
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

void PhaseMagNetCUDA::setInput(const uchar* const ucharptr) {
	Layer* layPtr = layers.getHead()->getElemPtr();
	Matrix<DTYPE> tempR(layPtr->layerDataR.mdim);
	Matrix<DTYPE> tempI(layPtr->layerDataI.mdim);
	for (size_t i = 0; i < layPtr->layParams.matDim.getNumElems(); ++i) {
		DTYPE rad = PI * ((DTYPE)(ucharptr[i]) / 255.0f) - PI / 2;
		DTYPE real = cosf(rad);
		DTYPE imag = sinf(rad);
		tempR.setElemFlatten(i, real);
		tempI.setElemFlatten(i, imag);
	}
	layPtr->layerDataR.fillFromMatrix(tempR);
	layPtr->layerDataI.fillFromMatrix(tempI);
}

Matrix<DTYPE> PhaseMagNetCUDA::getOutput() const {
	Layer* layPtr = layers.getTail()->getElemPtr();
	Matrix<DTYPE> outR(layPtr->layerDataR);
	Matrix<DTYPE> outI(layPtr->layerDataI);
	Matrix<DTYPE> out(layPtr->layParams.matDim);
	
	for (unsigned int i = 0; i < out.mdim.cdim; ++i) {
		out.setElem(0, i, abs2(outR.getElem(0, i), outI.getElem(0, i))); // magnitude
	}
#ifdef PRINT_MAG
	printf("Mag Output: ");
	for (unsigned int i = 0; i < out.mdim.cdim; ++i) {
		printf("%5.5f \t", out.getElem(0, i));
	}
	printf("\n");
#endif // PRINT_MAG
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
			cudaError_t cudaStatus(cudaSuccess);
			switch (nextLayerPtr->layParams.layType) {
			case LayerType::conv:
				//printf("CONV\n");
				cudaStatus = complexConvolutionWithCuda(prevLayerPtr->layerDataR, prevLayerPtr->layerDataI,
					nextLayerPtr->weightsPrevR, nextLayerPtr->weightsPrevI, nextLayerPtr->layParams.convParams,
					nextLayerPtr->layerDataR, nextLayerPtr->layerDataI);
				break;
			case LayerType::avgpool:
				//printf("AVG_POOL\n");
				cudaStatus = complexAveragePoolWithCuda(prevLayerPtr->layerDataR, prevLayerPtr->layerDataI,
					nextLayerPtr->layParams.convParams, nextLayerPtr->layerDataR, nextLayerPtr->layerDataI);
				break;
			case LayerType::fc:
				//printf("FC\n");
				cudaStatus = vecMatMultWithCuda(prevLayerPtr->layerDataR, prevLayerPtr->layerDataI,
					*(prevLayerPtr->getWeightsNextR()), *(prevLayerPtr->getWeightsNextI()),
					nextLayerPtr->layerDataR, nextLayerPtr->layerDataI); // writes results in nextLayerRef.layerdata
				complexAddBiasWithCuda(nextLayerPtr->layerDataR, nextLayerPtr->layerDataI,
					nextLayerPtr->biasR, nextLayerPtr->biasI);
				break;
			default:
				throw std::logic_error("Not yet implemented\n");
				break;
			}
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "forward propagate with cuda failed! %d %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
			}
#ifdef PRINT_WEIGHT
			Matrix<DTYPE> wgtR(nextLayerPtr->weightsPrevR[0]);
			Matrix<DTYPE> wgtI(nextLayerPtr->weightsPrevI[0]);
			for (int i = 0; i < wgtR.getNumElems(); ++i) {
				if (isnan(wgtR.data[i]) || isnan(wgtI.data[i])) {
					printf("It: %d \t Re(%5.5f) Im(%5.5f) \n", i, wgtR.data[i], wgtI.data[i]);
				}
			}
#endif // PRINT_WEIGHT
#ifdef PRINT_ACT
			printf("Activation\n");
			for (int i = 0; i < nextR.getNumElems(); ++i) {
				printf("Re(%5.5f) Im(%5.5f) ", nextR.data[i], nextI.data[i]);
			}
			printf("\n");
#endif // PRINT_ACT
		}
	};
	// print the input
	//Matrix<DTYPE> inpR(layers.getHead()->getElemPtr()->layerDataR);
	//Matrix<DTYPE> inpI(layers.getHead()->getElemPtr()->layerDataI);
	/*printf("Input\n");
	for (int i = 0; i < inpR.mdim.getNumElems(); ++i) {
		printf("Re(%5.5f) Im(%5.5f) ", inpR.data[i], inpI.data[i]);
	}
	printf("\n");*/
	layers.forEach(propagatefunc);
}

void PhaseMagNetCUDA::backwardPropagate(const Matrix<DTYPE>& expected, float lrnRate) {
	// calculate error at the outputs
	auto subtract = [](DTYPE e, DTYPE o) {return e - o; };
	Matrix<DTYPE> err = Matrix<DTYPE>::pointwiseOp(expected, getOutput(), subtract);
	((layers.getTail()->getElemPtr())->errorDataMag).fillFromMatrix(err);
	setValueWithCuda(((layers.getTail()->getElemPtr())->errorDataAng), 0.0f);
	// walk Layer list in reverse and update layer error incrementally, adjust "next" weights
	// call matmul kernels on transpose weights
	// call weightUpdate kernels
	auto backpropagatefunc = [lrnRate](LinkedListNode<Layer>* ptr) {
		cudaError_t cudaStatus;
		if (ptr->hasPrev()) {
			Layer* prevLayerPtr = (ptr->getPrev())->getElemPtr();
			Layer* nextLayerPtr = ptr->getElemPtr(); // iterating backwards, so this appears backwards wrt forwardpropagate
			/* Print the error at each neuron in next*/
#ifdef PRINT_ERROR
			printf("Error: \n");
			Matrix<DTYPE> errMag(nextLayerPtr->errorDataMag);
			Matrix<DTYPE> errAng(nextLayerPtr->errorDataAng);
			for (int i = 0; i < errMag.mdim.getNumElems(); ++i)
				printf("Mag(%5.5f) Ang(%5.5f) ", errMag.data[i], errAng.data[i]);
			printf("\n");
#endif // PRINT_ERROR
			// update the bias, backpropagate error, update weights
			switch (nextLayerPtr->layParams.layType) {
			case LayerType::conv:
				cudaStatus = complexConvBackpropWithCuda(prevLayerPtr->layerDataR, prevLayerPtr->layerDataI, prevLayerPtr->errorDataMag, 
					prevLayerPtr->errorDataAng, nextLayerPtr->weightsPrevR, nextLayerPtr->weightsPrevI, nextLayerPtr->layParams.convParams, 
					nextLayerPtr->layerDataR, nextLayerPtr->layerDataI, nextLayerPtr->errorDataMag, nextLayerPtr->errorDataAng, lrnRate);
				break;
			case LayerType::avgpool:
				cudaStatus = complexAvgPoolBackpropWithCuda(prevLayerPtr->layerDataR, prevLayerPtr->layerDataI, prevLayerPtr->errorDataMag,
					prevLayerPtr->errorDataAng, nextLayerPtr->layParams.convParams, nextLayerPtr->layerDataR, nextLayerPtr->layerDataI,
					nextLayerPtr->errorDataMag, nextLayerPtr->errorDataAng);
				break;
			case LayerType::fc:
				cudaStatus = complexBackpropWithCuda(prevLayerPtr->layerDataR, prevLayerPtr->layerDataI, prevLayerPtr->errorDataMag,
					prevLayerPtr->errorDataAng, *(nextLayerPtr->weightsPrevR), *(nextLayerPtr->weightsPrevI), nextLayerPtr->biasR, nextLayerPtr->biasI,
					nextLayerPtr->layerDataR, nextLayerPtr->layerDataI, nextLayerPtr->errorDataMag, nextLayerPtr->errorDataAng, lrnRate);
				break;
			default:
				throw std::logic_error("Not yet implemented\n");
				break;
			}
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "backward propagate with cuda failed! %d %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
			}
		}
	};
	for (LinkedListNode<Layer>* ptr = layers.getTail(); ptr->hasPrev(); ptr = ptr->getPrev())
		backpropagatefunc(ptr);
}

void PhaseMagNetCUDA::resetState(void) {
	// use CUDA to set everything to zero?
	auto resetfunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) { // input doesn't need to be reset; will be overwritten
			Layer* elemPtr = ptr->getElemPtr(); // reference to this element
			setValueWithCuda(elemPtr->layerDataR, 0);
			setValueWithCuda(elemPtr->layerDataI, 0);
			setValueWithCuda(elemPtr->errorDataMag, 0);
			setValueWithCuda(elemPtr->errorDataAng, 0);
		}
	};
	layers.forEach(resetfunc);
}

// thanks - https://stackoverflow.com/questions/22899595/saving-a-2d-array-to-a-file-c/22899753#22899753

void writeMatrixDim(std::ostream& os, const MatrixDim& mdim) {
	os << mdim.adim << " " << mdim.rdim << " " << mdim.cdim << " " << mdim.size <<
		" " << mdim.astride << " " << mdim.rstride << " ";
}

void writeConvParams(std::ostream& os, const ConvParams& cP) {
	writeMatrixDim(os, cP.filterDim);
	os << cP.numFilters << " " << cP.pad << " " << cP.stride << " ";
}

template <typename T>
void writeMatrix(std::ostream& os, const Matrix<T>& mat)
{
	os << "matrix ";
	writeMatrixDim(os, mat.mdim);
	os << " \n";
	for (int i = 0; i < mat.mdim.getNumElems(); ++i)
	{
		os << mat.data[i] << " ";
	}
	os << "\n";
}

MatrixDim readMatrixDim(std::ifstream& istrm) {
	MatrixDim mdim;
	istrm >> mdim.adim >> mdim.rdim >> mdim.cdim >> mdim.size >> mdim.astride >> mdim.rstride;
	return mdim;
}

ConvParams readConvParams(std::ifstream& istrm) {
	ConvParams cP;
	cP.filterDim = readMatrixDim(istrm);
	istrm >> cP.numFilters >> cP.pad >> cP.stride;
	return cP;
}

template <typename T>
Matrix<T> readMatrix(std::ifstream& is) {
	char output[10];
	is >> output;
	assert(std::strcmp(output, "matrix") == 0);
	Matrix<T> temp(readMatrixDim(is));
	for (size_t i = 0; i < temp.mdim.getNumElems(); ++i) {
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
	os << "layer " << static_cast<int>(layptr->layParams.layType) << " " << static_cast<int>(layptr->layParams.actType) << " ";
	writeMatrixDim(os, mdim);
	os << "ConvParams ";
	writeConvParams(os, layptr->layParams.convParams);
	os << " \n";
	switch (layptr->layParams.layType) {
	case LayerType::input:
		break;
	default: // has biases and weightsPrev
		Matrix<DTYPE> tempBiasR(layptr->biasR);
		writeMatrix(os, tempBiasR);
		Matrix<DTYPE> tempBiasI(layptr->biasI);
		writeMatrix(os, tempBiasI);
		for (int i = 0; i < layptr->layParams.convParams.numFilters; ++i) {
			Matrix<DTYPE> tempR((layptr->weightsPrevR)[i]);
			Matrix<DTYPE> tempI((layptr->weightsPrevI)[i]);
			writeMatrix(os, tempR);
			writeMatrix(os, tempI);
		}
		break;
	}
}

Layer readLayer(std::ifstream& istrm) {
	char output[15];
	istrm >> output;
	assert(std::strcmp(output, "layer") == 0); // to read a layer, gotta have "layer" first
	int layType, actType;
	istrm >> layType >> actType;
	MatrixDim mdim = readMatrixDim(istrm);
	istrm >> output;
	assert(std::strcmp(output, "ConvParams") == 0);
	ConvParams cP = readConvParams(istrm);
	LayerParams lp(static_cast<LayerType>(layType), static_cast<ActivationType>(actType), mdim, cP);
	Layer lay(lp);
	switch (lp.layType) {
	case LayerType::input:
		// weightsPrev will remain nullptr and bias will be 0; move on
		break;
	default:
		lay.biasR.fillFromMatrix(readMatrix<DTYPE>(istrm));
		lay.biasI.fillFromMatrix(readMatrix<DTYPE>(istrm));
		lay.weightsPrevR = new CudaMatrix<DTYPE>[cP.numFilters];
		lay.weightsPrevI = new CudaMatrix<DTYPE>[cP.numFilters];
		for (int i = 0; i < cP.numFilters; ++i) {
			lay.weightsPrevR[i] = CudaMatrix<DTYPE>(readMatrix<DTYPE>(istrm));
			lay.weightsPrevI[i] = CudaMatrix<DTYPE>(readMatrix<DTYPE>(istrm)); // allocated and copied
		}
		break;
	}
	return lay;
	
}


void PhaseMagNetCUDA::save(const std::string& path) {
	assert(initialized);

	std::fstream of(path, std::ios::out); // do not want to append to existing file. path should be to a new file
	if (of.is_open()) {
		of << layers.getSize() << " \n";
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
