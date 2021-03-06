#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>


#include "PhaseMagNetCUDA.cuh"
#include "cudafuncs.cuh"

#define SEED 15231

#define PRINT_WEIGHT
// #define PRINT_ACT
// #define PRINT_MAG
// #define PRINT_ERROR
// #define PRINT_OUTPUT


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
	srand(SEED);
	// iterate through the layers list and initialize weights
	auto initfunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) { // only do this for weight "owners"
			Layer* elemPtr = (ptr->getElemPtr()); // reference to this element
			LinkedListNode<Layer>* prevNodePtr = ptr->getPrev();
			Layer* prevPtr = prevNodePtr->getElemPtr();
			MatrixDim matDim;
			unsigned int numSets = 1;
			switch (elemPtr->layParams.layType) {
			case LayerType::maxpool: // use convparams stride and filterDim, pad must be zero
				elemPtr->initializeRNG(SEED); // initialize RNG in case of dropout, continue on
				setupRNGWithCuda(elemPtr->layerData.mdim, elemPtr->layerRNG, SEED); // any error will be printed inside function
			case LayerType::avgpool:
			case LayerType::conv:
			case LayerType::phasorconv: // ensure that the dimensions match up
			case LayerType::phasorconv2:
				// expect that the dimensions for both the conv filters and activation maps are set
				assert(elemPtr->layParams.convParams.getNextActDim(prevPtr->layParams.matDim, 
					sizeof(DTYPE)) == elemPtr->layParams.matDim);
				matDim = elemPtr->layParams.convParams.filterDim;
				numSets = elemPtr->layParams.convParams.numFilters;
				break;
			case LayerType::fc:
				matDim.adim = 1;
				matDim.rdim = prevPtr->layParams.matDim.getNumElems();
				matDim.cdim = elemPtr->layParams.matDim.getNumElems();
				matDim.size = matDim.rdim * matDim.cdim * sizeof(DTYPE);
				matDim.astride = matDim.rdim * matDim.cdim;
				matDim.rstride = matDim.cdim;
				break;
			default:
				throw std::logic_error("Layer Type not implemented\n");
			}
			elemPtr->initializeWeightsPrev(matDim, numSets); // weightsprev pointer set, device CudaMatrixArg pointers allocated
			printf("%s\n", getLayerTypeString(elemPtr->layParams.layType));
			prevPtr->linkWeightsNext(elemPtr); // link ptr
		}
	};
	auto initfromfilefunc = [](LinkedListNode<Layer>* ptr) { // weights already allocated
		if (ptr->hasPrev()) {
			Layer* elemPtr = (ptr->getElemPtr()); // reference to this element
			if (elemPtr->layParams.layType == LayerType::maxpool) { // prep for possible dropout
				elemPtr->initializeRNG(SEED);
				setupRNGWithCuda(elemPtr->layerData.mdim, elemPtr->layerRNG, SEED);
			}
			LinkedListNode<Layer>* prevNodePtr = ptr->getPrev();
			Layer* prevPtr = prevNodePtr->getElemPtr();
			printf("%s\n", getLayerTypeString(elemPtr->layParams.layType));
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

void PhaseMagNetCUDA::train(const unsigned int num_examples, uchar** inputData, uchar* labels, float lrnRate, bool verbose, float dropout, float linfNoise, float gaussNoise) {
	assert(initialized);
	Matrix<DTYPE> ex(layers.getTail()->getElemPtr()->layParams.matDim);
	ex.fill(0.0);
	for (unsigned int i = 0; i < num_examples; ++i) {
		ex.setElem(0, labels[i], 1.0);
		setInput(inputData[i], linfNoise, gaussNoise);
		resetState(); // doesn't overwrite input
		forwardPropagate(dropout);
		backwardPropagate(ex, lrnRate);
		ex.setElem(0, labels[i], 0.0); // prep for next example
		if (verbose) {
			printf("Training Progress: %5.2f\t\r", 100.0 * ((float)i+1) / ((float)num_examples));
		}
		Matrix<DTYPE> output(getOutput());
		if (isnan(output.getElem(0, 0))) {
			printf("\nisNaN output\n");
			throw std::runtime_error("output isNaN\n");
		}
		if (i % 1000 == 0) {
			// sample activation
			Matrix<DTYPE> sample(layers.getTail()->getPrev()->getElemPtr()->layerData);
			for (int j = 0; j < sample.getNumElems(); ++j) {
				assert(!isnan(sample.getElemFlatten(j)));
			}
		}
	}
	printf("\n");
}


void PhaseMagNetCUDA::genAdv(const std::string name, const unsigned int num_examples, const unsigned int img_rows, const unsigned int img_cols, uchar** inputData, uchar* labels, float epsilon, int numIts, bool targeted, bool randomStart, bool verbose) {
	assert(initialized);
	assert(numIts > 0);
	std::cout << "Generating Adversarial Examples for File " << name << std::endl;
	printf("Allocating Adversarial Output Size: %d Images %d ImageSize\n", num_examples, img_rows * img_cols);
	const unsigned int img_size = img_rows * img_cols;
	uchar** outputData = new uchar * [num_examples];
	for (unsigned int i = 0; i < num_examples; ++i) {
		outputData[i] = new uchar[img_size];
	}
	printf("Output Allocated\n");
	Matrix<DTYPE> ex(layers.getTail()->getElemPtr()->layParams.matDim);
	ex.fill(0.0);
	float stepEpsilon = epsilon / ((float)numIts);
	if (randomStart) {
		stepEpsilon /= 2.0f; // use alpha = 0.5eps
	}
	for (unsigned int i = 0; i < num_examples; ++i) {
		if (randomStart) {
			Matrix<DTYPE> h_inpLayAct(layers.getHead()->getElemPtr()->layParams.matDim);
			h_inpLayAct.fillFromUbyte(inputData[i]); // bypass setInput
			Matrix<DTYPE> noise(h_inpLayAct.mdim);
			// add random uniform noise at half epsilon
			noise.fillRandom((-epsilon) / 2.0f, epsilon / 2.0f);
			h_inpLayAct.addMe(noise);
			layers.getHead()->getElemPtr()->layerData.fillFromMatrix(h_inpLayAct);
		}
		else {
			setInput(inputData[i]);
		}
		ex.setElem(0, labels[i], 1.0);
		
		for (int stepInd = 0; stepInd < numIts; ++stepInd) {
			resetState(); // doesn't overwrite input activation, does overwrite input error to zero
			forwardPropagate();
			backwardPropagate(ex, 0.0, true, stepEpsilon, targeted);
		}
		Matrix<DTYPE> advExample(layers.getHead()->getElemPtr()->layerData);
		advExample.dumpToUbyte(outputData[i]);
		ex.setElem(0, labels[i], 0.0);
		if (verbose) {
			printf("\rGenerating Adversarial Examples %3.2f Percent Complete \t", 100.0 * ((float)(i+1)) / ((float)num_examples));
		}
	}
	printf("\n Saving Adversarial Examples\n");
	write_mnist_images(name, num_examples, img_rows, img_cols, outputData);
	std::cout << "Images Saved to " << name << std::endl;
	printf("Deallocating Adversarial Output\n");
	for (unsigned int i = 0; i < num_examples; ++i) {
		delete[] outputData[i];
	}
	delete[] outputData;
	printf("Adversarial Output deallocated\n");
}

Matrix<float> PhaseMagNetCUDA::predict(const unsigned int num_examples, uchar** inputData, bool verbose, float linfNoise) {
	assert(initialized);
	MatrixDim mdim(num_examples, layers.getTail()->getElem().layParams.matDim.cdim, sizeof(float)); // hardcode float here
	Matrix<DTYPE> predictions(mdim);
	for (unsigned int i = 0; i < num_examples; ++i) {
		setInput(inputData[i], linfNoise);
		forwardPropagate();
		Matrix<DTYPE> output(getOutput());
		for (unsigned int j = 0; j < predictions.mdim.cdim; ++j) {
			predictions.setElem(i, j, output.data[j]);
		}
#ifdef PRINT_OUTPUT
		printf("Output:\t");
		for (unsigned int j = 0; j < predictions.mdim.cdim; ++j) {
			printf("%3.3f\t", output.data[j]);
		}
		printf("\n");
#endif // PRINT_OUTPUT
		if (verbose) {
			printf("Prediction Progress: %5.2f\t\r", 100.0 * ((float)(i+1)) / ((float)num_examples));
		}
	}
	if (verbose)
		printf("\n");
	return predictions;
}

float PhaseMagNetCUDA::evaluate(const unsigned int num_examples, uchar** inputData, uchar* labels, bool verbose, float linfNoise) {
	assert(initialized);
	Matrix<float> predictions(predict(num_examples, inputData, verbose, linfNoise));
	unsigned int num_correct = 0;
	for (unsigned int i = 0; i < num_examples; ++i) {
		float maxval = 0;
		int maxind = 0;
		for (unsigned int j = 0; j < predictions.mdim.cdim; ++j) {
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

void PhaseMagNetCUDA::setInput(const uchar* const ucharptr, float linfNoise, float gaussNoise) {
	Layer* layPtr = layers.getHead()->getElemPtr();
	Matrix<DTYPE> temp(layPtr->layerData.mdim);
	temp.fillFromUbyte(ucharptr);
	if (linfNoise > 0.0f) {
		temp.addLinfNoise(linfNoise);
	}
	else if (gaussNoise > 0.0f) {
		temp.addGaussNoise(gaussNoise);
	}
	layPtr->layerData.fillFromMatrix(temp);
}

Matrix<DTYPE> PhaseMagNetCUDA::getOutput() const {
	Layer* layPtr = layers.getTail()->getElemPtr();
	Matrix<DTYPE> out(layPtr->layerData);
	
#ifdef PRINT_MAG
	printf("Mag Output: ");
	for (unsigned int i = 0; i < out.mdim.cdim; ++i) {
		printf("%5.5f \t", out.getElem(0, i));
	}
	printf("\n");
#endif // PRINT_MAG
	// iterate through output and calculate softmax
	// first get a good scaling constant
	DTYPE shift = out.getElemFlatten(0);
	// find maximum output
	for (unsigned int i = 0; i < out.mdim.getNumElems(); ++i) {
		DTYPE elem = out.getElemFlatten(i);
		if (elem > shift)
			shift = elem;
	}
	DTYPE agg = 0.0;
	for (unsigned int i = 0; i < out.mdim.getNumElems(); ++i) {
		DTYPE expval = exp(out.getElemFlatten(i) - shift);
		out.setElemFlatten(i, expval);
		agg += expval;
	}
	for (unsigned int i = 0; i < out.mdim.getNumElems(); ++i) {
		out.setElemFlatten(i, out.getElemFlatten(i) / agg);
	}
	return out;
}

void PhaseMagNetCUDA::forwardPropagate(float dropout) {
	// walk Layer list and update layer state incrementally
	// call matmul kernels in loop
	auto propagatefunc = [dropout](LinkedListNode<Layer>* ptr) {
		if (ptr->hasNext()) { // only propagate if not last layer
			Layer* prevLayerPtr = ptr->getElemPtr();
			Layer* nextLayerPtr = (ptr->getNext())->getElemPtr();
			
			cudaError_t cudaStatus(cudaSuccess);
			
			switch (nextLayerPtr->layParams.layType) {
			case LayerType::phasorconv:
			case LayerType::phasorconv2:
				//printf("CONV\n");
				cudaStatus = complexConvolutionWithCuda(prevLayerPtr->layerData,
					nextLayerPtr->weightsPrevR, nextLayerPtr->weightsPrevI, nextLayerPtr->bias,
					nextLayerPtr->layParams.convParams, nextLayerPtr->layParams.layType, nextLayerPtr->layParams.actType,
					nextLayerPtr->layerData, nextLayerPtr->layerDataMag, nextLayerPtr->layerDataAng);
				break;
			case LayerType::conv:
				cudaStatus = scalarConvolutionWithCuda(prevLayerPtr->layerData, nextLayerPtr->getWeightsPrevR(), nextLayerPtr->bias,
					nextLayerPtr->layParams.convParams, nextLayerPtr->layerData, nextLayerPtr->layParams.actType);
				break;
			case LayerType::avgpool:
				//printf("AVG_POOL\n");
				cudaStatus = scalarAvgPoolWithCuda(prevLayerPtr->layerData,
					nextLayerPtr->layParams.convParams, nextLayerPtr->layerData, nextLayerPtr->layParams.actType);
				break;
			case LayerType::maxpool:
				cudaStatus = scalarMaxPoolWithCuda(prevLayerPtr->layerData, nextLayerPtr->layParams.convParams, nextLayerPtr->layerData,
					nextLayerPtr->layParams.actType, nextLayerPtr->layerRNG, dropout);
				break;
			case LayerType::fc:
				cudaStatus = scalarFCForwardPropWithCuda(prevLayerPtr->layerData,
					*(nextLayerPtr->getWeightsPrevR()), nextLayerPtr->bias,
					nextLayerPtr->layerData, nextLayerPtr->layParams.actType);
				break;
			default:
				throw std::logic_error("Not yet implemented\n");
				break;
			}

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "forward propagate with cuda failed! %d %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
				throw std::logic_error("Forward Propagate Failed\n");
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
			Matrix<DTYPE> nextAct(nextLayerPtr->layerData);
			printf("Activation\n");
			for (int i = 0; i < nextAct.getNumElems(); ++i) {
				printf("%5.5f ", nextAct.data[i]);
			}
			printf("\n");
#endif // PRINT_ACT
		}
	};
	// would do a forEach here but dropout needs to be in the sensitivity list of the lambda
	for (LinkedListNode<Layer>* ptr = layers.getHead(); ptr->hasNext(); ptr = ptr->getNext()) // will end at tail
		propagatefunc(ptr);
	propagatefunc(layers.getTail()); // don't forget the tail
}

void PhaseMagNetCUDA::backwardPropagate(const Matrix<DTYPE>& expected, float lrnRate, bool adjInput, float eps, bool targeted) {
	Matrix<DTYPE> err; // forward declaration
	// calculate error at the outputs
	if (adjInput) {
		auto subtract = [](DTYPE e, DTYPE o) { return (e == ((DTYPE)1.0f) ? e - o : ((DTYPE)0.0f));};
		err = Matrix<DTYPE>::pointwiseOp(expected, getOutput(), subtract); // assignment
	}
	else {
		auto subtract = [](DTYPE e, DTYPE o) {return e - o; };
		err = Matrix<DTYPE>::pointwiseOp(expected, getOutput(), subtract);
	}
	((layers.getTail()->getElemPtr())->errorData).fillFromMatrix(err);
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
			Matrix<DTYPE> errMag(nextLayerPtr->errorData);
			for (int i = 0; i < errMag.mdim.getNumElems(); ++i)
				printf("%5.5f ", errMag.data[i]);
			printf("\n");
#endif // PRINT_ERROR
			// update the bias, backpropagate error, update weights
			
			switch (nextLayerPtr->layParams.layType) {
			case LayerType::phasorconv:
			case LayerType::phasorconv2:
				cudaStatus = complexConvBackpropWithCuda(prevLayerPtr->layerData, prevLayerPtr->errorData, 
					nextLayerPtr->weightsPrevR, nextLayerPtr->weightsPrevI, nextLayerPtr->bias, nextLayerPtr->layParams.convParams, 
					nextLayerPtr->layParams.layType, nextLayerPtr->layParams.actType,
					nextLayerPtr->layerData, nextLayerPtr->layerDataMag, nextLayerPtr->layerDataAng, nextLayerPtr->errorData, lrnRate);
				break;
			case LayerType::conv:
				cudaStatus = scalarConvolutionBackpropWithCuda(prevLayerPtr->layerData, prevLayerPtr->errorData,
					nextLayerPtr->weightsPrevR, nextLayerPtr->bias, nextLayerPtr->layParams.convParams,
					nextLayerPtr->layerData, nextLayerPtr->errorData, lrnRate, nextLayerPtr->layParams.actType);
				break;
			case LayerType::avgpool:
				cudaStatus = scalarAvgPoolBackpropWithCuda(prevLayerPtr->errorData, nextLayerPtr->layParams.convParams, nextLayerPtr->layerData,
					nextLayerPtr->errorData, nextLayerPtr->layParams.actType);
				break;
			case LayerType::maxpool:
				cudaStatus = scalarMaxPoolBackpropWithCuda(prevLayerPtr->layerData, prevLayerPtr->errorData, nextLayerPtr->layParams.convParams,
					nextLayerPtr->layerData, nextLayerPtr->errorData, nextLayerPtr->layParams.actType);
				break;
			case LayerType::fc:
				cudaStatus = scalarFCBackpropWithCuda(prevLayerPtr->layerData, prevLayerPtr->errorData,
					*(nextLayerPtr->weightsPrevR), nextLayerPtr->layerData, nextLayerPtr->errorData,
					nextLayerPtr->bias, nextLayerPtr->layParams.actType, lrnRate);
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
	// cute trick: add error to input activation
	if (adjInput) {
		Layer* inpLayer = layers.getHead()->getElemPtr();
		addSubAndClipWithCuda(inpLayer->layerData, inpLayer->errorData, inpLayer->layerData, eps, targeted);
	}
}

void PhaseMagNetCUDA::resetState(void) {
	// use CUDA to set everything to zero?
	auto resetfunc = [](LinkedListNode<Layer>* ptr) {
		if (ptr->hasPrev()) { // input doesn't need to be reset; will be overwritten
			Layer* elemPtr = ptr->getElemPtr(); // reference to this element
			setValueWithCuda(elemPtr->layerData, 0);
			setValueWithCuda(elemPtr->layerDataAng, 0);
			setValueWithCuda(elemPtr->errorData, 0);
		}
	};
	layers.forEach(resetfunc);
	setValueWithCuda(layers.getHead()->getElemPtr()->errorData, 0);
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
	for (unsigned int i = 0; i < temp.mdim.getNumElems(); ++i) {
		is >> temp.data[i];
	}
	return temp;
}

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
		Matrix<DTYPE> tempBias(layptr->bias);
		writeMatrix(os, tempBias);
		for (unsigned int i = 0; i < layptr->layParams.convParams.numFilters; ++i) {
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
		lay.bias.fillFromMatrix(readMatrix<DTYPE>(istrm));
		lay.weightsPrevR = new CudaMatrix<DTYPE>[cP.numFilters];
		lay.weightsPrevI = new CudaMatrix<DTYPE>[cP.numFilters];
		for (unsigned int i = 0; i < cP.numFilters; ++i) {
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
		unsigned int num_layers;
		ifile >> num_layers;
		for (unsigned int i = 0; i < num_layers; ++i) {
			layers.append(readLayer(ifile));
		}
		initialize(true);
		ifile.close();
	}
}
