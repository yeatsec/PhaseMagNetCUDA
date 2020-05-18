
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include "LinkedList.cuh"
#include "pmncudautils.cuh"
#include "cudafuncs.cuh"
#include "PhaseMagNetCUDA.cuh"
#include "readdataset.cuh"

void buildNetwork(PhaseMagNetCUDA& net) {
    /* Input Layer */
    MatrixDim in_mdim(28, 28, sizeof(DTYPE), 1);
    LayerParams input(LayerType::input, ActivationType::relu, in_mdim);
    net.addLayer(input);
    /* Conv1 */
    ConvParams conv1;
    MatrixDim fDim(5, 5, sizeof(DTYPE), 1);
    conv1.filterDim = fDim;
    conv1.pad = 2;
    conv1.stride = 1;
    conv1.numFilters = 6;
    MatrixDim convMdim(conv1.getNextActDim(in_mdim, sizeof(DTYPE)));
    LayerParams convLayer(LayerType::conv, ActivationType::relu, convMdim, conv1);
    net.addLayer(convLayer);
    /* Average Pooling */
    ConvParams avgPoolParams;
    MatrixDim tmp(2, 2, sizeof(DTYPE), 6);
    avgPoolParams.filterDim = tmp;
    avgPoolParams.pad = 0;
    avgPoolParams.stride = 2;
    avgPoolParams.numFilters = 6;
    MatrixDim avgPoolMdim(avgPoolParams.getNextActDim(convMdim, sizeof(DTYPE)));
    LayerParams avgPoolLayer(LayerType::avgpool, ActivationType::relu, avgPoolMdim, avgPoolParams);
    net.addLayer(avgPoolLayer);
    /* FC 1 */
    MatrixDim mid1_mdim(1, 120, sizeof(DTYPE));
    LayerParams mid1(LayerType::fc, ActivationType::relu, mid1_mdim);
    net.addLayer(mid1);
    /* FC 2 */
    MatrixDim mid2_mdim(1, 84, sizeof(DTYPE));
    LayerParams mid2(LayerType::fc, ActivationType::relu, mid2_mdim);
    net.addLayer(mid2);
    /* FC 2 >>> OUTPUT <<< */
    MatrixDim out_mdim(1, 10, sizeof(DTYPE));
    LayerParams output(LayerType::fc, ActivationType::softmax, out_mdim);
     net.addLayer(output);
     net.initialize();
}

int main()
{
    int n_ims_train = 50000;
    int n_ims_test = 10000;
    int image_size = 784;
    PhaseMagNetCUDA net;
    //buildNetwork(net);
    net.load("convpmnn7.txt");

    printf("Loading Data...\n");
    uchar** imdata = read_mnist_images("..\\..\\..\\..\\mnist\\train-images-idx3-ubyte", n_ims_train, image_size);
    uchar* ladata = read_mnist_labels("..\\..\\..\\..\\mnist\\train-labels-idx1-ubyte", n_ims_train);
    uchar** imdata_test = read_mnist_images("..\\..\\..\\..\\mnist\\t10k-images-idx3-ubyte", n_ims_test, image_size); //t10k-images-idx3-ubyte // ann_a_advclp_0.2eps-ubyte
    uchar* ladata_test = read_mnist_labels("..\\..\\..\\..\\mnist\\t10k-labels-idx1-ubyte", n_ims_test);
    printf("Finished Loading Data.\n");

    float lrnRate = 0.0005f;
    for (int i = 1; i <= 1; ++i) {
        printf("Epoch: %d\n", i);
        float acc = net.evaluate(/*n_ims_test*/ 10000, imdata_test, ladata_test, /* verbose */ true);
        printf("Acc: %4.2f \n", acc * 100.0);
        net.train(/* n_ims_train */ 50000, imdata, ladata, /* */ lrnRate, /* verbose */ true);
        printf("\n");
    }
    net.save(".\\convpmnn7.txt");
    net.free();
    // printf("index: %d %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f true: %d\n", i,  o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], ladata[i]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}