
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

int main()
{
    int num_images = 50000;
    int num_images_test = 10000;
    int n_ims_train = 50000;
    int n_ims_test = 10000;
    int image_size = 784;
    printf("Loading Data...\n");
    uchar** imdata = read_mnist_images("..\\..\\..\\..\\mnist\\train-images-idx3-ubyte", n_ims_train, image_size);
    uchar* ladata = read_mnist_labels("..\\..\\..\\..\\mnist\\train-labels-idx1-ubyte", n_ims_train);
    uchar** imdata_test = read_mnist_images("..\\..\\..\\..\\mnist\\t10k-images-idx3-ubyte", n_ims_test, image_size);
    uchar* ladata_test = read_mnist_labels("..\\..\\..\\..\\mnist\\t10k-labels-idx1-ubyte", n_ims_test);
    printf("Finished Loading Data.\n");
    PhaseMagNetCUDA net;
    MatrixDim in_mdim(1, 784, sizeof(DTYPE));
    LayerParams input(LayerType::input, ActivationType::relu, in_mdim);
    MatrixDim mid_mdim(1, 80, sizeof(DTYPE));
    LayerParams mid(LayerType::fc, ActivationType::relu, mid_mdim);
    /*net.addLayer(mid);*/
    MatrixDim out_mdim(1, 10, sizeof(DTYPE));
    LayerParams output(LayerType::fc, ActivationType::softmax, out_mdim);
   /* net.addLayer(input);
    net.addLayer(mid);
    net.addLayer(mid); 
    net.addLayer(output);
    net.initialize();*/
    net.load("testpmnn2.txt");
    float acc = net.evaluate(num_images_test, imdata_test, ladata_test);
    printf("Acc: %4.2f \n", acc * 100.0);
    for (int i = 1; i <= 15; ++i) {
        printf("Epoch: %d\n", i);
        net.train(num_images, imdata, ladata, /* verbose */ true);
        printf("\n");
    }
    net.save(".\\testpmnn2.txt");
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