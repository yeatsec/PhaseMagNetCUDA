
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
    int num_images = 10000;
    int image_size = 784;
    uchar** imdata = read_mnist_images("..\\..\\..\\..\\mnist\\train-images-idx3-ubyte", num_images, image_size);
    uchar* ladata = read_mnist_labels("..\\..\\..\\..\\mnist\\train-labels-idx1-ubyte", num_images);
    PhaseMagNetCUDA net;
    //MatrixDim in_mdim(1, 784, sizeof(DTYPE));
    //LayerParams input(LayerType::input, ActivationType::relu, in_mdim);
    //net.addLayer(input);
    //MatrixDim mid_mdim(1, 40, sizeof(DTYPE));
    //LayerParams mid(LayerType::fc, ActivationType::relu, mid_mdim);
    ///*net.addLayer(mid);
    //net.addLayer(mid);
    //net.addLayer(mid);*/
    //MatrixDim out_mdim(1, 10, sizeof(DTYPE));
    //LayerParams output(LayerType::fc, ActivationType::relu, out_mdim);
    //net.addLayer(output);
    //net.initialize();
    net.load("test.txt");
    float acc = net.evaluate(1000, imdata, ladata);
    printf("Acc: %4.2f \n", acc * 100.0);
    net.train(50000, imdata, ladata, /* verbose */ true);
    net.save(".\\test.txt");
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