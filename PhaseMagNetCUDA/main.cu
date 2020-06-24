
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include "LinkedList.cuh"
#include "pmncudautils.cuh"
#include "cudafuncs.cuh"
#include "PhaseMagNetCUDA.cuh"
#include "read_dataset.cuh"

void buildNetworkLenet5(PhaseMagNetCUDA& net) {
    const int numConv1Filters = 24;
    const int numConv2Filters = 64;
    //LayerType convLayType(LayerType::conv);
    LayerType poolLayType(LayerType::avgpool);
    /* Input Layer */
    MatrixDim in_mdim(28, 28, sizeof(DTYPE), 1);
    LayerParams input(LayerType::input, ActivationType::softmax, in_mdim); // should be linear in case of adversarial example generation
    net.addLayer(input);
    /* Conv1 */
    ConvParams conv1;
    MatrixDim fDim(5, 5, sizeof(DTYPE), 1);
    conv1.filterDim = fDim;
    conv1.pad = 2;
    conv1.stride = 1; // conv only supports stride of 1
    conv1.numFilters = numConv1Filters;
    MatrixDim convMdim(conv1.getNextActDim(in_mdim, sizeof(DTYPE)));
    LayerParams convLayer(LayerType::conv, ActivationType::relu, convMdim, conv1);
    net.addLayer(convLayer);
    printf("Conv1 Created\n");
    /* Average Pooling */
    ConvParams avgPoolParams;
    MatrixDim tmp(2, 2, sizeof(DTYPE), numConv1Filters); // change back to 6 for conv
    avgPoolParams.filterDim = tmp;
    avgPoolParams.pad = 0;
    avgPoolParams.stride = 2;
    avgPoolParams.numFilters = numConv1Filters; // change back to 6
    MatrixDim avgPoolMdim(avgPoolParams.getNextActDim(convMdim, sizeof(DTYPE)));
    LayerParams avgPoolLayer(poolLayType, ActivationType::relu, avgPoolMdim, avgPoolParams);
    net.addLayer(avgPoolLayer);
    printf("AvgPool Created\n");
    /* Conv 2 */
    ConvParams conv2;
    MatrixDim fdim2(5, 5, sizeof(DTYPE), numConv1Filters);
    conv2.filterDim = fdim2;
    conv2.pad = 0;
    conv2.numFilters = numConv2Filters;
    conv2.stride = 1; // conv only supports stride of 1
    MatrixDim convMdim2(conv2.getNextActDim(avgPoolMdim, sizeof(DTYPE)));
    LayerParams convLayer2(LayerType::conv, ActivationType::relu, convMdim2, conv2);
    net.addLayer(convLayer2);
    printf("Conv2 Layer Created\n");
    /* Average Pooling 2 */
    ConvParams avgPoolParams2;
    MatrixDim avgpfilt(2, 2, sizeof(DTYPE), numConv2Filters);
    avgPoolParams2.filterDim = avgpfilt;
    avgPoolParams2.pad = 0;
    avgPoolParams2.stride = 2;
    avgPoolParams2.numFilters = numConv2Filters;
    MatrixDim avgPoolMdim2(avgPoolParams2.getNextActDim(convMdim2, sizeof(DTYPE)));
    LayerParams avgPoolLayer2(poolLayType, ActivationType::relu, avgPoolMdim2, avgPoolParams2);
    net.addLayer(avgPoolLayer2);
    printf("AvgPool 2 Created\n");
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

void run_mnist(char* model_name, char* model_savename, bool buildNetwork, char* testName, float lrnRate, int numEpochs, float dropout) {
    printf("Running MNIST Model... \n");
    int n_ims_train = 50000;
    int n_ims_test = 10000;
    int image_size = 784;
    /* Network File / Test Set file changes */
    PhaseMagNetCUDA net;

    if (buildNetwork) {
        buildNetworkLenet5(net);
    }
    else {
        std::cout << "Loading: " << model_name << std::endl;
        net.load(model_name);
    }
   
    std::cout << "Will Save as: " << model_savename << std::endl;
    
    printf("Loading Data...\n");
    printf("Test Set: %s \n", testName);
    //uchar** imdata_train = read_mnist_images("..\\..\\..\\..\\fashion-mnist\\data\\fashion\\train-images.idx3-ubyte", n_ims_train, image_size);
    uchar** imdata_train = read_mnist_images("..\\..\\..\\..\\mnist\\train-images.idx3-ubyte", n_ims_train, image_size);
    uchar* ladata_train = read_mnist_labels("..\\..\\..\\..\\mnist\\train-labels.idx1-ubyte", n_ims_train);
    uchar** imdata_test = read_mnist_images(testName, n_ims_test, image_size);
    uchar* ladata_test = read_mnist_labels("..\\..\\..\\..\\mnist\\t10k-labels.idx1-ubyte", n_ims_test);
    printf("Finished Loading Data.\n");
    float acc;


    //char* advName = "fmnist_scalar0_chkpt0_fgsm_eps0.3_step10.idx3-ubyte";
    //net.genAdv(advName, n_ims_test, 28, 28, imdata_test, ladata_test, 0.30, 10, /* targeted */ false, /* randomStart */ false, /* verbose */ true);
    //int n_ims_adv, adv_ims_size;
    //uchar** adv_set = read_mnist_images(advName, n_ims_adv, adv_ims_size);
    //acc = net.evaluate(/*n_ims_test*/ n_ims_adv, adv_set, ladata_test, /* verbose */ true);
    //printf("Adv Acc: %4.2f \n", acc * 100.0);
    acc = net.evaluate(/*n_ims_test*/ 10000, imdata_test, ladata_test, /* verbose */ true);
    printf("Acc: %4.2f \n", acc * 100.0);

    for (int i = 1; i <= numEpochs; ++i) {
        printf("Epoch: %d\n", i);
        net.train( n_ims_train, imdata_train, ladata_train, /* */ lrnRate, /* verbose */ true, dropout);
        printf("\n");
        acc = net.evaluate( n_ims_test, imdata_test, ladata_test, /* verbose */ true);
        printf("Acc: %4.2f \n", acc * 100.0);
        char buffer[300];
        int n = sprintf(buffer, "mnist_autosave_%s", model_name);
        net.save(buffer);
        net.save("mnist_autosave.txt");
    }
    std::cout << "Saving network file to " << model_savename << std::endl;
    net.save(model_savename);
    net.free();
}

void buildNetworkVGG8(PhaseMagNetCUDA& net) {
    LayerType convLayType1(LayerType::conv);
    ActivationType convActType(ActivationType::relu);
    LayerType convLayType2(LayerType::phasorconv); // check where these are applied
    LayerType poolLayType(LayerType::avgpool);
    int nFconv1_1 = 32;
    int nFconv1_2 = 32;
    int nFconv2_1 = 64;
    int nFconv2_2 = 64;
    int nFconv3_1 = 128;
    int nFconv3_2 = 128;
    int nfc = 64;
    int nout = 10;

    /* Input Layer */
    MatrixDim in_mdim(CIFAR_DIM, CIFAR_DIM, sizeof(DTYPE), CIFAR_CHAN);
    LayerParams input(LayerType::input, ActivationType::softmax, in_mdim);
    net.addLayer(input);
    /* Conv1_1 */
    ConvParams conv1_1;
    MatrixDim fDim1_1(5, 5, sizeof(DTYPE), 3);
    conv1_1.filterDim = fDim1_1;
    conv1_1.pad = 2;
    conv1_1.stride = 1; // conv only supports stride of 1
    conv1_1.numFilters = nFconv1_1;
    MatrixDim convMdim1_1(conv1_1.getNextActDim(in_mdim, sizeof(DTYPE)));
    LayerParams convLayer1_1(convLayType2, convActType, convMdim1_1, conv1_1);
    net.addLayer(convLayer1_1);
    printf("Conv1_1 Created\n");
    /* Conv1_2 */
    ConvParams conv1_2;
    MatrixDim fDim1_2(3, 3, sizeof(DTYPE), nFconv1_1);
    conv1_2.filterDim = fDim1_2;
    conv1_2.pad = 1;
    conv1_2.stride = 1;
    conv1_2.numFilters = nFconv1_2;
    MatrixDim convMdim1_2(conv1_2.getNextActDim(convMdim1_1, sizeof(DTYPE)));
    LayerParams convLayer1_2(convLayType1, convActType, convMdim1_2, conv1_2);
    net.addLayer(convLayer1_2);
    printf("Conv1_2 Created\n");
    /* Max Pool 1 */
    ConvParams maxpool1;
    MatrixDim maxpool1fdim(2, 2, sizeof(DTYPE), nFconv1_2);
    maxpool1.filterDim = maxpool1fdim;
    maxpool1.pad = 0;
    maxpool1.stride = 2;
    maxpool1.numFilters = nFconv1_2; // must equal that of prev
    MatrixDim maxpool1mdim(maxpool1.getNextActDim(convMdim1_2, sizeof(DTYPE)));
    LayerParams maxpoolLayer1(poolLayType, convActType, maxpool1mdim, maxpool1);
    net.addLayer(maxpoolLayer1);
    printf("MaxPool1 Created\n");
    /* Conv 2_1 */
    ConvParams conv2_1;
    MatrixDim fDim2_1(3, 3, sizeof(DTYPE), nFconv1_2); // depth must be same as that of prev layer
    conv2_1.filterDim = fDim2_1;
    conv2_1.pad = 1;
    conv2_1.stride = 1;
    conv2_1.numFilters = nFconv2_1;
    MatrixDim convMdim2_1(conv2_1.getNextActDim(maxpool1mdim, sizeof(DTYPE)));
    LayerParams convLayer2_1(convLayType1, convActType, convMdim2_1, conv2_1);
    net.addLayer(convLayer2_1);
    printf("Conv2_1 Created\n");
    /* Conv 2_2 */
    ConvParams conv2_2;
    MatrixDim fDim2_2(3, 3, sizeof(DTYPE), nFconv2_1);
    conv2_2.filterDim = fDim2_2;
    conv2_2.pad = 1;
    conv2_2.stride = 1;
    conv2_2.numFilters = nFconv2_2;
    MatrixDim convMdim2_2(conv2_2.getNextActDim(convMdim2_1, sizeof(DTYPE)));
    LayerParams convLayer2_2(convLayType1, convActType, convMdim2_2, conv2_2);
    net.addLayer(convLayer2_2);
    printf("Conv2_2 Created\n");
    /* Max Pool 2 */
    ConvParams maxpool2;
    MatrixDim maxpool2fdim(2, 2, sizeof(DTYPE), nFconv2_2);
    maxpool2.filterDim = maxpool2fdim;
    maxpool2.pad = 0;
    maxpool2.stride = 2;
    maxpool2.numFilters = nFconv2_2; // must equal that of prev
    MatrixDim maxpool2mdim(maxpool2.getNextActDim(convMdim2_2, sizeof(DTYPE)));
    LayerParams maxpoolLayer2(poolLayType, convActType, maxpool2mdim, maxpool2);
    net.addLayer(maxpoolLayer2);
    printf("MaxPool2 Created\n");
    /* Conv 3_1 */
    ConvParams conv3_1;
    MatrixDim fDim3_1(3, 3, sizeof(DTYPE), nFconv2_2);
    conv3_1.filterDim = fDim3_1;
    conv3_1.pad = 1;
    conv3_1.stride = 1;
    conv3_1.numFilters = nFconv3_1;
    MatrixDim convMdim3_1(conv3_1.getNextActDim(maxpool2mdim, sizeof(DTYPE)));
    LayerParams convLayer3_1(convLayType1, convActType, convMdim3_1, conv3_1);
    net.addLayer(convLayer3_1);
    printf("Conv3_1 Created\n");
    /* Conv 3_2 */
    ConvParams conv3_2;
    MatrixDim fDim3_2(3, 3, sizeof(DTYPE), nFconv3_1);
    conv3_2.filterDim = fDim3_2;
    conv3_2.pad = 1;
    conv3_2.stride = 1;
    conv3_2.numFilters = nFconv3_2;
    MatrixDim convMdim3_2(conv3_2.getNextActDim(convMdim3_1, sizeof(DTYPE)));
    LayerParams convLayer3_2(convLayType2, convActType, convMdim3_2, conv3_2);
    net.addLayer(convLayer3_2);
    printf("Conv3_2 Created\n");
    /* Max Pool 3 */
    ConvParams maxpool3;
    MatrixDim maxpool3fdim(2, 2, sizeof(DTYPE), nFconv3_2);
    maxpool3.filterDim = maxpool3fdim;
    maxpool3.pad = 0;
    maxpool3.stride = 2;
    maxpool3.numFilters = nFconv3_2; // must equal that of prev
    MatrixDim maxpool3mdim(maxpool3.getNextActDim(convMdim3_2, sizeof(DTYPE)));
    LayerParams maxpoolLayer3(poolLayType, ActivationType::sigmoid, maxpool3mdim, maxpool3);
    net.addLayer(maxpoolLayer3);
    printf("Max Pool 3 Created\n");
    /* FC 1 */
    MatrixDim fc1_mdim(1, nfc, sizeof(DTYPE));
    LayerParams fc1(LayerType::fc, ActivationType::sigmoid, fc1_mdim);
    net.addLayer(fc1);
    printf("FC 1 Created\n");
    /* Output */
    MatrixDim fc2_mdim(1, nout, sizeof(DTYPE));
    LayerParams fc2(LayerType::fc, ActivationType::softmax, fc2_mdim);
    net.addLayer(fc2);
    printf("Output Created\n");
    net.initialize();
}

void run_cifar(char* model_name, char* model_savename, bool buildNetwork, char* testName, float lrnRate, int numEpochs, float dropout) {
    printf("Running CIFAR10 Model... \n");
    PhaseMagNetCUDA net;
    
    
    if (buildNetwork) {
        buildNetworkVGG8(net);
    }
    else {
        std::cout << "Loading: " << model_name << std::endl;
        net.load(model_name);
    }
   
    std::cout << "Will Save as: " << model_savename << std::endl;

    const int n_ims_train = 10000;
    const int n_ims_test = 10000;
    printf("Loading Data...\n");
    printf("Test Set: %s \n", testName);
    uchar* ladata_train1;
    uchar** imdata_train1 = read_cifar10_images_labels("..\\..\\..\\..\\cifar10\\data_batch_1.bin", n_ims_train, &ladata_train1);
    uchar* ladata_train2;
    uchar** imdata_train2 = read_cifar10_images_labels("..\\..\\..\\..\\cifar10\\data_batch_2.bin", n_ims_train, &ladata_train2);
    uchar* ladata_train3;
    uchar** imdata_train3 = read_cifar10_images_labels("..\\..\\..\\..\\cifar10\\data_batch_3.bin", n_ims_train, &ladata_train3);
    uchar* ladata_train4;
    uchar** imdata_train4 = read_cifar10_images_labels("..\\..\\..\\..\\cifar10\\data_batch_4.bin", n_ims_train, &ladata_train4);
    uchar* ladata_train5;
    uchar** imdata_train5 = read_cifar10_images_labels("..\\..\\..\\..\\cifar10\\data_batch_5.bin", n_ims_train, &ladata_train5);
    uchar* ladata_test;
    uchar** imdata_test = read_cifar10_images_labels(testName, n_ims_test, &ladata_test);
    printf("Finished Loading Data.\n");

    printf("Dropout: %2.3f\n", dropout);
    for (int i = 1; i <= numEpochs; ++i) {
        printf("Epoch: %d\n", i);
        float acc = net.evaluate(/*n_ims_test*/ 10000, imdata_test, ladata_test, /* verbose */ true);
        printf("Acc: %4.2f \n", acc * 100.0);
        net.train(/* n_ims_train */ 10000, imdata_train1, ladata_train1, /* */ lrnRate, /* verbose */ true, dropout);
        net.save("cifar_autosave.txt");
        net.train(/* n_ims_train */ 10000, imdata_train2, ladata_train2, /* */ lrnRate, /* verbose */ true, dropout);
        net.save("cifar_autosave.txt");
        net.train(/* n_ims_train */ 10000, imdata_train3, ladata_train3, /* */ lrnRate, /* verbose */ true, dropout);
        net.save("cifar_autosave.txt");
        net.train(/* n_ims_train */ 10000, imdata_train4, ladata_train4, /* */ lrnRate, /* verbose */ true, dropout);
        net.save("cifar_autosave.txt");
        net.train(/* n_ims_train */ 10000, imdata_train5, ladata_train5, /* */ lrnRate, /* verbose */ true, dropout);
        net.save("cifar_autosave.txt");
        printf("\n");
        char buffer[300];
        int n = sprintf(buffer, "cifar_autosave_%s", model_name);
        net.save(buffer);
    }
    std::cout << "Saving network file to " << model_savename << std::endl;
    net.save(model_savename);
    net.free();
}

int main()
{
    bool mnist = true;
    if (mnist)
    {
        char* modelName = "lenet5_phasor2_chkpt0.txt";
        char* saveName = "lenet5_phasor2_chkpt1.txt";
        bool build = false; // check buildLenet5 if true, check SEED
        //char* testName = "..\\..\\..\\..\\fashion-mnist\\data\\fashion\\fmnist_net0.pth_eps0.040.idx3-ubyte"; // t10k-images.idx3-ubyte // ann_a_advclp_0.2eps-ubyte // lowfreq_eps0.200_mnist_test_images.idx3-ubyte
        char* testName = "..\\..\\..\\..\\mnist\\ann_a_advclp_0.2eps-ubyte"; // mnist_net1.pth_eps0.100.idx3-ubyte
        //char* testName = "fmnist_scalar0_chkpt0_fgsm_eps0.1_step1.idx3-ubyte";
        float lrnRate = 0.0002f;
        int numEpochs = 5;
        float dropout = 0.0f;
        run_mnist(modelName, saveName, build, testName, lrnRate, numEpochs, dropout);
    }
    else {
        char* modelName = "VGG8_mixed4_chkpt0.txt";
        char* saveName = "VGG8_mixed4_chkpt0.txt";
        bool build = true; // check buildVGG8Network if true, check SEED
        char* testName = "..\\..\\..\\..\\cifar10\\test_batch.bin"; // test_batch.bin //VGG8Net_nonorm_fgsm_eps8.bin
        float lrnRate = 0.001f;
        int numEpochs = 5;
        float dropout = 0.0f;

        run_cifar(modelName, saveName, build, testName, lrnRate, numEpochs, dropout);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}