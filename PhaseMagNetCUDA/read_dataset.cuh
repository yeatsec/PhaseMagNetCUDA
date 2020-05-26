

#ifndef READDATASET_CUH
#define READDATASET_CUH

// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c


#include <fstream>
#include <iostream>
#include <vector>
#include <string>

typedef unsigned char uchar;

uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size);

// msb is c4, lsb is c1
void makeReverseInt(int i, uchar& c1, uchar& c2, uchar& c3, uchar& c4);

void write_mnist_images(std::string full_path, int number_of_images, int image_rows, int image_cols, uchar** data);


uchar* const read_mnist_labels(std::string full_path, int& number_of_labels);

void printimage(uchar* image, size_t rows, size_t cols);


#endif // READDATASET_CUH