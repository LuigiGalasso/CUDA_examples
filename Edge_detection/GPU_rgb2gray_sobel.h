#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

void rgb2gray_kernel_call(unsigned char * const grayimage, unsigned char * const rgbimage, size_t numrows, size_t numcols);
void sobel_kernel_call(unsigned char * const sobelimage, unsigned char * const grayimage , size_t numrows, size_t numcols);
