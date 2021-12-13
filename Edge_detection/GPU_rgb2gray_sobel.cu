#include "GPU_rgb2gray_sobel.h"
#include <math.h>

#define MASK_DIM 3
#define MASK_OFFSET (MASK_DIM / 2)





__global__ void rgb2gray_GPU(unsigned char * const grayimage, unsigned char * const rgbimage, size_t numrows, size_t numcols)
{
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;


	if((row<numrows) &&(col<numcols))
	{
		  	grayimage[row*numcols+col]=rgbimage[row*numcols+col]*0.299+rgbimage[row*numcols+col+numrows*numcols]*0.587+rgbimage[row*numcols+col+2*numrows*numcols]*00.114;
	}

}



void rgb2gray_kernel_call(unsigned char * const grayimage, unsigned char * const rgbimage, size_t numrows, size_t numcols)
{
	//const dim3 blockSize(ceil(numcols/14.0), ceil(numrows/14.0), 1);  //TODO
	//const dim3 gridSize( 14, 14, 1);  //TODO
	const dim3 blockSize(ceil(numcols/40.0), ceil(numrows/40.0), 1);  //TODO
	const dim3 gridSize( 40, 40, 1);  //TODO	
	unsigned char * d_in;	
	unsigned char * d_out;	

	cudaMalloc((void **) &d_in, numrows*numcols*3);
	cudaMalloc((void **) &d_out, numrows*numcols);

	cudaMemcpy(d_in, rgbimage,numrows*numcols*3,cudaMemcpyHostToDevice);
	
	
	rgb2gray_GPU<<<gridSize,blockSize>>>(d_out,d_in,numrows,numcols);
	cudaDeviceSynchronize(); 
	cudaMemcpy(grayimage, d_out,numrows*numcols,cudaMemcpyDeviceToHost);
	
	cudaFree(d_in);
	cudaFree(d_out);
	
}
__global__ void convolution_2d( unsigned char const *matrix,char const *mask,unsigned char  *result,  size_t numrows, size_t numcols) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_r = row - MASK_OFFSET;
  int start_c = col - MASK_OFFSET;
  int temp = 0;
// Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Range check for rows
      if ((start_r + i) >= 0 && (start_r + i) < numrows) {
        // Range check for columns
        if ((start_c + j) >= 0 && (start_c + j) < numcols) {
          // Accumulate result
          temp += (int)matrix[(start_r + i) * numcols + (start_c + j)] *  (int)mask[i * MASK_DIM + j];

	  
        }
      }
    }
  }
  // Write back the result
  result[row * numcols + col] = temp; 
}
void sobel_kernel_call(unsigned char * const sobelimage, unsigned char * const grayimage , size_t numrows, size_t numcols)
{

	 unsigned char * d_in;	
	 unsigned char * d_out;	
	 char * d_mask;
	 char  maskX[9];
	 char  maskY[9];
	//sobel
	maskX[0] = (char)-1; maskX[1] = (char)0; maskX[2] = (char)1;
	maskX[3] = (char)-2; maskX[4] = (char)0; maskX[5] = (char)2;	
	maskX[6] = (char)-1; maskX[7] = (char)0; maskX[8] = (char)1;

	maskY[0] = (char)1; maskY[1] = (char)2; maskY[2] = (char)1;
	maskY[3] = (char)0; maskY[4] = (char)0; maskY[5] = (char)0;	
	maskY[6] = (char)-1; maskY[7] =(char)-2; maskY[8] =(char)-1;

	//prewit
/*	maskX[0] = 1; maskX[1] = 0; maskX[2] = -1;
	maskX[3] = 2; maskX[4] = 0; maskX[5] = -2;	
	maskX[6] = 1; maskX[7] = 0; maskX[8] = -1;

	maskY[0] = 1; maskY[1] = 2; maskY[2] = 1;
	maskY[3] = 0; maskY[4] = 0; maskY[5] = 0;	
	maskY[6] = -1; maskY[7] =-2; maskY[8] =-1;
*/
	cudaMalloc((void **) &d_in, numrows*numcols*sizeof( char));
	cudaMalloc((void **) &d_out, numrows*numcols*sizeof(char));
	cudaMalloc((void **) &d_mask, MASK_DIM*MASK_DIM*sizeof( char));	
	
	char *sobelimageX = new char[numrows*numcols];
	char *sobelimageY = new char[numrows*numcols];

	//const dim3 blockSize(ceil(numcols/14.0), ceil(numrows/14.0), 1);  //TODO
	//const dim3 gridSize( 14, 14, 1);  //TODO
	const dim3 blockSize(ceil(numcols/40.0), ceil(numrows/40.0), 1);  //TODO
	const dim3 gridSize( 40, 40, 1);  //TODO	
	cudaMemcpy(d_in, grayimage,numrows*numcols,cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask,maskX,MASK_DIM*MASK_DIM,cudaMemcpyHostToDevice);
	convolution_2d<<<gridSize,blockSize>>>(d_in,d_mask,d_out,numrows,numcols);
	cudaDeviceSynchronize(); 
	cudaMemcpy(sobelimageX, d_out,numrows*numcols,cudaMemcpyDeviceToHost);
	
	
	cudaMemcpy(d_mask,maskY,MASK_DIM*MASK_DIM,cudaMemcpyHostToDevice);
	convolution_2d<<<gridSize,blockSize>>>(d_in,d_mask,d_out,numrows,numcols);
	cudaDeviceSynchronize(); 
	cudaMemcpy(sobelimageY, d_out,numrows*numcols,cudaMemcpyDeviceToHost);
	
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_mask);


	for (int i = 0; i < numrows*numcols; i++){
		int gradientApprox = sqrt((int)(sobelimageX[i]*sobelimageX[i] + sobelimageY[i]*sobelimageY[i]));
		if (gradientApprox > 200) gradientApprox = 0;
		//else   gradientApprox = 255 ;
		
		//if (gradientApprox > 255 ) gradientApprox = 255;
		sobelimage[i] = (unsigned char)gradientApprox;
		//sobelimage[i] = 0;

				
	}
	return;
}

