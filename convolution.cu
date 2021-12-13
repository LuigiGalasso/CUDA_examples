#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <math.h>
#include <fstream>

using namespace std;

//convolution mask
#define MASK_LENGTH 7
//allocate space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

/*some threads load multiple elements into shared memory
all threads compute 1 element in final array
*/
__global__ void convolution1d(int *array, int *result, int N){
	int tid = threadIdx.x+ blockIdx.x*blockDim.x;
	extern __shared__ int shared_array[];
	
	//r is the number of padded alements on either sides of shared memory
	int r = MASK_LENGTH /2;
	//d is the total number of padded elements
	int d = 2 * r;
	//size of padded shared memory
	int n_padded = blockDim.x + d;
	//offset for the load in shared memory
	int offset = threadIdx.x + blockDim.x;
	//global offset for array in DRAM
	int g_offset = blockDim.x * blockIdx.x + offset;
	//load lower elements first starting at the halo (right side let's say)
	shared_array[threadIdx.x] = array[tid];
	//load the upper elements 
	if(offset < n_padded)
		shared_array[threadIdx.x] = array[g_offset];
	__syncthreads();
	int temp = 0;
	
	for (int i = 0; i< MASK_LENGTH, i++)
		temp += shared_array[threadIdx.x + i] * mask[i];
	
	result[tid] = temp;
}
	


int main(){
	int N = 1 << 19;

	
	//radius for padding the array
	int r = MASK_LENGTH / 2;
	int n_p = N + r * 2;
	
	int *array = new int[n_p];
	//initialize
	for(int i = 0; i<n_p ; i++){
		if((i < r)||(i>=(N+r)){
			array[i] = 0;	
		}
		else 
			array[i] = rand()%10;
	}	
	
	//mask inititialization
	int *mask = new int[MASK_LENGTH];
	for(int i = 0; i< MASK_LENGTH;i++)
		mask[i]= rand()%100;
	
	int *dev_array,*dev_result;

	cudaMalloc( (void**)&dev_array, n_p*sizeof(char) ) ;
	cudaMalloc( (void**)&dev_result, MASK_LENGTH*sizeof(int) ) ;
	
	cudaMemcpy( dev_array,array,n_p*sizeof(char),cudaMemcpyHostToDevice ) ;
	
	//copy mask directly to the symbol
	//instead of use 2 API with cudaMemcpy
	cudaMemcpyToSymbol(mask,dev_mask,MASK_LENGTH*sizeof(int))
;
	int THREADS = 256;
	int BLOCKS = (N + THREADS -1)/THREADS;
	
	size_t SHMEM = (THREADS +r*2) * sizeof(int);

	convolution<<<BLOCKS,THREADS,SHMEM>>>(dev_array,dev_result,N); 
	cudaMemcpy(result,dev_result,N*sizeof(int),cudaMemcpyDeviceToHost ) ;

//	for(int i=0;i<BINS;i++) printf("BINS %d : %d\n",i,result[i]);
	cudaFree(dev_result);
	cudaFree(dev_array);


	return 0;


}
