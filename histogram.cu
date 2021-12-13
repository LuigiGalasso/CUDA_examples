#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <math.h>
#include <fstream>

using namespace std;

//number of bins of the plot
const int BINS = 7;
const int DIV  = ((26+BINS -1)/BINS); //alphabet is composed of 26 letters

//GPU kernel for coputing a histogram
//a:array in global memory
//N:size of array

__global__ void histogram(char *a, int *result, int N){
	int tid = threadIdx.x+ blockIdx.x*blockDim.x;
	//bin position where threads are grouped together
	int alpha_position;
	for(int i = tid;i<N;i+=(blockDim.x * gridDim.x)){
	//position in the alphabet subtracting ASCII value
		alpha_position = a[i] - 'a';
		atomicAdd(&result[alpha_position/DIV],1);
	}
}

int main(){
	int N = 1 << 22;
	char vector [N];
	int result [BINS];
	//initialize array
	for (int i = 0;i < N;i++) vector[i] = 'a' + rand()%26;

	
	char *input;
	int *output;
	cudaMalloc( (void**)&input, N*sizeof(char) ) ;
	cudaMalloc( (void**)&output, BINS*sizeof(int) ) ;
	
	cudaMemcpy( input,vector,N*sizeof(char),cudaMemcpyHostToDevice ) ;
	
	int THREADS = 512;
	int BLOCKS = (N + THREADS -1)/THREADS;
	
	histogram<<<BLOCKS,THREADS>>>(input,output,N); 
	cudaMemcpy(result,output,BINS*sizeof(int),cudaMemcpyDeviceToHost ) ;
	for(int i=0;i<BINS;i++) printf("BINS %d : %d\n",i,result[i]);
	cudaFree(input);
	cudaFree(output);


	return 0;


}
