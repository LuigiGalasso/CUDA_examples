#include <iostream>
#define imin(a,b) (a<b?a:b)
const int N = 33*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32,(N+threadsPerBlock-1)/ threadsPerBlock);


__global__ void dot(float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock]; //shared by threads of the same block!

	int tid = threadIdx.x+ blockIdx.x*blockDim.x;
	int cacheIndex =threadIdx.x;
	float temp = 0;
	while(tid<N){
		temp+= a[tid] * b[tid];
		tid+=blockDim.x*gridDim.x;
	}
	cache[cacheIndex] = temp;
	
	__syncthreads();

//reduction sum performed by the initial half of the threads, then halfed and halfed
	int i= blockDim.x/2;
	while(i!=0){
		if(cacheIndex <i)
			cache[cacheIndex]+= cache[cacheIndex +i];
	__syncthreads();//each thread need to be informed so wait until done
	i/=2;	
	}
//sum of each block is in cache[0](shared by each thread in a block)
	if(cacheIndex == 0)
		c[blockIdx.x] = cache[0];

}

int main( void ) {
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	
	//allocate memory on the CPU side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));	
	
	cudaMalloc( (void**)&dev_a, N*sizeof(float) ) ;
	cudaMalloc( (void**)&dev_b, N*sizeof(float) ) ;
	cudaMalloc( (void**)&dev_partial_c, blocksPerGrid*sizeof(float) ) ;
	
	for(int i=0;i<N;i++) {
		a[i] = i;
		b[i] = i*2;	
	}
	cudaMemcpy( dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice ) ;
	cudaMemcpy( dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice ) ;


dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b, dev_partial_c );

cudaDeviceSynchronize();	cudaMemcpy( partial_c,dev_partial_c,blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost ) ;

//final sum implmented at CPU side, sum each of the partial_c
	c=0;
	for(int i=0;i<blocksPerGrid;i++) c+=partial_c[i];

	
	printf("result is %f\n",c);	
	

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	
	free(a);
	free(b);
	free(partial_c);
	
return 0;
}
