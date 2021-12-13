#include <iostream>
#define shMemSize 256 


__global__ void sumReduction(int *a, int *sum_a) {
	__shared__ float partial_sum[shMemSize];

	int tid = threadIdx.x+ blockIdx.x*blockDim.x;
	
	partial_sum[threadIdx.x]= a[tid];
	__syncthreads();


	for(int i=1;i<blockDim.x;i*=2){
		int index= i*2*threadIdx.x;
		if(index<blockDim.x) partial_sum[index]+=partial_sum[index+i];	
		__syncthreads();
	}

	if(threadIdx.x == 0)
		sum_a[blockIdx.x] = partial_sum[0];
	
}

int main() {
	int *a, *sum_a;
	int *dev_a, *dev_sum_a;
	int N = 1 << 16; 
	
	a = (int*)malloc(N*sizeof(int));
	sum_a = (int*)malloc(N*sizeof(int));
		
	
	cudaMalloc( (void**)&dev_a, N*sizeof(int) ) ;
	cudaMalloc( (void**)&dev_sum_a, N*sizeof(int) ) ;
	
	
	for(int i=0;i<N;i++) a[i] = i;

	cudaMemcpy( dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice ) ;
	
	int threads = 256;
	int grid = N/threads; 

sumReduction<<<grid,threads>>>( dev_a, dev_sum_a);
cudaDeviceSynchronize();
sumReduction<<<1,threads>>>( dev_sum_a, dev_sum_a);


cudaDeviceSynchronize();	
cudaMemcpy( sum_a,dev_sum_a,N*sizeof(int),cudaMemcpyDeviceToHost ) ;

	
	printf("N %d result is %d\n",N,sum_a[0]);	
	

	cudaFree(dev_a);
	cudaFree(dev_sum_a);

	
	free(a);
	free(sum_a);

	
	return 0;
}
