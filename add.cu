#include <iostream>

//grid composed of blocks composed of threads
__global__ void add(int *a, int *b, int *c, int v) {
	int tid = threadIdx.x+ blockIdx.x*blockDim.x;
	while(tid < v){
		c[tid]= a[tid] +b[tid];
		tid += blockDim.x*gridDim.x;//after thread finishes tid is incremented of the the total number of threads running in the grid to start working on the next element
	}
}

int main( void ) {
	int v = 123476;// not power of 2 need to pad the dimensions 
	int m = 1<<15;
	int a[v],b[v], c[v];
	int *dev_a, *dev_b, *dev_c;
	cudaMalloc( (void**)&dev_a, v*sizeof(int) ) ;
	cudaMalloc( (void**)&dev_b, v*sizeof(int) ) ;
	cudaMalloc( (void**)&dev_c, v*sizeof(int) ) ;
	
	for(int i=0;i<v;i++) {
		a[i] = i;
		if(i < m ) b[i] = i^2;
		else b[i]=0;	
	}
	cudaMemcpy( dev_a,a,v*sizeof(int),cudaMemcpyHostToDevice ) ;
	cudaMemcpy( dev_b,b,v*sizeof(int),cudaMemcpyHostToDevice ) ;

	int nthreadPerBlock = 128;
	int nblock = (v+nthreadPerBlock-1)/nthreadPerBlock;
	add<<<nblock,nthreadPerBlock>>>( dev_a, dev_b, dev_c,v );

	cudaMemcpy( &c,dev_c,v*sizeof(int),cudaMemcpyDeviceToHost ) ;
	
	for(int i=0;i<v;i++){
		printf("%d + %d = %d\n",a[i],b[i],c[i]);	
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	return 0;
}
