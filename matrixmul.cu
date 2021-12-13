#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>

//total number of multiplication is: 1024 rows for 1024 columns; this means that each of the 1024 row is multiplied for each of the 1024 columns element by element

//kernel: each thread process N=1024 elements (a row multiplied for a column) : each thread compute a single element of the result matrix c!
__global__ void matrixMul(int *a, int *b, int *c, int N){
	int row =blockIdx.y * blockDim.y + threadIdx.y;
	int column =blockIdx.x * blockDim.x + threadIdx.x;
	
	int sum = 0;
	if(row<N && column<N){
		for(int k=0;k<N;k++){//every single elements
				sum += a[row*N +k] * b[k*N +column];
			}	
		c[row*N + column] = sum;
	}
}
//result check function (serial implementation on CPU)
void verify(int *a, int *b, int *c, int N){
	int *verify_c;
	verify_c= (int*)malloc(N*N*sizeof(int));
	for(int i=0;i<N;i++){//every rows
		for(int j=0;j<N;j++){//every columns
			for(int k=0;k<N;k++){//every single elements
				verify_c[i*N + j] += a[i*N +k] * b[k*N +j];
			}		
		}	
	}
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			assert(c[i*N +j]==verify_c[i*N +j]);		
		}	
	}
}
int main(){
//matrix 1024*1024
int N = 1<<10;
//size of matrix in bytes is N*N*sizeof(int)
int *a, *b, *c;
int *dev_a, *dev_b, *dev_c;
//host matrix allocation and initialization
a = (int*)malloc(N*N*sizeof(int));
b = (int*)malloc(N*N*sizeof(int));
c = (int*)malloc(N*N*sizeof(int));

for(int i=0;i<N*N;i++){
	a[i]=rand()%100;
	b[i]=rand()%100;
}
//GPGPU pointers allocation
cudaMalloc(&dev_a,N*N*sizeof(int));
cudaMalloc(&dev_b,N*N*sizeof(int));
cudaMalloc(&dev_c,N*N*sizeof(int));

cudaMemcpy( dev_a,a,N*N*sizeof(int),cudaMemcpyHostToDevice ) ;
cudaMemcpy( dev_b,b,N*N*sizeof(int),cudaMemcpyHostToDevice ) ;


int BLOCK_SIZE = 32; //threads per block
int GRID_SIZE  = (int)ceil(N/BLOCK_SIZE); //grid size
//use dim3 structs for block and grid dimensions(both squared)
// 32*32(2D threads) for each of the block 32*32(2D blocks, grid)
//total number of thread is 1024 *1024  = N * N = matrix dimensions
dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
dim3 grid(GRID_SIZE,GRID_SIZE);

matrixMul <<<grid,threads>>> (dev_a,dev_b,dev_c,N);

cudaDeviceSynchronize();	cudaMemcpy(c,dev_c,N*N*sizeof(int),cudaMemcpyDeviceToHost ) ;
printf("GPU DONE \n");
verify(a, b, c, N);
printf("CPU DONE \n");
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);

free(a);
free(b);
free(c);

return 0;
}

