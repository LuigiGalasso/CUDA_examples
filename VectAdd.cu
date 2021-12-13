#include <stdio.h>
//#include <opencv2/opencv.hpp>

__global__ void square(float * d_out, float * d_in){
int idx=(blockIdx.x*blockDim.x)+threadIdx.x;
float f=d_in[idx];
//if (idx<200)
	//d_out[idx]=__frsqrt_rn(f);
	//d_out[idx]=1.f/sqrtf(f);
	asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(d_out[idx]) : "f"(f));
	asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d_out[idx]) : "f"(f));
}

int main(int argc, char **argv){

const int ARRAY_SIZE=256*1024;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

float h_in[ARRAY_SIZE];
for(int i=0; i<ARRAY_SIZE;i++){
	h_in[i]=float(i);
}

float h_out[ARRAY_SIZE];

float * d_in;
float * d_out;

cudaMalloc((void **) &d_in, ARRAY_BYTES);
cudaMalloc((void **) &d_out, ARRAY_BYTES);

cudaMemcpy(d_in, h_in, ARRAY_BYTES,cudaMemcpyHostToDevice);

square<<<256,ARRAY_SIZE/256>>>(d_out,d_in);

cudaMemcpy(h_out, d_out, ARRAY_BYTES,cudaMemcpyDeviceToHost);


for(int i=0; i<ARRAY_SIZE;i++){
	printf("%f", h_out[i]);
	printf(((i%4)!=3) ? "\t" : "\n");
}

cudaFree(d_in);
cudaFree(d_out);

return 0;

}
