#include <iostream>

__global__ void add( int a, int b, int *c ) {
*c = a + b;
}
int main( void ) {
int c;
int *dev_c;
cudaMalloc( (void**)&dev_c, sizeof(int) ) ;
add<<<1,1>>>( 2, 7, dev_c );
cudaMemcpy( &c,dev_c,sizeof(int),cudaMemcpyDeviceToHost ) ;
printf( "2 + 7 = %d\n", c ); //example
int count;
printf("%d\n",cudaGetDeviceCount( &count ));
cudaFree( dev_c );
//Collect device information
cudaDeviceProp prop;
cudaGetDeviceProperties( &prop, 0) ;
printf("%d\n",prop);
printf( " --- General Information for device %d ---\n", 0 );
printf( "Name: %s\n", prop.name );
printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
printf( "Clock rate: %d\n", prop.clockRate );
printf( "Device copy overlap: " );
if (prop.deviceOverlap)
printf( "Enabled\n" );
else
printf( "Disabled\n" );
printf( "Kernel execition timeout : " );
if (prop.kernelExecTimeoutEnabled)
printf( "Enabled\n" );
else
printf( "Disabled\n" );
printf( " --- Memory Information for device %d ---\n", 0);
printf( "Total global mem: %ld\n", prop.totalGlobalMem );
printf( "Total constant Mem: %ld\n", prop.totalConstMem );
printf( "Max mem pitch: %ld\n", prop.memPitch );
printf( "Texture Alignment: %ld\n", prop.textureAlignment );
printf( " --- MP Information for device %d ---\n", 0 );
printf( "Multiprocessor count: %d\n",
prop.multiProcessorCount );
printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
printf( "Registers per mp: %d\n", prop.regsPerBlock );
printf( "Threads in warp: %d\n", prop.warpSize );
printf( "Max threads per block: %d\n",
prop.maxThreadsPerBlock );
printf( "Max thread dimensions: (%d, %d, %d)\n",
prop.maxThreadsDim[0], prop.maxThreadsDim[1],
prop.maxThreadsDim[2] );
printf( "Max grid dimensions: (%d, %d, %d)\n",
prop.maxGridSize[0], prop.maxGridSize[1],
prop.maxGridSize[2] );
printf( "\n" );

return 0;
}
