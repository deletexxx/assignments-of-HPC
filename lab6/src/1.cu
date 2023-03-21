/* File:     1.cu
 *
 * Purpose:  Implement parallel version of general matrix multiplication through CUDA
 *
 * Compile:  nvcc -o 1 1.cu
 *
 * Run:      ./1 <thread_block_size> <M> <K> <N>
 *           thread_block_size is threads of each block
 *           M, K, N is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
 */

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

__host__ void PRINT(double *A, int x, int y){
    for(int i=0;i<x;++i){
        for(int j=0;j<y;++j) cout<<A[i*y+j]<<"  ";
        cout<<endl;
    }
}

__global__ void GEMM(double *A, double *B, double *C,int M, int N, int K){
    // int i = blockIdx.x;
    // int j = threadIdx.x;
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int i = n/N;
    int j = n%N;
    C[i*N+j] = 0;
    for(int k=0; k<K; ++k)
        C[i*N+j] += A[i*K+k]*B[j+k*N];
}

int main(int argc, char *argv[]){
    int M, K, N, thread_block_size;
    double *A,*B,*C;
    thread_block_size = atoi(argv[1]);
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    N = atoi(argv[4]);
    cout<<"the size of matrix: M = "<<M<<", K = "<<K<<", N = "<<N<<endl;

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);//在GPU中分配显存也算时间

    cudaMallocManaged((void**)&A, M*K*sizeof(double));
    cudaMallocManaged((void**)&B, K*N*sizeof(double));
    cudaMallocManaged((void**)&C, M*N*sizeof(double));

    for(int i=0; i<M; ++i)
        for(int j=0; j<K; ++j) A[i*K+j] = rand()%100;
    for(int i=0; i<K; ++i)
        for(int j=0; j<N; ++j) B[i*N+j] = rand()%100;
    
    dim3 blockSize(thread_block_size);
    // dim3 blockSize(N);
    // dim3 gridSize(M);
    dim3 gridSize((M*N)/thread_block_size);


    GEMM<<<gridSize, blockSize>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();//设备同步

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime=0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout<<"所用时间为: "<< elapsedTime<<" <ms>"<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cout<<"A = "<<endl;
    // PRINT(A, M, K);
    // cout<<"B = "<<endl;
    // PRINT(B, K, N);
    // cout<<"C = "<<endl;
    // PRINT(C, M, N);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
    
}