/* File:     3.cu
 *
 * Purpose:  Realize direct convolution through CUDA (sliding window method), see task 3 for specific requirements
 *
 * Compile:  nvcc -o 3 3.cu
 *
 * Run:      ./3 <N> <thread_block_size>
 *           thread_block_size is threads of each block
 *           N is size of input matrix
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print the intput, the kernel, and the output of convolution result
 */
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

__host__ void PRINT(double **matrix, int dim, int channel = 1){
    for(int i=0; i<channel; ++i){
        cout<<"when channel = "<< i << endl;
        for(int a=0; a<dim; ++a){
            for(int b=0; b<dim; ++b){
                cout<<matrix[i][a*dim + b]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}

__host__ void CUDA_malloc(double **matrix, int N){
    cudaError_t temp;
    for(int i=0; i<3; ++i){
        temp = cudaMallocManaged(&matrix[i], N*N*sizeof(double));
        if(temp != 0) cout<<"ERROR! "<<temp<<endl;
    }
}

__host__ void get_rand(double **matrix, int N, int a, int b){
    srand(time(NULL));
    for(int i=0; i<3; ++i){
        for(int j=0; j<N; ++j){
            matrix[i][j] = rand()%(b-a+1) + a;
        }
    }
}

__host__ void pad(double **matrix, int dim, double **matrix_new, int padding){
    CUDA_malloc(matrix_new, dim+padding);
    for(int x=0; x<3; ++x){
        for(int i=0; i<dim+padding; ++i){
            for(int j=0; j<dim+padding; ++j){
                if(i == 0 || j == 0){
                    matrix_new[x][i*(dim+padding)+j] = 0;
                }
                else{
                    matrix_new[x][i*(dim+padding)+j] = matrix[x][(i-1)*dim + (j-1)];
                }
                if(padding == 2){
                    if(i==dim+padding-1 || j==dim+padding-1){
                        matrix_new[x][i*(dim+padding)+j] = 0;
                    }
                }
            }
        }
    }
}

__global__ void CNN(double *image1, double *image2, double *image3, int image_dim, \
    double *kernel1_, double *kernel2_, double *kernel3_, int kernel_dim, \
    int stride, double *map, int map_dim){
    
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int i = n/map_dim;
    int j = n%map_dim;
    // int i = blockIdx.x;
    // int j = threadIdx.x;//卷出来的是第i行j列的元素
    map[i*map_dim + j] = 0;
    
    for(int a=0; a<kernel_dim; ++a){
        for(int b=0; b<kernel_dim; ++b){
            map[i*map_dim + j] += image1[(i*stride+a)*image_dim+j*stride+b] * kernel1_[a*3+b];
        }
    }

    for(int a=0; a<kernel_dim; ++a){
        for(int b=0; b<kernel_dim; ++b){
            map[i*map_dim + j] += image2[(i*stride+a)*image_dim+j*stride+b] * kernel2_[a*3+b];
        }
    }

    for(int a=0; a<kernel_dim; ++a){
        for(int b=0; b<kernel_dim; ++b){
            map[i*map_dim + j] += image3[(i*stride+a)*image_dim+j*stride+b] * kernel3_[a*3+b];
        }
    }

}


int main(int argc, char *argv[]){
    int N, thread_block_size;
    N = atoi(argv[1]);//the highth or width of three channels picture
    thread_block_size = atoi(argv[2]);//每个block用到的线程数
    
    double *image[3];
    double *kernel_1[3];
    double *kernel_2[3];
    double *kernel_3[3];
    double *map1[3]; //stride = 1时，3种kernel的卷积结果
    double *map2[3]; //stride = 2时，3种kernel的卷积结果
    double *map3[3]; //stride = 3时，3种kernel的卷积结果
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //分配空间给image和kernel
    CUDA_malloc(image, N);
    CUDA_malloc(kernel_1, 3);
    CUDA_malloc(kernel_2, 3);
    CUDA_malloc(kernel_3, 3);

    //获取某个范围[a,b]内的随机整数
    get_rand(image, N*N, 0, 50);
    get_rand(kernel_1, 3*3, -1, 1);
    get_rand(kernel_2, 3*3, -1, 1);
    get_rand(kernel_3, 3*3, -1, 1);

    //stride=1
    int output_size[3];
    output_size[0] = (N-3)/1 + 1;
    CUDA_malloc(map1, output_size[0]);
    dim3 blockSize(thread_block_size);
    dim3 gridSize((output_size[0]*output_size[0])/thread_block_size);
    
    CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_1[0], kernel_1[1], kernel_1[2], 3, 1, map1[0], output_size[0]);   
    CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_2[0], kernel_2[1], kernel_2[2], 3, 1, map1[1], output_size[0]);
    CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_3[0], kernel_3[1], kernel_3[2], 3, 1, map1[2], output_size[0]);


    //判断stride=2的时候是否需要填充
    int padding = (N-3)%2 ;
    if (padding){//需要填充
        double *image_new[3];
        pad( image, N, image_new, padding);//扩展成一个新矩阵
        output_size[1] = (N+padding-3)/2 + 1;
        CUDA_malloc(map2, output_size[1]);//计算得到卷积结果的大小
        blockSize = thread_block_size;
        gridSize = (output_size[1]*output_size[1])/thread_block_size;
        CNN<<<gridSize, blockSize>>>(image_new[0], image_new[1], image_new[2], N+padding, kernel_1[0], kernel_1[1], kernel_1[2], 3, 2, map2[0], output_size[1]);   
        CNN<<<gridSize, blockSize>>>(image_new[0], image_new[1], image_new[2], N+padding, kernel_2[0], kernel_2[1], kernel_2[2], 3, 2, map2[1], output_size[1]);
        CNN<<<gridSize, blockSize>>>(image_new[0], image_new[1], image_new[2], N+padding, kernel_3[0], kernel_3[1], kernel_3[2], 3, 2, map2[2], output_size[1]);
        // cout<<"After expanding the original image, the new image, stride = "<<2<<" , padding ="<<padding<<endl;
        // PRINT(image_new,N+padding,3);
        for(int i=0;i<3;++i){
            cudaFree(image_new[i]);
        }
    }
    else{
        output_size[1] = (N-3)/2 + 1;
        CUDA_malloc(map2, output_size[1]);
        blockSize = thread_block_size;
        gridSize = (output_size[1]*output_size[1])/thread_block_size;
        CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_1[0], kernel_1[1], kernel_1[2], 3, 2, map2[0], output_size[1]);   
        CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_2[0], kernel_2[1], kernel_2[2], 3, 2, map2[1], output_size[1]);
        CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_3[0], kernel_3[1], kernel_3[2], 3, 2, map2[2], output_size[1]);
    }
    

    //判断stride=3的时候是否需要填充
    padding = (3 - (N-3)%3)%3;//如果除余的结果是1，就需要填充“一圈",增加两个size；如果是2，说明需要填充一维（）这里随机加在最上面和最左边
    if (padding){
        double *image_new2[3];
        pad( image, N, image_new2, padding);
        output_size[2] = (N+padding-3)/3 + 1;
        CUDA_malloc(map3, output_size[2]);
        blockSize = thread_block_size;
        gridSize = (output_size[2]*output_size[2])/thread_block_size;
        CNN<<<gridSize, blockSize>>>(image_new2[0], image_new2[1], image_new2[2], N+padding, kernel_1[0], kernel_1[1], kernel_1[2], 3, 3, map3[0], output_size[2]);   
        CNN<<<gridSize, blockSize>>>(image_new2[0], image_new2[1], image_new2[2], N+padding, kernel_2[0], kernel_2[1], kernel_2[2], 3, 3, map3[1], output_size[2]);
        CNN<<<gridSize, blockSize>>>(image_new2[0], image_new2[1], image_new2[2], N+padding, kernel_3[0], kernel_3[1], kernel_3[2], 3, 3, map3[2], output_size[2]);
        // cout<<"After expanding the original image, the new image, stride = "<<3<<" , padding ="<<padding<<endl;
        // PRINT(image_new2,N+padding,3);
        for(int i=0;i<3;++i){
            cudaFree(image_new2[i]);
        }
    }
    else{
        output_size[2] = (N-3)/3 + 1;
        CUDA_malloc(map3, output_size[2]);
        blockSize = thread_block_size;
        gridSize = (output_size[2]*output_size[2])/thread_block_size;
        CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_1[0], kernel_1[1], kernel_1[2], 3, 3, map3[0], output_size[2]);   
        CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_2[0], kernel_2[1], kernel_2[2], 3, 3, map3[1], output_size[2]);
        CNN<<<gridSize, blockSize>>>(image[0], image[1], image[2], N, kernel_3[0], kernel_3[1], kernel_3[2], 3, 3, map3[2], output_size[2]);
    }

    
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout<<"所用时间为: "<< elapsedTime<<" <ms>"<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    

    // cout<<"image: "<<endl;
    // PRINT(image, N, 3);
    // cout<<"kernel:"<<endl;
    // cout<<"kernel_1: "<<endl;
    // PRINT(kernel_1, 3, 3);
    // cout<<"kernel_2: "<<endl;
    // PRINT(kernel_2, 3, 3);
    // cout<<"kernel_3: "<<endl;
    // PRINT(kernel_3, 3, 3);
    // cout<<"answer:"<<endl;
    // cout<<"when stride = 1, kerenl1/ kernel2/ kernel, the answer :"<<endl;
    // PRINT(map1, output_size[0], 3);
    // cout<<"when stride = 2, kerenl1/ kernel2/ kernel, the answer :"<<endl;
    // PRINT(map2, output_size[1], 3);
    // cout<<"when stride = 3, kerenl1/ kernel2/ kernel, the answer :"<<endl;
    // PRINT(map3, output_size[2], 3);
    

    for(int i=0; i<3; ++i){
        cudaFree(image[i]);
        cudaFree(kernel_1[i]);
        cudaFree(kernel_2[i]);
        cudaFree(kernel_3[i]);
        cudaFree(map1[i]);
        cudaFree(map2[i]);
        cudaFree(map3[i]);
    }

}