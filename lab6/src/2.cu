/* File:     2.cu
 *
 * Purpose:  Calculate matrix multiplication through NVDIA's matrix calculation function library CUBLAS
 *
 * Compile:  nvcc -o 2 2.cu -lcublas
 *
 * Run:      ./2 <M> <K> <N>
 *           M, K, N is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
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

__host__ void PRINT_single(double *matrix, int M, int N){
        for(int a=0; a<M; ++a){
            for(int b=0; b<N; ++b){
                cout<<matrix[a*N + b]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
}

__host__ void CUDA_malloc(double **matrix, int N){
    cudaError_t temp;
    for(int i=0; i<3; ++i){
        temp = cudaMallocManaged(&matrix[i], N*N*sizeof(double));
        if(temp != 0) cout<<"ERROR! "<<temp<<endl;
    }
}

__host__ void get_rand(double **matrix, int N, int a, int b){
    srand(clock());
    for(int i=0; i<3; ++i){
        for(int j=0; j<N; ++j){
            matrix[i][j] = rand()%(b-a+1) + a;
        }
    }
}

__host__ void pad(double **matrix, int dim, double **matrix2, int padding){
    CUDA_malloc(matrix2, dim+padding);
    for(int x=0; x<3; ++x){
        for(int i=0; i<dim+padding; ++i){
            for(int j=0; j<dim+padding; ++j){
                if(i == 0 || j == 0){
                    matrix2[x][i*(dim+padding)+j] = 0;
                }
                else{
                    matrix2[x][i*(dim+padding)+j] = matrix[x][(i-1)*dim + (j-1)];
                }
                if(padding == 2){
                    if(i==dim+padding-1 || j==dim+padding-1){
                        matrix2[x][i*(dim+padding)+j] = 0;
                    }
                }
            }
        }
    }
    
}


__host__ void convertx(double **image, int dim, int stride, double *matrix){
    int col = 0;
    for(int i=0; i<dim-2; i=i+stride){
        for(int j=0; j<dim-2; j=j+stride){//i j表示这次划分从image的i行j列开始
            for(int x=0; x<3; ++x){//3是通道数
                for(int a=0; a<3; ++a){//3是kernel的维度
                    for(int b=0; b<3; ++b){//a b x控制matrix的列数
                        matrix[col*27 + (x*3+a)*3+b] = image[x][(i+a)*dim+(j+b)];
                    }
                }
            }
            ++col;
        }
    }
}

__host__ void recover_answer(double *answer, double **answer_output, int channel, int a, int b){
    for(int i=0; i<channel; ++i){
        for(int j=0; j<a; ++j){
            for(int k=0;k<b;++k){
                answer_output[i][j*b+k] = answer[(j*b+k)*channel+i];
            }
        }
    }
}

__global__ void GEMM(double *A, double *B, double *C,int M, int N, int K){
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int i = n/N;
    int j = n%N;
    C[i*N+j] = 0;
    for(int k=0; k<K; ++k)
        C[i*N+j] += A[i*K+k]*B[j+k*N];
}

int main(int argc, char *argv[]){
    int N,thread_block_size;
    N = atoi(argv[1]);//the highth or width of three channels picture
    thread_block_size = atoi(argv[2]);//每个block用到的线程数
    
    double *image[3];//输入的3 channel“图像”
    double *kernel_1[3];
    double *kernel_2[3];
    double *kernel_3[3];//3个kernel，每个kernel 3个channel，每个channel规模为3*3
    double *map1[3]; //stride = 1时，3种kernel的卷积结果
    double *map2[3]; //stride = 2时，3种kernel的卷积结果
    double *map3[3]; //stride = 3时，3种kernel的卷积结果
	//[3]：是3通道

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);//开始计时

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

    //kernel convert
    double *kernel_;
    cudaMallocManaged(&kernel_, 27*3*sizeof(double));//3个kernel
    double *kernel_temp[3];
    cudaMallocManaged(&kernel_temp[0], 27*sizeof(double));
    cudaMallocManaged(&kernel_temp[1], 27*sizeof(double));
    cudaMallocManaged(&kernel_temp[2], 27*sizeof(double));
    convertx(kernel_1, 3, 1, kernel_temp[0]);
    convertx(kernel_2, 3, 1, kernel_temp[1]);
    convertx(kernel_3, 3, 1, kernel_temp[2]);//先将每个kernel重新整合成一行，再将三个kernel重新组合排列按照要求
    
    for(int j=0; j<3; ++j){
        for(int i=0; i<27; ++i){
            kernel_[i*3+j] = kernel_temp[j][i];
        }
    }//转置
    for(int i=0;i<3;++i){
        cudaFree(kernel_temp[i]);
    }
    // PRINT_single(kernel_, 27, 3);

    
    int output_size[3];//用来表示三种stride的卷积结果的规模大小。

    //stride=1,进行卷积
	//计算outputsize，并给output分配空间，并根据outputsize设置blockSize，gridSize
    output_size[0] = (N-3)/1 + 1;
    CUDA_malloc(map1, output_size[0]);
    dim3 blockSize(thread_block_size);
    dim3 gridSize((output_size[0]*output_size[0]*3)/thread_block_size);
    
	//将input根据im2col进行转化，转化结果就是上面介绍的A
    double *matrix ;
    cudaMallocManaged(&matrix, output_size[0]*output_size[0]*27*sizeof(double));
    convertx(image, N, 1, matrix);
    // PRINT_single(matrix, output_size[0]*output_size[0],27);
	
	//给卷积结果分配空间，并调用GEMM进行计算，然后进行同步
    double *answer ;
    cudaMallocManaged(&answer, output_size[0]*output_size[0]*3*sizeof(double));
    GEMM<<<gridSize, blockSize>>>(matrix, kernel_, answer, output_size[0]*output_size[0], 3, 27);
    cudaDeviceSynchronize();
    // cout<<"answer1, when stride = 1 :"<<endl;
    // PRINT_single(answer, output_size[0]*output_size[0], 3);

    recover_answer(answer, map1, 3, output_size[0], output_size[0]);//将结果转化恢复

    //matrix = NULL;
    //answer = NULL;
    //cudaFree(answer);
    //cudaFree(matrix);//我觉得有可能是野指针的问题，最好到最后再free



    //判断stride=2的时候是否需要填充
    int padding = (N-3)%2 ;
    if (padding){//需要填充
        double *image_new[3];
        pad( image, N, image_new, padding);//扩展成一个新矩阵
        output_size[1] = (N+padding-3)/2 + 1;
        CUDA_malloc(map2, output_size[1]);//计算得到卷积结果的大小
        
        blockSize = thread_block_size;
        gridSize = (output_size[1]*output_size[1]*3)/thread_block_size;
        
        double *matrix2 ;//= NULL;
        cudaMallocManaged(&matrix2, output_size[1]*output_size[1]*27*sizeof(double));
        convertx(image_new, N+padding, 2, matrix2);
        
        double *answer2 ;//= NULL;
        cudaMallocManaged(&answer2, output_size[1]*output_size[1]*3*sizeof(double));
        GEMM<<<gridSize, blockSize>>>(matrix, kernel_, answer2, output_size[1]*output_size[1], 3, 27);
        cudaDeviceSynchronize();
        // cout<<"answer2, when stride = 2 :"<<endl;
        // PRINT_single(answer2, output_size[1]*output_size[1], 3);

        recover_answer(answer2, map2, 3, output_size[1], output_size[1]);

        for(int i=0;i<3;++i){
            cudaFree(image_new[i]);
        }
        
        // matrix2 = NULL;
        // answer2 = NULL;
        // cudaFree(answer2);
        // cudaFree(matrix2);
    }
    else{
        output_size[1] = (N-3)/2 + 1;
        CUDA_malloc(map2, output_size[1]);

        blockSize = thread_block_size;
        gridSize = (output_size[1]*output_size[1]*3)/thread_block_size;

        double *matrix2 ;//= NULL;
        cudaMallocManaged(&matrix2, output_size[1]*output_size[1]*27*sizeof(double));
        convertx(image, N, 2, matrix2);

        double *answer2 ;//= NULL;
        cudaMallocManaged(&answer2, output_size[1]*output_size[1]*3*sizeof(double));
        GEMM<<<gridSize, blockSize>>>(matrix2, kernel_, answer2, output_size[1]*output_size[1], 3, 27);
        cudaDeviceSynchronize();
        // cout<<"answer2, when stride = 2 :"<<endl;
        // PRINT_single(answer2, output_size[1]*output_size[1], 3);

        recover_answer(answer2, map2, 3, output_size[1], output_size[1]);
        
        // matrix2 = NULL;
        // answer2 = NULL;
        // cudaFree(matrix2);
        // cudaFree(answer2);

    }
    

    //判断stride=3的时候是否需要填充
    padding = (3 - (N-3)%3)%3;//如果除余的结果是1，就需要填充“一圈",增加两个size；如果是2，说明需要填充一维（）这里随机加在最上面和最左边
    if (padding){
        double *image_new2[3];      
        CUDA_malloc(image_new2, N+padding);
        pad( image, N, image_new2, padding);//bug1
        output_size[2] = (N+padding-3)/3 + 1;
        CUDA_malloc(map3, output_size[2]);

        blockSize = thread_block_size;
        gridSize = (output_size[2]*output_size[2]*3)/thread_block_size;

        double *matrix3;
        cudaMallocManaged(&matrix3, output_size[2]*output_size[2]*27*sizeof(double));
        convertx(image_new2, N+padding, 3, matrix3);//bug2
        
        double *answer3;
        cudaMallocManaged(&answer3, output_size[2]*output_size[2]*3*sizeof(double));
        GEMM<<<gridSize, blockSize>>>(matrix3, kernel_, answer3, output_size[2]*output_size[2], 3, 27);
        cudaDeviceSynchronize();
        // cout<<"answer3, when stride = 3 :"<<endl;
        // PRINT_single(answer3, output_size[2]*output_size[2], 3);

        recover_answer(answer3, map3, 3, output_size[2], output_size[2]);//bug3

        for(int i=0;i<3;++i){
            cudaFree(image_new2[i]);
        }
        // cudaFree(matrix3);
        // cudaFree(answer3);
    }
    else{
        output_size[2] = (N-3)/3 + 1;
        CUDA_malloc(map3, output_size[2]);

        blockSize = thread_block_size;
        gridSize = (output_size[2]*output_size[2]*3)/thread_block_size;

        double *matrix3;
        cudaMallocManaged(&matrix3, output_size[2]*output_size[2]*27*sizeof(double));
        convertx(image, N, 3, matrix3);

        double *answer3;
        cudaMallocManaged(&answer3, output_size[2]*output_size[2]*3*sizeof(double));
        GEMM<<<gridSize, blockSize>>>(matrix3, kernel_, answer3, output_size[2]*output_size[2], 3, 27);
        cudaDeviceSynchronize();
        // cout<<"answer3, when stride = 3 :"<<endl;
        // PRINT_single(answer3, output_size[2]*output_size[2], 3);

        recover_answer(answer3, map3, 3, output_size[2], output_size[2]);

        // cudaFree(matrix3);
        // cudaFree(answer3);
    }

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
    //  cout<<"kernel_1: "<<endl;
    //  PRINT(kernel_1, 3, 3);
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

    // matrix = NULL;
    // answer = NULL;
    // cudaFree(answer);
    // cudaFree(matrix);
    //matrix2 = NULL;
    //answer2 = NULL;
    //cudaFree(answer2);
    //cudaFree(matrix2);
    //matrix3 = NULL;
    //answer3 = NULL;
    //cudaFree(answer3);
    //cudaFree(matrix3);

}