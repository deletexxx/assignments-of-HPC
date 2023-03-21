/* File:     5.cu
 *
 * Purpose:  Use the convolution method provided by cuDNN to perform convolution operations.
 *
 * Compile:  nvcc -o 5 5.cu -lcudnn -I /opt/conda/include/
 *           In order to successfully use the `-lcudnn` command (dynamic link library), you need to put the .so file 
 *               in the /usr/lib directory `cp /opt/conda/lib/libcudnn.so /usr/lib` (one method).
 *           The dynamic link library directory of the school’s cluster: /opt/conda/lib/, the header file directory /opt/conda/include/
 *
 * Run:      ./5 <N> <stride> <padding>
 *           N is size of input matrix
 *           stride is the number of steps in each sliding step of the convolution
 *           padding is the number of rows/columns that need to be filled with 0 in the matrix
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print the intput, the kernel, and the output of convolution result
 */
 
#include<iostream>
#include<cstdlib>
#include<cudnn.h>
using namespace std;
//该宏检查此状态对象是否存在任何错误条件，并在出现问题时中止程序的执行。然后，我们可以简单地包装我们用该宏调用的任何库函数：cudnnCreatecudnnStatus_t
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

__host__ void PRINT(float *matrix, int channel, int h, int w){
    int x = 0;
    for(int i=0;i<h;++i){
        for(int j=0;j<w;++j){
            for(int k=0;k<channel;++k){
                cout<<matrix[x++]<<" ";
            }
            cout<<endl;
        }
    }
    cout<<endl;
}

int main(int argc, char *argv[]){
    int N,stride,padding;
    N = atoi(argv[1]);
    stride = atoi(argv[2]);
    padding = atoi(argv[3]);

    //handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int batch = 1;
    int channels = 3;
    //input
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
        /*format*/CUDNN_TENSOR_NHWC,// NHWC/NCWH
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batchsize*/batch,
        /*channels=*/channels,
        /*height=*/N,
        /*width=*/N));

    float *input;
    cudaMallocManaged( &input, batch*channels*N*N*sizeof(float) ) ;
    for(int channel=0; channel<channels;++channel){
        for(int h=0;h<N;++h){
            for(int w=0;w<N;++w) {
                input[channel*N*N+h*N+w] = rand()%50;
            }
        }
    }

    //kernel
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                                /*out_channels=*/3,
                                /*in_channels=*/3,
                                /*kernel_height=*/3,
                                /*kernel_width=*/3));


    //描述卷积内核
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                               /*pad_height=*/padding,
                                               /*pad_width=*/padding,
                                               /*vertical_stride=*/stride,
                                               /*horizontal_stride=*/stride,
                                               /*dilation_height=*/1,
                                               /*dilation_width=*/1,
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));

    float *kernel_;// NCHW
    cudaMallocManaged( &kernel_, 3*3*3*3*sizeof(float) );
    int a = -1,b =1;
    for (int kernel = 0; kernel < 3; ++kernel) {
       for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    kernel_[((kernel*3+channel)*3+row)*3+column] = rand()%(b-a+1) + a;
                }
            }
        }
    }
    
    //计算输出的大小
    int out_n, out_c, out_h, out_w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convolution_descriptor, input_descriptor, kernel_descriptor,
        &out_n, &out_c, &out_h, &out_w));
    //cout<<"the output: "<<out_n<<" "<<out_c<<" "<<out_h<<" "<<out_w<<endl;

    //output
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
        /*format*/CUDNN_TENSOR_NHWC,// NHWC/NCWH
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batchsize*/out_n,
        /*channels=*/out_c,
        /*height=*/out_h,
        /*width=*/out_w));

    
        float *output;
        cudaMallocManaged( &output, out_n*out_c*out_h*out_w*sizeof(float) );

    // 卷积算法的描述
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
        cudnnGetConvolutionForwardAlgorithm(cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT（在内存受限的情况下，memoryLimitInBytes 设置非 0 值）
        /*memoryLimitInBytes=*/0,
        &convolution_algorithm));



// 计算 cuDNN 它的操作需要多少内存
size_t workspace_bytes{ 0 };
checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
    input_descriptor,
    kernel_descriptor,
    convolution_descriptor,
    output_descriptor,
    convolution_algorithm,
    &workspace_bytes));

    // 分配内存， 从 cudnnGetConvolutionForwardWorkspaceSize 计算而得
    void* d_workspace{ nullptr };
    cudaMallocManaged(&d_workspace, workspace_bytes);



    

    const float alpha = 1.0f, beta = 0.0f;
 
    // 进行卷积操作 
    checkCUDNN(cudnnConvolutionForward(cudnn,
        &alpha,
        input_descriptor,
        input,
        kernel_descriptor,
        kernel_,
        convolution_descriptor,
        convolution_algorithm,
        d_workspace, // 注意，如果我们选择不需要额外内存的卷积算法，d_workspace可以为nullptr。
        workspace_bytes,
        &beta,
        output_descriptor,
        output));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout<<"所用时间为: "<< elapsedTime<<" <ms>"<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // cout<<"kernel:"<<endl;
    // PRINT(kernel_,3*3,3,3);
    // cout<<"input:"<<endl;
    // PRINT(input,3,N,N);
    // cout<<"output:"<<endl;
    // PRINT(output,out_c,out_h,out_w);

    cudaFree(d_workspace);
    cudaFree(input);
    cudaFree(output);
    cudaFree(kernel_);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);

    cudnnDestroy(cudnn);

}