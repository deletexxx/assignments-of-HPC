#include<stdio.h>
#include <iostream>
#include<math.h>
#include<mpi.h>
#include<time.h>
using namespace std;

#define random(x)(rand()%50)

int main(int argv, char *argc[]){
    int size;
    if (argv == 2){
    //判断有无矩阵大小size接收，无则初始化为5
    size = atoi(argc[1]);
    }
    else { size = 5; }
    srand((int)time(0));
    int *a, *b, *c, *pce, *ans;
    int rank, numprocess, line;
    double starttime, endtime;
    MPI_Init(&argv, &argc);//MPI Initialize
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//获得当前进程号
    MPI_Comm_size(MPI_COMM_WORLD, &numprocess);//获得进程个数
    line = size / numprocess;//将数据分为(进程数)个块,主进程也要处理数据
    a = (int*)malloc(sizeof(int)*size*size);
    b = (int*)malloc(sizeof(int)*size*size);
    c = (int*)malloc(sizeof(int)*size*size);
    pce = (int*)malloc(sizeof(int)*size*line);
    ans = (int*)malloc(sizeof(int)*size*line);

    starttime = MPI_Wtime();
    if (rank==0){
        for (int i = 0; i<size; i++){
            for (int j = 0; j<size; j++){
            //随机两个相乘的矩阵a，b
            a[i*size + j] = random(10);
            b[i*size + j] = random(10);
            }
        }
        for (int i = 1; i<numprocess; i++)
        {
        //将b矩阵传递给各进程
            MPI_Send(b, size*size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 1; i<numprocess; i++){
        //将a矩阵分成块矩阵，传递给各进程
            MPI_Send(a + (i - 1)*line*size, size*line, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
        for (int k = 1; k<numprocess; k++)
        {
            MPI_Recv(ans, line*size, MPI_INT, k, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //接收传递结果到数组c
            for (int i = 0; i<line; i++)
            {
                for (int j = 0; j<size; j++)
                {
                    c[((k - 1)*line + i)*size + j] = ans[i*size + j];
                }
            }
        }
        for (int i = (numprocess - 1)*line; i<size; i++)
        {
            //计算分配给主进程的块矩阵
            for (int j = 0; j<size; j++)
            {
                int temp = 0;
                for (int k = 0; k<size; k++){
                    temp += a[i*size + k] * b[k*size + j];}
                c[i*size + j] = temp;
            }
        }
        // for (int i = 0; i < size; i++)
        // {
        //     //打印显示
        //     for (int j = 0; j<size; j++) { printf("%d ", a[size*i + j]); }printf("  ");
        //     for (int j = 0; j<size; j++) { printf("%d ", b[size*i + j]); }printf("  ");
        //     for (int j = 0; j<size; j++) { printf("%d ", c[size*i + j]); }printf("\n");
        // }
        endtime = MPI_Wtime();
        printf("Took %f secodes.\n", endtime - starttime);
    }

    else{
        MPI_Recv(b, size*size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //接收b矩阵
        MPI_Recv(pce, size*line, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //接收块矩阵
        for (int i = 0; i<line; i++){
            //计算块矩阵与b矩阵相乘的结果
            for (int j = 0; j<size; j++){
                int temp = 0;
                for (int k = 0; k<size; k++)
                    temp += pce[i*size + k] * b[k*size + j];
                ans[i*size + j] = temp;
            }
        }
        MPI_Send(ans, line*size, MPI_INT, 0, 2, MPI_COMM_WORLD);
        //将结果传递到主进程
    }
    MPI_Finalize();
    return 0;
}