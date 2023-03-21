/* File:     lab4_3_3_3_h.c
 *
 * Purpose:  Use pthread_for.so to implement matrix multiplication
 *
 * Compile:  gcc lab4_3_3_h.c -L. -lparallel_for -o lab4_3_3_h -pthread
 *
 * Run:      ./lab4_3_3_h <num> <m> <k> <n>
 *           num is the thread number that we will create
 *           m,k,n is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
 */

#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>
#include"parallel_for.h"

int m,n,k,thread_num;
double *A,*B,*C;

    
void Get_args(int argc,char *argv[]){
    thread_num = strtol(argv[1], NULL, 10);
    m = strtol(argv[2], NULL, 10);
    n = strtol(argv[3], NULL, 10);
    k = strtol(argv[4], NULL, 10);
}

void PRINT(double *x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) printf("%lf  ",x[i*n+j]); 
        printf("\n");
    }
    printf("\n");
}


void *computer(void *arg){
    struct for_index *thread_assign;
    thread_assign = (struct for_index*)arg;

    for(int i= thread_assign->start; i<thread_assign->end; i=i+thread_assign->increment){
        for(int j=0; j<k ;++j){
            int temp=0;
            for(int a=0;a<n;++a) temp+=A[i*n+a]*B[a*k+j];
            C[i*n+j] = temp;
        }
    }
}

int main(int argc,char *argv[]){
    Get_args(argc,argv);
    
    //定义
    int starttime,finishtime;

    A = (double*)malloc(sizeof(double)*m*n);
    B = (double*)malloc(sizeof(double)*n*k);
    C = (double*)malloc(sizeof(double)*m*k);

    
    //给矩阵赋值
    srand(time(NULL));
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j) A[i*n+j]=rand()%50;
    for(int i=0; i<n; ++i)
        for(int j=0; j<k; ++j) B[i*k+j]=rand()%50; 

    starttime=clock();
    parallel_for(0, n, 1, computer, NULL, thread_num);//computer the 0~n row of C
    finishtime=clock();

    printf("completed:\n");
    printf("the number of process is %d, the time cost = %lf s\n",thread_num,(double)(finishtime - starttime)/CLOCKS_PER_SEC);

    //print
        printf("\n");
        printf("A:\n");
        PRINT(A,m,n);
        printf("B:\n");
        PRINT(B,n,k);
        printf("C=A*B:\n");
        PRINT(C,m,k);
    
    free(A);
    free(B);
    free(C);

    return 0;
    
}