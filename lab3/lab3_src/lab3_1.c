/* File:     lab3_1.c
 *
 * Purpose:  Use pthread to implement matrix multiplication
 *
 * Compile:  gcc -o lab3_1 lab3_1.c -lpthread 
 *
 * Run:      ./lab3_1 <num> <m> <k> <n>
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

int m,n,k,thread_num,partA;
struct bag{
        double *A_;
        double *B_;
        double *C_;
};
    
void Get_args(int argc,char *argv[]){
    thread_num = strtol(argv[1], NULL, 10);
    m = strtol(argv[2], NULL, 10);
    n = strtol(argv[3], NULL, 10);
    k = strtol(argv[4], NULL, 10);
    partA = m/thread_num;
}

void PRINT(double *x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) printf("%lf  ",x[i*n+j]); 
        printf("\n");
    }
    printf("\n");
}


void *computer(void *arg){
    struct bag *part;
    part=(struct bag*)arg;
    for(int i=0; i<partA; ++i){
        for(int j=0; j<k ;++j){
            int temp=0;
            for(int a=0;a<n;++a) temp+=part->A_[i*n+a]*part->B_[a*k+j];
            part->C_[i*n+j]=temp;
        }
    }
}

int main(int argc,char *argv[]){
    Get_args(argc,argv);
    
    //定义
    double *A,*B,*C;
    int starttime,finishtime;
    struct bag part[thread_num];

    A = (double*)malloc(sizeof(double)*m*n);
    B = (double*)malloc(sizeof(double)*n*k);
    C = (double*)malloc(sizeof(double)*m*k);

    
    //给矩阵赋值
    srand(time(NULL));
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j) A[i*n+j]=rand()%50;
    for(int i=0; i<n; ++i)
        for(int j=0; j<k; ++j) B[i*k+j]=rand()%50; 

    for(int i=0; i<thread_num; ++i){
        part[i].A_ = A+i*partA*n;
        part[i].B_ = B;
        part[i].C_ = C+i*partA*k;
    }

    starttime=clock();
    pthread_t pth[thread_num];

    for(int i=0 ;i<thread_num; ++i){
        pthread_create(&pth[i], NULL, computer, &part[i]);
    }

    if(partA * thread_num < m){
        for(int i=partA*thread_num ; i<m; ++i){
            for(int j=0; j<k; ++j){
                int temp=0;
                for(int a=0; a<n; ++a) temp+=A[i*n+a]*B[a*k+j];
                C[i*k+j]=temp;
            }
        }
    }

    for(int i=0; i<thread_num; ++i){
        pthread_join(pth[i],NULL);
    }

    finishtime=clock();

    printf("completed:\n");
    printf("the number of process is %d, the time cost = %lf s\n \n",thread_num,(double)(finishtime - starttime)/CLOCKS_PER_SEC);

    //print
        // printf("\n");
        // printf("A:\n");
        // PRINT(A,m,n);
        // printf("B:\n");
        // PRINT(B,n,k);
        // printf("C=A*B:\n");
        // PRINT(C,m,k);
    
    free(A);
    free(B);
    free(C);

    return 0;
    
}