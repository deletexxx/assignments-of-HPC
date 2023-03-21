/* File:     lab2_2_1.c
 *
 * Purpose:  Use collective communication of mpi to implement matrix multiplication
 *
 * Compile:  mpicc -o lab2_2_1 lab2_2_1.c
 *
 * Run:      mpiexec -np 4 ./lab2_2_1 <m> <k> <n>
 *           m,k,n is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
 */
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<mpi.h>

int m,n,k;
void Get_args(int argc, char* argv[]) {
   //thread_count = strtol(argv[1], NULL, 10);
   //process_num=strtol(argv[1], NULL, 10);
   m=strtol(argv[1], NULL, 10);
   k=strtol(argv[2], NULL, 10);
   n = strtoll(argv[3], NULL, 10);
} 

void PRINT(double *x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) printf("%lf  ",x[i*n+j]); 
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]){
    Get_args(argc,argv);

    int rank,numprocess,block_line;
    MPI_Init(0,0);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocess);

    block_line=m/numprocess;
    int strtime,fintime;
    double *A,*B,*C,*send_A,*recv_C;
    A=(double*)malloc(m*k*sizeof(double));
    B=(double*)malloc(k*n*sizeof(double));
    C=(double*)malloc(m*n*sizeof(double));
    send_A=(double*)malloc(block_line*k*sizeof(double));
    recv_C=(double*)malloc(block_line*n*sizeof(double));

    if(rank==0){
        srand(time(NULL));
        strtime=clock();

        for(int i=0;i<m;++i)
            for(int j=0;j<k;++j) A[i*k+j]=rand()%50;
        for(int i=0;i<k;++i)
            for(int j=0;j<n;++j) B[i*n+j]=rand()%50;
    }
    //将A散播
    MPI_Scatter(A, block_line*k, MPI_DOUBLE, send_A, block_line*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //将B广播
    MPI_Bcast(B,k*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    //computer
    int x=0;
    for(int l=rank*block_line; l<(rank+1)*block_line; ++l){
        for(int j=0; j<n; ++j){
            double temp=0;
            for(int i=0; i<k; ++i) temp+=send_A[(l-rank*block_line)*k+i]*B[i*n+j];
            recv_C[x++]=temp;
        }
    }

    MPI_Gather(recv_C, block_line*n, MPI_DOUBLE, C, block_line*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank==0){
        //if we are unable to divide the number of rows of matrix A evenly according to numprocess
        if((m/numprocess)!=0){
            for(int l=numprocess*block_line-1; l<m; ++l){
                for(int j=0; j<n; ++j){
                    double temp=0;
                    for(int i=0; i<k; ++i) temp+=A[l*k+i]*B[i*n+j];
                    C[l*n+j]=temp;
                }
            }
        }

        fintime=clock();
        printf("completed:\n");
        printf("the number of process is %d, the time cost = %lf s\n \n",numprocess,(double)(fintime - strtime)/CLOCKS_PER_SEC);
        
        //print
        // printf("\n");
        // printf("A:\n");
        // PRINT(A,m,k);
        // printf("B:\n");
        // PRINT(B,k,n);
        // printf("C=A*B:\n");
        // PRINT(C,m,n);
    }
    free(A);
    free(B);
    free(C);
    free(send_A);
    free(recv_C);

    MPI_Finalize();
    return 0;
    
}