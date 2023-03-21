/* File:     lab2_2_3.c
 *
 * Purpose:  Use collective communication and creat a new datatype of mpi to implement matrix multiplication
 *
 * Compile:  mpicc -o lab2_2_3 lab2_2_3.c
 *
 * Run:      mpiexec -np 4 ./lab2_2_3 <m> <k> <n>
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

int main(int argc,char *argv[]){
    Get_args(argc,argv);

    int rank,numprocess,block_line;
    MPI_Init(0,0);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocess);

    block_line=m/numprocess;
    struct var{
        double A_[block_line*k];
        double B[k*n];
    };

    int var_count=2;
    int var_everycount[2]={block_line*k, k*n};
    MPI_Aint var_displace[2]={0, 8*block_line*k};
    MPI_Datatype var_type[2]={MPI_DOUBLE,MPI_DOUBLE};
    MPI_Datatype mytype;       
    MPI_Type_create_struct(var_count, var_everycount, var_displace, var_type, &mytype);
    MPI_Type_commit(&mytype);

    double *recv_C;
    int strtime,fintime;
    recv_C=(double*)malloc(block_line*n*sizeof(double));

    struct var ABC[numprocess];
    double A[m*k],B[k*n],C[m*n];

    if(rank==0){
        strtime=clock();

        srand(time(NULL));
        for(int i=0;i<m;++i)
            for(int j=0;j<k;++j) A[i*k+j]=rand()%50;
        for(int i=0;i<k;++i)
            for(int j=0;j<n;++j) B[i*n+j]=rand()%50;

        for(int i=0;i<numprocess;++i){
            for(int j=0;j<block_line*k;++j){
                ABC[i].A_[j]=A[i*block_line*k+j];
            }
            for(int j=0;j<n*k;++j){
                ABC[i].B[j] = B[j];
            }
        }
    }

    struct var tempvar;
    //将A散播
    MPI_Scatter(&ABC, 1, mytype, &tempvar, 1, mytype, 0, MPI_COMM_WORLD);
    //computer
    int x=0;
    for(int l=rank*block_line; l<(rank+1)*block_line; ++l){
        for(int j=0; j<n; ++j){
            double temp=0;
            for(int i=0; i<k; ++i) temp+=tempvar.A_[(l-rank*block_line)*k+i]*tempvar.B[i*n+j];
            recv_C[x++]=temp;
        }
    }

    MPI_Gather(recv_C, block_line*n, MPI_DOUBLE, C, block_line*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank==0){
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
        
        // //print
        // printf("\n");
        // printf("A:\n");
        // PRINT(A,m,k);
        // printf("B:\n");
        // PRINT(B,k,n);
        // printf("C=A*B:\n");
        // PRINT(C,m,n);
    }

    MPI_Finalize();
    return 0;
    
}