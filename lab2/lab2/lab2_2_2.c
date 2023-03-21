/* File:     lab2_2_2.c
 *
 * Purpose:  Use point-to-point communication of mpi and creat a new datatype to implement matrix multiplication
 *
 * Compile:  mpicc -o lab2_2_2 lab2_2_2.c
 *
 * Run:      mpiexec -np 4 ./lab2_2_2 <m> <k> <n>
 *           m,k,n is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
 */
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<mpi.h>

int m,n,k,process_num;
void Get_args(int argc, char* argv[]) {
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

int main(int argc, char *argv[]){
    Get_args(argc,argv);
    
    int rank, numprocess, block_line;

    MPI_Init(&argc, &argv);
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
        // MPI_Address(&tempvar.A_[0],&var_displace[0]);//第一个double型相对于MPI_BOTTOM的偏移
        // MPI_Address(&tempvar.B[0],&var_displace[1]);//第2个double型相对于MPI_BOTTOM的偏移
        // var_displace[1]=var_displace[1]-var_displace[0];
        // var_displace[0]=0;

    MPI_Type_create_struct(var_count, var_everycount, var_displace, var_type, &mytype);
    MPI_Type_commit(&mytype);
    double *recv_C;
    recv_C=(double*)malloc(block_line*n*sizeof(double));

    if(rank==0){
        double A[m*k],B[k*n],C[m*n];
        struct var tempvar;

        int strtime,fintime;
        strtime=clock();

        srand(time(NULL));
        for(int i=0;i<m;++i)
            for(int j=0;j<k;++j) A[i*k+j]=rand()%50;
        for(int i=0;i<k;++i)
            for(int j=0;j<n;++j) B[i*n+j]=rand()%50;

        struct var ABC[numprocess];
        for(int i=0;i<numprocess;++i){
            for(int j=0;j<block_line*k;++j){
                ABC[i].A_[j]=A[i*block_line*k+j];
            }
            for(int j=0;j<n*k;++j){
                ABC[i].B[j] = B[j];
            }
        }

        for(int i=1; i<numprocess; ++i){
            MPI_Send(&ABC[i], 1, mytype, i, 0, MPI_COMM_WORLD);
        }
        

        for(int l=rank*block_line; l<(rank+1)*block_line; ++l){
            for(int j=0; j<n; ++j){
                double temp=0;
                for(int i=0; i<k; ++i) temp+=ABC[0].A_[(l-block_line*rank)*k+i]*ABC[0].B[i*n+j];
                C[l*n+j]=temp;
            }
        }

        if((m/numprocess)!=0){
            for(int l=numprocess*block_line-1; l<m; ++l){
                for(int j=0; j<n; ++j){
                    double temp=0;
                    for(int i=0; i<k; ++i) temp+=A[l*k+i]*B[i*n+j];
                    C[l*n+j]=temp;
                }
            }
        }

        //接收计算结果
        for(int i=1; i<numprocess; ++i){
            MPI_Recv(recv_C, block_line*n, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int x=i*block_line; x<(i+1)*block_line; ++x)
                for(int y=0; y<n; ++y)    C[x*n+y]=recv_C[(x-i*block_line)*n+y];
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
    else{
        struct var tempvar;
        MPI_Recv(&tempvar, 1, mytype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int x=0;
        for(int l=rank*block_line; l<(rank+1)*block_line; ++l){
            for(int j=0; j<n; ++j){
                double temp=0;
                for(int i=0; i<k; ++i) temp+=tempvar.A_[(l-rank*block_line)*k+i]*tempvar.B[i*n+j];
                recv_C[x++]=temp;
            }
        }
        
        MPI_Send(recv_C, block_line*n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    

    MPI_Finalize();
    return 0;
}