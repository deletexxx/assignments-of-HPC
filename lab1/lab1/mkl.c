#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include "mkl.h"

void PRINT(double *A,int m,int n){
  for(int i=0;i<m;++i){
    for(int j=0;j<n;++j) printf("%lf ",A[i*n+j]);
    printf("\n");
  }
}

int main()
{
    double *A, *B, *C;//matrix
    int m, k, n;
    double alpha=1.0, beta=0.0;
    srand(time(NULL));
    time_t st,fin;

    scanf("%d %d %d",&m,&n,&k);

    // //allocate memory
    A = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    B = (double *)mkl_malloc( n*k*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    //if we dont't have memory, return 
    if (A == NULL || B == NULL || C == NULL) {
      printf( "ERROR: Can't allocate memory for matrices. \n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    //Intialize random matrix 
    for(int i=0;i<m*n;++i)     A[i]=(double)(rand()%50);

    for(int i=0;i<n*k;++i)     B[i]=(double)(rand()%50);

    for(int i=0;i<m*k;++i)     C[i]=0;

    //computer
    st=clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, alpha, A, n, B, k, beta, C, k);
    fin=clock();
    printf("time cost = %lf s\n",(double)(fin -st)/CLOCKS_PER_SEC);
    printf ("Computations completed.\n\n");

    //print
    // printf("A:\n");
    // PRINT(A,m,n);
    // printf("B:\n");
    // PRINT(B,n,k);
    // printf("C=A*B:\n");
    // PRINT(C,m,k); 

    //Deallocating memory
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    //printf (" Example completed. \n\n");
    return 0;
}
