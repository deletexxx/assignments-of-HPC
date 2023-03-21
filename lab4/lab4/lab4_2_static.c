/* File:     lab4_2_static.c
 *
 * Purpose:  Use the static scheduling method to optimize 
 *           the general matrix multiplication implemented by openmp
 *
 * Compile:  gcc -o lab4_2_static lab4_2_static.c -fopenmp
 *
 * Run:      ./lab4_2_static <num> <m> <k> <n>
 *           num is the thread number that we will create
 *           m,k,n is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
 */

#include<stdio.h>
#include<omp.h>
#include<time.h>
#include<stdlib.h>
int thread_num, m, n, k;

void PRINT(double *x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) printf("%lf  ",x[i*n+j]); 
        printf("\n");
    }
    printf("\n");
}

void Get_args(int argc,char *argv[]){
    //thread_num = strtol(argv[1], NULL, 10); //the first  omp will automatically set the first as the number of threads
    m = strtol(argv[2], NULL, 10);
    n = strtol(argv[3], NULL, 10);
    k = strtol(argv[4], NULL, 10);
}

int main(int argc, char *argv[]){
    Get_args(argc, argv);
    
    double temp_c,start_time,finish_time;
    double *A, *B, *C;
    A = (double*)malloc(sizeof(double)*m*n);
    B = (double*)malloc(sizeof(double)*n*k);
    C = (double*)malloc(sizeof(double)*m*k);

    srand(time(NULL));
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j) A[i*n+j] = rand()%50;
    for(int i=0; i<n; ++i)
        for(int j=0; j<k; ++j) B[i*k+j] = rand()%50;

    start_time = clock();
    #pragma omp parallel for private(temp_c) shared(A, B, C) schedule(static)
        for(int i=0; i<m; ++i){
            for(int j=0; j<k; ++j){
                temp_c = 0;
                #pragma omp parallel for firstprivate(i,j) shared(A, B, C) reduction(+:temp_c) schedule(static)
                for(int x=0; x<n; ++x) temp_c+=A[i*n+x]*B[x*k+j];
                
                #pragma omp critical
                {
                    C[i*n+j] = temp_c;
                }
            }
        }
    
    finish_time = clock();

    printf("completed:\n");
    printf("the number of thread is %d, the time cost = %lf s\n \n",omp_get_num_threads(),(double)(finish_time - start_time)/CLOCKS_PER_SEC);

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

    
}