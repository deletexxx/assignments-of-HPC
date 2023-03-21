/* File:     lab4_3_2.c
 *
 * Purpose:  Use pthread_for.so to implement replacement of serial code
 *
 * Compile:  gcc lab4_3_2.c -L. -lparallel_for -o lab4_3_2 -pthread
 *
 * Run:      ./lab4_3_2 <num> <n>
 *           num is the thread number that we will create
 *           n is size of array A,B,C
 *
 * Input:    none
 * Output:   the array A,B,C.
 */
#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include"parallel_for.h"

double *A, *B, *C, x = 10;
int thread_num,n;

void Get_args(int argc,char *argv[]){
    thread_num = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);
}

void * functor (void * args){
    struct for_index * index = (struct for_index *) args;
    for (int i = index->start; i < index->end; i = i + index->increment){
        A[i]=B[i] * x + C[i];
    }
}

int main(int argc,char *argv[]){
    Get_args(argc, argv);
    A = (double*)malloc(sizeof(double)*n);
    B = (double*)malloc(sizeof(double)*n);
    C = (double*)malloc(sizeof(double)*n);

    for(int i=0; i<n; ++i){
        C[i]=i;
        B[i]=i;
        A[i]=0;
    }

    parallel_for(0, n, 1, functor, NULL, thread_num);

    for(int i=0; i<n; ++i){
        printf(" a[%d]=%f, b[%d]=%f, c[%d]=%f\n",i,A[i],i,B[i],i,C[i]);
    }
    return 0;

}