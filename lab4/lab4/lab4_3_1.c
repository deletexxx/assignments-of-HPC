/* File:     lab4_3_1.c
 *
 * Purpose:  Use pthread_for to implement replacement of serial code
 *
 * Compile:  gcc -o lab4_3_1 lab4_3_1.c -lpthread 
 *
 * Run:      ./lab4_3_1 <num> <n>
 *           num is the thread number that we will create
 *           n is size of array A,B,C
 *
 * Input:    none
 * Output:   the array A,B,C.
 */

#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>

double *A, *B, *C, x = 10;
int thread_num,n;

void Get_args(int argc,char *argv[]){
    thread_num = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);
}

struct for_index {
    int start;
    int end;
    int increment;
};

void parallel_for(int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads){
    pthread_t pth[num_threads];
    int divid = (end - start + 1)/num_threads;
    struct for_index thread_assign[num_threads];
    for(int i=0; i<num_threads; ++i){
        thread_assign[i].start = i*divid ;
        if(i == (num_threads - 1) ){            
            thread_assign[i].end = end;
        }
        else{
            thread_assign[i].end = thread_assign[i].start + divid;
        }
        thread_assign[i].increment = increment;
    }

    for(int i=0; i<num_threads; ++i){
        pthread_create(&pth[i], NULL, functor, &thread_assign[i]);
    }

    for(int i=0; i<num_threads; ++i){
        pthread_join(pth[i], NULL);
    }

    printf("paraller finish\n");
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