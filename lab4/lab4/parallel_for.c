/* File:     parallel_for.c
 *
 * Purpose:  Compile the parallel_for function into a .so file, which is called by other programs
 *
 * Compile:  gcc parallel_for.c --shared -fPIC -o libparallel_for.so
 * 
 * Move the libparallel_for.so file into the folder /usr/lib/ where the general dynamic link library is located:
 *           sudo cp libparallel_for.so /usr/lib/  
 *
 * Input:    none
 * Output:   none , a file--libparallel_for.so
 */
#include<stdio.h>
#include"parallel_for.h"

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