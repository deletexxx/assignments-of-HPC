#ifndef _PARALLEL_FOR_
#define _PARALLEL_FOR_

#include<pthread.h>

struct for_index {
    int start;
    int end;
    int increment;
};

void parallel_for(int start, int end, int increment, void *(*functor)(void*), void *arg , int num_threads);

#endif
