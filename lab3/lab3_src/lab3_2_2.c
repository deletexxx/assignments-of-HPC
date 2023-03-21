/* File:     lab3_2_2.c
 *
 * Purpose:  Use pthread to calculate the sum of all values in array a
 *
 * Compile:  gcc -o lab3_2_2 lab3_2_2.c -lpthread 
 *
 * Run:      ./lab3_2_2 <num> 
 *           num is the thread number that we will create
 *
 * Input:    none
 * Output:   A single thread and the sum of all threads
 */
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>

int a[1000];
int global_index=0;
int n;//thread_num
pthread_mutex_t mutex;

void *computer(void *arg){
    int *sum_;
    sum_= (int*)arg;
    int temp_index;
    while(global_index < 1000){
        pthread_mutex_lock(&mutex);
            temp_index = global_index;
            global_index+=10;
        pthread_mutex_unlock(&mutex);
             int delay = 0xffff;
             while (delay) --delay;
        for(int i=0; i<10 && temp_index+i<1000; ++i){
            (*sum_)+=a[temp_index+i];
        }
    }
    printf("the sum in a thread: %d\n",*sum_);
}

int main(int argc,char *argv[]){
    n = strtol(argv[1], NULL, 10);

    srand(time(NULL));
    for(int i=0; i<1000; ++i){
        //a[i] = rand()%50;
        a[i]=i;
    }

    pthread_t pth[n];
    pthread_mutex_init(&mutex, NULL);
    
    int x[n];
    for(int i=0; i<n ;++i){
        x[i] = 0;
    }
    for(int i=0;i<n;++i){
        pthread_create(&pth[i], NULL, computer, &x[i]);
    }
    int sum = 0;
    void *temp;
    for(int i=0; i<n; ++i){
        pthread_join(pth[i], NULL);
        sum+=x[i];
    }
    printf("Sum of a[1000] = %d\n",sum);
    return 0;
}