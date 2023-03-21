/* File:     lab3_3.c
 *
 * Purpose:  利用Monte-carlo方法,估算y=x^2曲线与x轴之间区域的面积，其中x的范围为[0,1]。
 *
 * Compile:   gcc -o lab3_4 lab3_4.c -lpthread
 *
 * Run:      ./lab3_4 <n> <thread_num>
 *           n is the total number of throws
 *           thread_num is the number of thread that we will create;
 *
 * Input:    none
 * Output:   the number of points, The number of points in the shadow.
 *           Estimated area
 */
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<pthread.h>

int n,thread_num,part,shadowpoint;
pthread_mutex_t mutex;
double *x,*y;

void Get_args(int argc,char *argv[]){
    thread_num = strtol(argv[1], NULL, 10);
    n= strtol(argv[2], NULL, 10);
    part = n/thread_num;
}

void *judge(void *arg){
    int *i;
    i = (int*)arg;
    for(int j=0; j<part ; ++j){
        if(y[(*i)+j]<=x[(*i)+j]*x[(*i)+j]){
            pthread_mutex_lock(&mutex);
            shadowpoint++;
            pthread_mutex_unlock(&mutex);
        }
    }
}

int main(int argc, char *argv[]){
    Get_args(argc,argv);
    pthread_t pth[thread_num];
    pthread_mutex_init(&mutex, NULL);

    x = (double*)malloc(n);
    y = (double*)malloc(n);

    srand(time(NULL));
    for(int i=0;i<n;++i){
        x[i]=((double)rand())/RAND_MAX;
        y[i]=((double)rand())/RAND_MAX;
    }
    
    int temp[thread_num];
    for(int i=0; i<thread_num; ++i){
        temp[i]=i*part;
        pthread_create(&pth[i], NULL, judge, &temp[i]);
    }

    for(int i=0; i<thread_num;++i){
        pthread_join(pth[i],NULL);
    }
    printf("Number of points = %d, The number of points in the shadow = %d\n",thread_num*part,shadowpoint);

    double area=(double)shadowpoint/(thread_num*part);
    printf("Estimated area = %lf\n",area);
    return 0;
}