/* File:     lab3_3.c
 *
 * Purpose:  Use multithreading to calculate the solution of a quadratic equation in one variable step by step.
 *
 * Compile:   gcc -o lab3_3 lab3_3.c -lpthread -lm
 *
 * Run:      ./lab3_3 <a> <b> <c>
 *           a,b,c is constant variables of the equation.
 *
 * Input:    none
 * Output:   Two solutions of the equation.
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<pthread.h>

int two_to_three=0;
int onethree_to_four=0;
double a,b,c;
double ans[5];
pthread_mutex_t first,second;

void *step1(){
    ans[0]=(-b)/(2*a);
    pthread_mutex_lock(&second);
    onethree_to_four++;
    pthread_mutex_unlock(&second);
}

void *step2(){
    ans[1] = sqrt(b*b - 4*a*c);

    pthread_mutex_lock(&first);
    two_to_three++;
    pthread_mutex_unlock(&first);
}

void *step3(){
    while(two_to_three != 1);
    ans[2] = ans[1]/(2*a);

    pthread_mutex_lock(&second);
    onethree_to_four++;
    pthread_mutex_unlock(&second);
}

void *step4(){
    while(onethree_to_four != 2);
    ans[3] = ans[0] + ans[2];
    ans[4] = ans[0] - ans[2];
}

void Get_args(int argc,char *argv[]){
    a = strtol(argv[1], NULL, 10);
    b = strtol(argv[2], NULL, 10);
    c = strtol(argv[3], NULL, 10);
}

int main(int argc,char *argv[]){
    Get_args(argc, argv);
    if((b*b - 4*a*c)<0){
        printf("There is no solution to this equation.\n");
        return 0;
    }

    pthread_t pth[4];

    pthread_mutex_init(&first, NULL);
    pthread_mutex_init(&second, NULL);

    pthread_create(&pth[0], NULL, step1, NULL);
    pthread_create(&pth[1], NULL, step2, NULL);
    pthread_create(&pth[2], NULL, step3, NULL);
    pthread_create(&pth[3], NULL, step4, NULL);

    for(int i=0; i<4; ++i){
        pthread_join(pth[i], NULL);
    }

    printf("when a = %lf, b = %lf, c = %lf, x1 = %lf and x2 = %lf\n",a,b,c,ans[3],ans[4]);
    return 0;
}