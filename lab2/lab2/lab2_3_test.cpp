/* File:     lab2_3_test.c
 *
 * Purpose:  test the use of head file---"matrix_multiply.h"
 * 
 * Generate the so file: gcc lab2_3_vector.cpp lab2_3_vectorpoint.cpp lab2_3_doublepoint.cpp --shared -fPIC -o libmatrixmult.so
 *
 * Compile:  gcc lab2_3_test.cpp -L. -lmatrixmult -o test
 * 
 * Change the location of the so file, you can find the .so file:  sudo cp libmatrixmult.so /usr/lib/
 *
 * Run:      ./test  <m> <n> <k>
 *           m,n,k is size of matrix A,B,C
 *
 * Input:    none
 * Output:   the cost time. If you want ,you can also print matrix A,B,C.
 */

#include<iostream>
#include<vector>
#include<time.h>
#include"matrix_multiply.h"
using namespace std;

int m,n,k;
void Get_args(int argc, char* argv[]) {
   m=strtol(argv[1], NULL, 10);
   n=strtol(argv[2], NULL, 10);
   k = strtoll(argv[3], NULL, 10);
} 

int main(int argc, char* argv[]){
    Get_args(argc, argv);

    //vector
    srand(time(NULL));
    vector<vector<double>> A,B,C;
    A=vector<vector<double>>(m,vector<double>(n,0));
    B=vector<vector<double>>(n,vector<double>(k,0));
    C=vector<vector<double>>(m,vector<double>(k,0));

    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j) A[i][j]=rand()%50;
    for(int i=0;i<n;++i)
        for(int j=0;j<k;++j) B[i][j]=rand()%50;

    C=matrix_multiply(A,B,m,n,k);
    
    //vectorpoint
    vector<vector<double>> *A2,*B2,*C2;
    vector<vector<double>> A_=vector<vector<double>>(m,vector<double>(n,0));
    vector<vector<double>> B_=vector<vector<double>>(n,vector<double>(k,0));
    vector<vector<double>> C_=vector<vector<double>>(m,vector<double>(k,0));
    A2=&A_,B2=&B_,C2=&C_;

    srand(time(NULL));
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)    A2->at(i)[j]=rand()%50;    
    for(int i=0;i<n;++i)
        for(int j=0;j<k;++j)    B2->at(i)[j]=rand()%50;

    matrix_multiply(A2,B2,C2,m,n,k);


    //double point
    double *A3,*B3,*C3;
    A3=(double*)malloc(m*n*sizeof(double));
    B3=(double*)malloc(n*k*sizeof(double));
    C3=(double*)malloc(m*k*sizeof(double));

    srand(time(NULL));
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)    A3[i*m+j]=rand()%50;      
    for(int i=0;i<n;++i)
        for(int j=0;j<k;++j)    B3[i*n+j]=rand()%50;

    matrix_multiply(A3,B3,C3,m,n,k);

    free(A3);
    free(B3);
    free(C3);

    return 0;
}
