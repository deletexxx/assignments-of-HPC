#include<iostream>
#include<vector>
#include<time.h>
using namespace std;

void PRINT(double * x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) cout<<x[i*m+j]<<" "; 
        cout<<endl;
    }
    cout<<endl;
}

void GEMM(double * A, double * B, double *C,int m, int n, int k){  
    double sumt;
    for(int i=0;i<k;++i){
        for(int j=0;j<m;++j){
            sumt=0;
            for(int x=0;x<n;++x) sumt+=A[j*m+x]*B[x*n+i];
            C[j*m+i]=sumt;
        }
    }
}

void matrix_multiply(double *A, double *B, double *C,int m,int n ,int k){
   
    //time 
    time_t st,fin;
    //computer
    st=clock();
    GEMM(A,B,C,m,n,k);       
    fin=clock();

    cout<<"time cost = "<<double(fin - st)/CLOCKS_PER_SEC<<" s"<<endl;

    //print
    // cout<<"A:"<<endl;
    // PRINT(A,m,n);
    // cout<<"B:"<<endl;
    // PRINT(B,n,k);
    // cout<<"C=A*B:"<<endl;
    // PRINT(C,m,k);
}
