#include<iostream>
#include<vector>
#include<time.h>
using namespace std;

void PRINT(vector<vector<double>> *x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) cout<<x->at(i)[j]<<" "; 
        cout<<endl;
    }
    cout<<endl;
}

void GEMM(vector<vector<double>> *A, vector<vector<double>> *B,vector<vector<double>> *C, int m, int n, int k){  
    double sumt;
    for(int i=0;i<k;++i){
        for(int j=0;j<m;++j){
            sumt=0;
            for(int x=0;x<n;++x) sumt+=A->at(j)[x]*B->at(x)[i];
            C->at(j)[i]=sumt;
        }
    }
}

void matrix_multiply(vector<vector<double>> *A,vector<vector<double>> *B,vector<vector<double>> *C, int m,int n ,int k){
   
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