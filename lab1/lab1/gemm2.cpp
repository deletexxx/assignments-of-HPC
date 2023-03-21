
//计算数值在50以内的随机数生成的矩阵，并进行矩阵乘法
//A，B是生成的两个矩阵，C=A*B

#include<iostream>
#include<vector>
#include<time.h>
using namespace std;

//输出矩阵
void PRINT(vector<vector<double>> x, int m, int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) cout<<x[i][j]<<" "; 
        cout<<endl;
    }
    cout<<endl;
}

//常规进行矩阵乘法
vector<vector<double>> GEMM(vector<vector<double>> A, vector<vector<double>> B, int M, int N, int K){  
    vector<vector<double>> C(M,vector<double>(K,0));
    for(int m=0;m<M;m+=4){
        for(int k=0;k<K;k+=4){
            for(int n=0;n<N;n++){
                //first 
                C[m][k+0]+=A[m][n]*B[n][k+0], C[m][k+1]+=A[m][n]*B[n][k+1];
                C[m][k+2]+=A[m][n]*B[n][k+2], C[m][k+3]+=A[m][n]*B[n][k+3];

                /*
                //second
                C[m+0][k+0]+=A[m+0][n]*B[n][k+0], C[m+0][k+1]+=A[m+0][n]*B[n][k+1];
                C[m+0][k+2]+=A[m+0][n]*B[n][k+2], C[m+0][k+3]+=A[m+0][n]*B[n][k+3];
                C[m+1][k+0]+=A[m+1][n]*B[n][k+0], C[m+1][k+1]+=A[m+1][n]*B[n][k+1];
                C[m+1][k+2]+=A[m+1][n]*B[n][k+2], C[m+1][k+3]+=A[m+1][n]*B[n][k+3];
                C[m+2][k+0]+=A[m+2][n]*B[n][k+0], C[m+2][k+1]+=A[m+2][n]*B[n][k+1];
                C[m+2][k+2]+=A[m+2][n]*B[n][k+2], C[m+2][k+3]+=A[m+2][n]*B[n][k+3];
                C[m+3][k+0]+=A[m+3][n]*B[n][k+0], C[m+3][k+1]+=A[m+3][n]*B[n][k+1];
                C[m+3][k+2]+=A[m+3][n]*B[n][k+2], C[m+3][k+3]+=A[m+3][n]*B[n][k+3];     
                
                //end
                C[m+0][k+0]+=A[m+0][n+0]*B[n+0][k+0], C[m+0][k+1]+=A[m+0][n+0]*B[n+0][k+1];
                C[m+0][k+2]+=A[m+0][n+0]*B[n+0][k+2], C[m+0][k+3]+=A[m+0][n+0]*B[n+0][k+3];
                C[m+1][k+0]+=A[m+1][n+0]*B[n+0][k+0], C[m+1][k+1]+=A[m+1][n+0]*B[n+0][k+1];
                C[m+1][k+2]+=A[m+1][n+0]*B[n+0][k+2], C[m+1][k+3]+=A[m+1][n+0]*B[n+0][k+3];
                C[m+2][k+0]+=A[m+2][n+0]*B[n+0][k+0], C[m+2][k+1]+=A[m+2][n+0]*B[n+0][k+1];
                C[m+2][k+2]+=A[m+2][n+0]*B[n+0][k+2], C[m+2][k+3]+=A[m+2][n+0]*B[n+0][k+3];
                C[m+3][k+0]+=A[m+3][n+0]*B[n+0][k+0], C[m+3][k+1]+=A[m+3][n+0]*B[n+0][k+1];
                C[m+3][k+2]+=A[m+3][n+0]*B[n+0][k+2], C[m+3][k+3]+=A[m+3][n+0]*B[n+0][k+3];

                C[m+0][k+0]+=A[m+0][n+1]*B[n+1][k+0], C[m+0][k+1]+=A[m+0][n+1]*B[n+1][k+1];
                C[m+0][k+2]+=A[m+0][n+1]*B[n+1][k+2], C[m+0][k+3]+=A[m+0][n+1]*B[n+1][k+3];
                C[m+1][k+0]+=A[m+1][n+1]*B[n+1][k+0], C[m+1][k+1]+=A[m+1][n+1]*B[n+1][k+1];
                C[m+1][k+2]+=A[m+1][n+1]*B[n+1][k+2], C[m+1][k+3]+=A[m+1][n+1]*B[n+1][k+3];
                C[m+2][k+0]+=A[m+2][n+1]*B[n+1][k+0], C[m+2][k+1]+=A[m+2][n+1]*B[n+1][k+1];
                C[m+2][k+2]+=A[m+2][n+1]*B[n+1][k+2], C[m+2][k+3]+=A[m+2][n+1]*B[n+1][k+3];
                C[m+3][k+0]+=A[m+3][n+1]*B[n+1][k+0], C[m+3][k+1]+=A[m+3][n+1]*B[n+1][k+1];
                C[m+3][k+2]+=A[m+3][n+1]*B[n+1][k+2], C[m+3][k+3]+=A[m+3][n+1]*B[n+1][k+3];

                C[m+0][k+0]+=A[m+0][n+2]*B[n+2][k+0], C[m+0][k+1]+=A[m+0][n+2]*B[n+2][k+1];
                C[m+0][k+2]+=A[m+0][n+2]*B[n+2][k+2], C[m+0][k+3]+=A[m+0][n+2]*B[n+2][k+3];
                C[m+1][k+0]+=A[m+1][n+2]*B[n+2][k+0], C[m+1][k+1]+=A[m+1][n+2]*B[n+2][k+1];
                C[m+1][k+2]+=A[m+1][n+2]*B[n+2][k+2], C[m+1][k+3]+=A[m+1][n+2]*B[n+2][k+3];
                C[m+2][k+0]+=A[m+2][n+2]*B[n+2][k+0], C[m+2][k+1]+=A[m+2][n+2]*B[n+2][k+1];
                C[m+2][k+2]+=A[m+2][n+2]*B[n+2][k+2], C[m+2][k+3]+=A[m+2][n+2]*B[n+2][k+3];
                C[m+3][k+0]+=A[m+3][n+2]*B[n+2][k+0], C[m+3][k+1]+=A[m+3][n+2]*B[n+2][k+1];
                C[m+3][k+2]+=A[m+3][n+2]*B[n+2][k+2], C[m+3][k+3]+=A[m+3][n+2]*B[n+2][k+3];

                C[m+0][k+0]+=A[m+0][n+3]*B[n+3][k+0], C[m+0][k+1]+=A[m+0][n+3]*B[n+3][k+1];
                C[m+0][k+2]+=A[m+0][n+3]*B[n+3][k+2], C[m+0][k+3]+=A[m+0][n+3]*B[n+3][k+3];
                C[m+1][k+0]+=A[m+1][n+3]*B[n+3][k+0], C[m+1][k+1]+=A[m+1][n+3]*B[n+3][k+1];
                C[m+1][k+2]+=A[m+1][n+3]*B[n+3][k+2], C[m+1][k+3]+=A[m+1][n+3]*B[n+3][k+3];
                C[m+2][k+0]+=A[m+2][n+3]*B[n+3][k+0], C[m+2][k+1]+=A[m+2][n+3]*B[n+3][k+1];
                C[m+2][k+2]+=A[m+2][n+3]*B[n+3][k+2], C[m+2][k+3]+=A[m+2][n+3]*B[n+3][k+3];
                C[m+3][k+0]+=A[m+3][n+3]*B[n+3][k+0], C[m+3][k+1]+=A[m+3][n+3]*B[n+3][k+1];
                C[m+3][k+2]+=A[m+3][n+3]*B[n+3][k+2], C[m+3][k+3]+=A[m+3][n+3]*B[n+3][k+3]; 
                   */                      
            }
        }
    }
    return C;
}

int main(){
    int m,n,k;
    cin>>m>>n>>k;
    srand(time(NULL));
    vector<vector<double>> A(m,vector<double>(n,0)),B(n,vector<double>(k,0)),C(m,vector<double>(k,0));
    time_t st,fin;

    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j) A[i][j]=rand()%50;
    for(int i=0;i<n;++i)
        for(int j=0;j<k;++j) B[i][j]=rand()%50;

    st=clock();
    C=GEMM(A,B,m,n,k);
    fin=clock();
    cout<<"time cost = "<<double(fin -st)/CLOCKS_PER_SEC<<" s"<<endl;

    //print
    // cout<<"A:"<<endl;
    // PRINT(A,m,n);
    // cout<<"B:"<<endl;
    // PRINT(B,n,k);
    // cout<<"C=A*B:"<<endl;
    // PRINT(C,m,k);
}
