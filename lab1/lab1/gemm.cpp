
//������ֵ��50���ڵ���������ɵľ��󣬲����о���˷�
//A��B�����ɵ���������C=A*B

#include<iostream>
#include<vector>
#include<time.h>
using namespace std;

//�������
void PRINT(vector<vector<double>> x,int m,int n){
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j) cout<<x[i][j]<<" "; 
        cout<<endl;
    }
    cout<<endl;
}

//������о���˷�
vector<vector<double>> GEMM(vector<vector<double>> A, vector<vector<double>> B, int m, int n, int k){  
    vector<vector<double>> C(m,vector<double>(k,0));
    int sumt;
    for(int i=0;i<k;++i){
        for(int j=0;j<m;++j){
            sumt=0;
            for(int x=0;x<n;++x) sumt+=A[j][x]*B[x][i];
            C[j][i]=sumt;
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