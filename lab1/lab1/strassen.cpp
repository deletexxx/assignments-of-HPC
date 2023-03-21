
//������ֵ��50���ڵ���������ɵľ��󣬲����о���˷�
//A��B�����ɵ���������C=A*B
//��strssen�㷨�������˷�

#include<iostream>
#include<vector>
#include<time.h>
using namespace std;

//�������
void PRINT(vector<vector<double>> x,int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j) cout<<x[i][j]<<" "; 
        cout<<endl;
    }
    cout<<endl;
}

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

//�������������
vector<vector<double>> ADDmatrix(vector<vector<double>> A, vector<vector<double>> B, int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j) A[i][j]+=B[i][j];
    }
    return A;
}

//�������������
vector<vector<double>> SUBmatrix(vector<vector<double>> A, vector<vector<double>> B, int n){
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j) A[i][j]=A[i][j]-B[i][j];
    }
    return A;
}

//strassen�㷨��������
vector<vector<double>> strassen(vector<vector<double>> A, vector<vector<double>> B, int n){
    vector<vector<double>> C(n,vector<double>(n,0));

    if(n==1){
        C[0][0]=A[0][0]*B[0][0];
        return C;
    }
    else if(n==16) return C=GEMM(A,B,n,n,n);

    //�����￪ʼn���ԭ����һ�룬�����о���Ļ���
    n=n/2;
    //���廮�ֺ��A��B��C�����Լ�s��p
    vector<vector<double>> A11(n, vector<double>(n,0)), A12(n, vector<double>(n,0)),
                        A21(n, vector<double>(n,0)), A22(n, vector<double>(n,0)),
                        B11(n, vector<double>(n,0)), B12(n, vector<double>(n,0)),
                        B21(n, vector<double>(n,0)), B22(n, vector<double>(n,0)),
                        C11(n, vector<double>(n,0)), C12(n, vector<double>(n,0)),
                        C21(n, vector<double>(n,0)), C22(n, vector<double>(n,0)),
                        s1(n, vector<double>(n,0)), s2(n, vector<double>(n,0)),
                        s3(n, vector<double>(n,0)), s4(n, vector<double>(n,0)),
                        s5(n, vector<double>(n,0)), s6(n, vector<double>(n,0)),
                        s7(n, vector<double>(n,0)), s8(n, vector<double>(n,0)),
                        s9(n, vector<double>(n,0)), s10(n, vector<double>(n,0)),
                        p1(n, vector<double>(n,0)), p2(n, vector<double>(n,0)),
                        p3(n, vector<double>(n,0)), p4(n, vector<double>(n,0)),
                        p5(n, vector<double>(n,0)), p6(n, vector<double>(n,0)),
                        p7(n, vector<double>(n,0));

    //break down matrix A B
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            A11[i][j]=A[i][j],   A12[i][j]=A[i][j+n];
            A21[i][j]=A[i+n][j], A22[i][j]=A[i+n][j+n];
            B11[i][j]=B[i][j],   B12[i][j]=B[i][j+n];
            B21[i][j]=B[i+n][j], B22[i][j]=B[i+n][j+n];
        }
    }

    //computer S1,S2,...,S10
    //S1=B12-B22, S2=A11+A12
    s1=SUBmatrix(B12,B22,n);
    s2=ADDmatrix(A11,A12,n);
    //S3=A21+A22, S4=B21-B11
    s3=ADDmatrix(A21,A22,n);
    s4=SUBmatrix(B21,B11,n);
    //S5=A11+A22, S6=B11+B22
    s5=ADDmatrix(A11,A22,n);
    s6=ADDmatrix(B11,B22,n);
    //S7=A12-A22, S8=B21+B22
    s7=SUBmatrix(A12,A22,n);
    s8=ADDmatrix(B21,B22,n);
    //S9=A11-A21, S10=B11+B12
    s9=SUBmatrix(A11,A21,n);
    s10=ADDmatrix(B11,B12,n);

    //Start recursion
    //p1=A11*s1, p2=s2*B22, p3=s3*B11, p4=A22*s4
    //p5=s5*s6, p6=s7*s8, p7=s9*s10
    p1=strassen(A11,s1,n);
    p2=strassen(s2,B22,n);
    p3=strassen(s3,B11,n);
    p4=strassen(A22,s4,n);
    p5=strassen(s5,s6,n);
    p6=strassen(s7,s8,n);
    p7=strassen(s9,s10,n);

    //computer four parts of matrix C
    //C11=p5+p4-p2+p6
    C11=ADDmatrix(p5,p4,n);
    C11=SUBmatrix(C11,p2,n);
    C11=ADDmatrix(C11,p6,n);
    //C12=p1+p2, C21=p3+p4
    C12=ADDmatrix(p1,p2,n);
    C21=ADDmatrix(p3,p4,n);
    //C22=p5+p1-p3-p7
    C22=ADDmatrix(p5,p1,n);
    C22=SUBmatrix(C22,p3,n);
    C22=SUBmatrix(C22,p7,n);

    //Integration matrix C
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            C[i][j]=C11[i][j],C[i][j+n]=C12[i][j];
            C[i+n][j]=C21[i][j],C[i+n][j+n]=C22[i][j];
        }
    }

    return C;
}

int main(){
    int n;
    cin>>n;

    if( (n & (n-1)) != 0 ){//judge n if 2^x
        cout<<n<<" isn't power of 2, we can't us strassen to computer."<<endl;
        return 0;
    }

    vector<vector<double>> A(n,vector<double>(n,0)),B(n,vector<double>(n,0)),C(n,vector<double>(n,0));
    
    //time 
    srand(time(NULL));
    time_t st,fin;
    //��СΪn*n���������A��B
    for(int i=0;i<n;++i)
        for(int j=0;j<n;++j)    A[i][j]=rand()%50;
    for(int i=0;i<n;++i)
        for(int j=0;j<n;++j)    B[i][j]=rand()%50;

    //computer
    st=clock();
    C=strassen(A,B,n);
    fin=clock();
    cout<<"time cost = "<<double(fin - st)/CLOCKS_PER_SEC<<" s"<<endl;

    //print
    // cout<<"A:"<<endl;
    // PRINT(A,n);
    // cout<<"B:"<<endl;
    // PRINT(B,n);
    // cout<<"C=A*B:"<<endl;
    // PRINT(C,n);

}
