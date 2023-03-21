#ifndef _MATRIX_MULTIPLY_
#define _MATRIX_MULTIPLY_

#include<iostream>
#include<vector>
using namespace std;

vector<vector<double>> matrix_multiply(vector<vector<double>> A,vector<vector<double>> B, int m,int n ,int k);

void matrix_multiply(vector<vector<double>> *A,vector<vector<double>> *B,vector<vector<double>> *C, int m,int n ,int k);

void matrix_multiply(double *A, double *B, double *C,int m,int n ,int k);

#endif