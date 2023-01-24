#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;

void print_matrix(matrix_t *m, bool is_short);

__global__
void matrix_dot_kernel(double *A, double *B, double *C, unsigned numARows, unsigned numAColumns, unsigned numBRows, unsigned numBColumns);

__global__
void matrix_sum_kernel(double *m1, double *m2, double *res, unsigned rows, unsigned cols);

__global__
void matrix_function_kernel_sig(double *m1, double *res, unsigned rows, unsigned cols);

__global__
void matrix_function_kernel_dsig(double *m1, double *res, unsigned rows, unsigned cols);

__device__
double sigmoid_kernel(double x);

__device__
double dsigmoid_kernel(double x);

__global__
void matrix_minus_kernel(double *m1, double *m2, double *res, unsigned rows, unsigned cols);

__global__
void hadamard_product_kernel(double *m1, double *m2, double *res, unsigned rows, unsigned cols);

__global__
void matrix_transpose_kernel(double *m1, double *res, unsigned rows, unsigned cols);

__global__
void matrix_scalar_kernel(double *m1,double s, double *res, unsigned rows, unsigned cols);

matrix_t * alloc_matrix_kernel(unsigned rows, unsigned columns);

void destroy_matrix_kernel(matrix_t *m);

#endif