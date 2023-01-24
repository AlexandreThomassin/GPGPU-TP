#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__device__
double sigmoid_kernel(double x)
{
    return 1 / (1 + expf(-x));
}

__device__
double dsigmoid_kernel(double x)
{
    return sigmoid_kernel(x)*(1-sigmoid_kernel(x));
}


matrix_t * alloc_matrix_kernel(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );

    double *m;
    cudaMalloc((void **) &m, columns * rows * sizeof(double));
    cudaMemset(m, 0, columns * rows * sizeof(double));
    res->m = m;
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix_kernel(matrix_t *m)
{
    cudaFree(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

__global__
void hadamard_product_kernel(double *m1, double *m2, double *res, unsigned rows, unsigned cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int idx;											// Element address

    // a_ij = a[i][j], where a is in row major order
	if(row < rows && col < cols){
		idx = col + row * cols; 				
		res[idx] = m1[idx] * m2[idx];
	}
}

__global__
void matrix_sum_kernel(double *m1, double *m2, double *res, unsigned rows, unsigned cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int idx;											// Element address

	if(row < rows && col < cols){
		idx = col + row * cols; 				
		res[idx] = m1[idx] + m2[idx];
	}
}


__global__
void matrix_minus_kernel(double *m1, double *m2, double *res, unsigned rows, unsigned cols)
{       
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int idx;											// Element address

	if(row < rows && col < cols){
		idx = col + row * cols; 				
		res[idx] = m1[idx] - m2[idx];
	}
}

__global__
void matrix_dot_kernel(double *A, double *B, double *C, unsigned numARows, unsigned numAColumns, unsigned numBRows, unsigned numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < numARows && col < numBColumns) {
        double sum = 0;
        for (int ii = 0; ii < numAColumns; ii++) {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

__global__
void matrix_function_kernel_sig(double *m1, double *res, unsigned rows, unsigned cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int idx;											// Element address

	if(row < rows && col < cols){
		idx = col + row * cols; 				
		res[idx] = sigmoid_kernel(m1[idx]);
	}

}

__global__
void matrix_function_kernel_dsig(double *m1, double *res, unsigned rows, unsigned cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int idx;											// Element address

	if(row < rows && col < cols){
		idx = col + row * cols; 				
		res[idx] = dsigmoid_kernel(m1[idx]);
	}

}


__global__
void matrix_transpose_kernel(double *m1, double *res, unsigned rows, unsigned cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address

	if(row < rows && col < cols){
        res[row + col * rows] = m1[col + row * cols];
	} 

}


__global__
void matrix_scalar_kernel(double *m1, double s, double *res, unsigned rows, unsigned cols)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int idx;											// Element address

    if(row < rows && col < cols){
        idx = col + row * cols; 
        res[idx] = m1[idx] * s;
	} 

}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}