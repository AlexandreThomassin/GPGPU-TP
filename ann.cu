#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

// const int N = 1024 * 1024;
// const int threadsPerBlock = 256;
// const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock ;

double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    static bool generate;
    static double z1;

	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    double *m = (double*) malloc(w->rows * w->columns * sizeof(double));
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        m[idx] =  normalRand(0, 1 / sqrt(nneurones_prev));
    }
    cudaMemcpy(w->m, m, w->rows * w->columns * sizeof(double), cudaMemcpyHostToDevice);
}

ann_t * create_ann_kernel(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t *nn;

    cudaMalloc((void **) &nn, sizeof(ann_t));

    layer_t **layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t*));

    cudaMemcpy(&(nn->number_of_layers), &number_of_layers, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn->alpha), &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(nn->minibatch_size), &minibatch_size, sizeof(unsigned), cudaMemcpyHostToDevice);

    layer_t *layer0 = create_layer_kernel(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    layers[0] = layer0;

    for (int l = 1; l < number_of_layers; l++)
    {
        layer_t *layerl = create_layer_kernel(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
        layers[l] = layerl;
    }

    
    layer_t **d_layers;
    cudaMalloc((void **) &d_layers, number_of_layers * sizeof(layer_t*));
    cudaMemcpy(d_layers, layers, number_of_layers * sizeof(layer_t), cudaMemcpyHostToDevice);

    cudaMemcpy(&(nn->layers), &d_layers, sizeof(layer_t**), cudaMemcpyHostToDevice);

    return nn;
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t * nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer_kernel(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t *layer;
    cudaMalloc((void **) &layer, sizeof(layer_t));

    cudaMemcpy(&(layer->number_of_neurons), &number_of_neurons, sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(&(layer->minibatch_size), &minibatch_size, sizeof(unsigned), cudaMemcpyHostToDevice);
    
    matrix_t *activations = alloc_matrix_kernel(number_of_neurons, minibatch_size);
    matrix_t *z = alloc_matrix_kernel(number_of_neurons, minibatch_size);
    matrix_t *delta = alloc_matrix_kernel(number_of_neurons, minibatch_size);
    matrix_t *weights = alloc_matrix_kernel(number_of_neurons, nneurons_previous_layer);    
    matrix_t *biases = alloc_matrix_kernel(number_of_neurons, 1);

    cudaMemcpy(&(layer->activations), &activations, sizeof(matrix_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(layer->z), &z, sizeof(matrix_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(layer->delta), &delta, sizeof(matrix_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(layer->biases), &biases, sizeof(matrix_t*), cudaMemcpyHostToDevice);

    if (layer_number > 0)
    {
        matrix_t *w = (matrix_t*)malloc(sizeof(matrix_t));
        init_weight(w, nneurons_previous_layer);
        cudaMemcpy(weights, w, sizeof(matrix_t), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(&(layer->weights), &weights, sizeof(matrix_t*), cudaMemcpyHostToDevice);
    return layer;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix_kernel(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix_kernel(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix_kernel(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix_kernel(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix_kernel(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn)
{

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix_kernel(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix_kernel(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix_kernel(1, nn->minibatch_size);

        // double *m = (double*) malloc(one->columns * one->rows * sizeof(double));
        // for (int idx = 0; idx < one->columns*one->rows; idx++)
        //     m[idx] = 1.0;

        // cudaMemcpy(one->m, m, one->rows * one->columns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(one->m, 1, one->rows * one->columns * sizeof(double));
        
        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(((double)nn->layers[l]->weights->columns) / blockDim.x), ceil(((double)nn->layers[l-1]->activations->rows) / blockDim.y));
        matrix_dot_kernel<<<gridDim, blockDim>>>(nn->layers[l]->weights->m, nn->layers[l-1]->activations->m, z1->m, nn->layers[l]->weights->rows, nn->layers[l]->weights->columns, nn->layers[l-1]->activations->rows, nn->layers[l-1]->activations->columns); // z1 <- w^l x a^(l-1)
        cudaDeviceSynchronize();

        dim3 gridDim2(ceil(((double)nn->layers[l]->biases->columns) / blockDim.x), ceil(((double)one->rows) / blockDim.y));
        matrix_dot_kernel<<<gridDim2, blockDim>>>(nn->layers[l]->biases->m, one->m, z2->m, nn->layers[l]->biases->rows, nn->layers[l]->biases->columns, one->rows, one->columns); // z2 <- b^l x 1        
        cudaDeviceSynchronize();

        dim3 gridDim3(ceil(((double)nn->layers[l]->z->columns) / blockDim.x), ceil(((double)nn->layers[l]->z->rows) / blockDim.y));
        matrix_sum_kernel<<<gridDim3, blockDim>>>(z1->m, z2->m, nn->layers[l]->z->m, nn->layers[l]->z->rows, nn->layers[l]->z->columns); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1  
        cudaDeviceSynchronize();

        matrix_function_kernel_sig<<<gridDim3, blockDim>>>(nn->layers[l]->z->m, nn->layers[l]->activations->m, nn->layers[l]->activations->rows, nn->layers[l]->activations->columns);
        cudaDeviceSynchronize();


        destroy_matrix_kernel(z1);
        destroy_matrix_kernel(z2);
        destroy_matrix_kernel(one);

    }
}

void backward(ann_t *nn, matrix_t *y)
{
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix_kernel(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)nn->layers[L]->activations->columns) / blockDim.x), ceil(((double)nn->layers[L]->activations->rows) / blockDim.y));

    matrix_minus_kernel<<<gridDim, blockDim>>>(nn->layers[L]->activations->m, y->m, nn->layers[L]->delta->m, y->rows, y->columns);  // delta^(L) = (a^L - y)

    matrix_function_kernel_dsig<<<gridDim, blockDim>>>(nn->layers[L]->z->m, dfzL->m, nn->layers[L]->z->rows, nn->layers[L]->z->columns); // f'(z^(L))

    hadamard_product_kernel<<<gridDim, blockDim>>>(nn->layers[L]->delta->m, dfzL->m, nn->layers[L]->delta->m, nn->layers[L]->delta->rows, nn->layers[L]->delta->columns); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix_kernel(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw, *delta_tmp, *dfz;
        tw = alloc_matrix_kernel(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = alloc_matrix_kernel(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix_kernel(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        dim3 gridDim2(ceil(((double)tw->rows) / blockDim.x), ceil(((double)tw->columns) / blockDim.y));
        matrix_transpose_kernel<<<gridDim2, blockDim>>>(nn->layers[l]->weights->m, tw->m, tw->columns, tw->rows); // (w^l)T      

        dim3 gridDim3(ceil(((double)tw->columns) / blockDim.x), ceil(((double)delta_tmp->rows) / blockDim.y));
        matrix_dot_kernel<<<gridDim3, blockDim>>>(tw->m, nn->layers[l]->delta->m, delta_tmp->m, tw->rows, tw->columns, nn->layers[l]->delta->rows, nn->layers[l]->delta->columns); // (w^l)T x delta^l

        dim3 gridDim4(ceil(((double)dfz->columns) / blockDim.x), ceil(((double)dfz->rows) / blockDim.y));
        matrix_function_kernel_dsig<<<gridDim4, blockDim>>>(nn->layers[l-1]->z->m, dfz->m, dfz->rows, dfz->columns); // f'(z^(l-1))
        hadamard_product_kernel<<<gridDim4, blockDim>>>(delta_tmp->m, dfz->m, nn->layers[l-1]->delta->m, dfz->rows, dfz->columns); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        destroy_matrix_kernel(tw);
        destroy_matrix_kernel(delta_tmp);
        destroy_matrix_kernel(dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix_kernel(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix_kernel(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        dim3 gridDim5(ceil(((double)ta->rows) / blockDim.x), ceil(((double)ta->columns) / blockDim.y));
        matrix_transpose_kernel<<<gridDim5, blockDim>>>(nn->layers[l-1]->activations->m, ta->m, ta->columns, ta->rows); // ta <- (a^(l-1))^T

        dim3 gridDim6(ceil(((double) w1->columns) / blockDim.x), ceil(((double)w1->rows) / blockDim.y));
        matrix_dot_kernel<<<gridDim6, blockDim>>>(nn->layers[l]->delta->m, ta->m, w1->m, nn->layers[l]->delta->rows, nn->layers[l]->delta->columns, ta->rows, ta->columns); // w1 <- delta^l x (a^(l-1))^T

        matrix_scalar_kernel<<<gridDim6, blockDim>>>(w1->m, nn->alpha / nn->minibatch_size, w1->m, w1->rows, w1->columns); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus_kernel<<<gridDim6, blockDim>>>(nn->layers[l]->weights->m, w1->m, nn->layers[l]->weights->m, w1->rows, w1->columns); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        destroy_matrix_kernel(w1);
        destroy_matrix_kernel(ta);

        matrix_t *one, *b1;
        b1 = alloc_matrix_kernel(nn->layers[l]->number_of_neurons, 1);
        one = alloc_matrix_kernel(nn->minibatch_size, 1);
        // double *m = (double*)malloc(one->columns * one->rows * sizeof(double));
        // for (int idx = 0; idx < one->columns*one->rows; idx++)
        //     m[idx] = 1.0;

        // cudaMemcpy(one->m, m, one->rows * one->columns * sizeof(double), cudaMemcpyHostToDevice);

        // free(m);
        cudaMemset(one->m, 1, one->rows * one->columns * sizeof(double));


        dim3 gridDim7(ceil(((double) b1->columns) / blockDim.x), ceil(((double)b1->rows) / blockDim.y));
        matrix_dot_kernel<<<gridDim7, blockDim>>>(nn->layers[l]->delta->m, one->m, b1->m, nn->layers[l]->delta->rows, nn->layers[l]->delta->columns, one->rows, one->columns); // b1 <- delta^l x 1^T
        matrix_scalar_kernel<<<gridDim7, blockDim>>>(b1->m,  nn->alpha / nn->minibatch_size, b1->m, b1->rows, b1->columns); // b1 <- alpha / m . delta^l x 1^T
        matrix_minus_kernel<<<gridDim7, blockDim>>>(nn->layers[l]->biases->m, b1->m, nn->layers[l]->biases->m, b1->rows, b1->columns); // b^l = b^l - alpha / m . delta^l x 1^T
        
        // destroy_matrix_kernel(one);
        // destroy_matrix_kernel(b1);
    }
}

void forward_CPU(ann_t *nn, double (*activation_function)(double))
{

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        matrix_dot(nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)
        matrix_dot(nn->layers[l]->biases, one, z2); // z2 <- b^l x 1        
        matrix_sum(z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1      

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
     
        destroy_matrix_kernel(z1);
        destroy_matrix_kernel(z2);
        destroy_matrix_kernel(one);
    }
}