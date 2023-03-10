// Compile gcc -o ./ann main.c matrix.c ann.c mnist.c -lm

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include <math.h>
#include <string.h>
#include <time.h>

void populate_minibatch(double *x, double* y, unsigned* minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size);

__global__
void populate_minibatch_kernel_x(double * x, unsigned * minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size);

__global__
void populate_minibatch_kernel_y(double * y, unsigned * minibatch_idx, unsigned minibatch_size, byte* label );

void zero_to_n(unsigned n, unsigned* t)
{
    for (unsigned i = 0; i < n; i++)
    {
        t[i] = i;
    }
}

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch)
{
    zero_to_n(size, t);
    for (unsigned i = 0; i < number_of_switch; i++)
    {
        unsigned x = rand() % size;
        unsigned y = rand() % size;
        unsigned tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}


double accuracy(image* test_img, byte* test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn)
{
    unsigned good = 0;
    unsigned idx[datasize];    
    double *x = (double *) malloc( 28 * 28 * minibatch_size * sizeof(double));
    double *y = (double *) malloc( 10 * minibatch_size * sizeof(double));
    double *m = (double *) malloc( 10 * minibatch_size * sizeof(double));
    zero_to_n(datasize, idx);
    
    for (int i = 0; i < datasize - minibatch_size; i+= minibatch_size)
    {        
        populate_minibatch(x, y, &idx[i], minibatch_size, test_img, 28*28, test_label, 10);
        cudaMemcpy(nn->layers[0]->activations->m, x, 28*28 * minibatch_size * sizeof(double), cudaMemcpyHostToDevice);     
        
        forward(nn);

        cudaMemcpy(m, nn->layers[nn->number_of_layers-1]->activations->m, 10 * minibatch_size * sizeof(double), cudaMemcpyDeviceToHost );

       
        for (int col = 0; col < minibatch_size; col ++)
        {
            int idxTrainingData = col + i ;
            double max = 0;
            unsigned idx_max = 0;
            for (int row = 0; row < 10; row++){
                int idx = col + row * minibatch_size;
                if (m[idx] > max){
                    max = m[idx];
                    idx_max = row;
                }
            }
            if (idx_max == test_label[idxTrainingData])
            {
                good ++;
            }
        }
    }    
    free(x);
    free(y);


    unsigned ntests = (datasize/minibatch_size) * minibatch_size;
    return (100.0* (double) (good) / ntests );
}

__global__
void populate_minibatch_kernel_x(double * x, unsigned * minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size)
{

    int row = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int col = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address

    if (col < minibatch_size)
    {
        if (row < img_size)
        {
            
            x[row * minibatch_size + col] = (double) img[minibatch_idx[col]][row]/255.;
        }
    }
}

__global__
void populate_minibatch_kernel_y(double * y, unsigned * minibatch_idx, unsigned minibatch_size, byte* label )
{
    int img_id = threadIdx.x % minibatch_size;
	int neuron_id = threadIdx.x / minibatch_size;

    if (label[minibatch_idx[img_id]] == neuron_id){
        y[threadIdx.x ] = 1.0;
    }
    else{
        y[threadIdx.x] = 0;
    }
}

void populate_minibatch(double * x, double * y, unsigned * minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size)
{
    for (int col = 0; col < minibatch_size; col ++)
    {
        for (int row = 0; row < img_size; row ++)
        {
            x[row * minibatch_size + col] = (double) img[minibatch_idx[col]][row]/255.;
        }

        for (int row = 0; row < 10; row ++)
        {
            y[row * minibatch_size + col] = 0.0;
        }

        y[ label[minibatch_idx[col]] * minibatch_size + col] = 1.0;
    }
}

int main(int argc, char *argv[])
{
    srand(time(0));
    unsigned datasize, ntest;
    float time = 0;

    // Lecture des images
    image* test_img = read_images("t10k-images-idx3-ubyte", &ntest);
    byte* test_label = read_labels("t10k-labels-idx1-ubyte", &ntest);
    image* train_img_kernel = read_images_kernel("train-images-idx3-ubyte", &datasize);
    byte* train_label_kernel = read_labels_kernel("train-labels-idx1-ubyte", &datasize);

    // D??finition des param??tres du r??seau de neurones
    ann_t * nn;
    double alpha = 0.05;
    unsigned minibatch_size = 16;
    unsigned number_of_layers = 3;
    unsigned nneurons_per_layer[3] = {28*28, 30, 10};
    nn = create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer);
    //print_nn(nn);

    printf("Neural Network copy to device Successful !\n\n");

    printf("Starting accuracy %lf\n\n", accuracy(test_img, test_label, ntest, minibatch_size, nn));

    
    unsigned *shuffled_idx = (unsigned *)malloc(datasize*sizeof(unsigned));


    matrix_t *out = alloc_matrix_kernel(10, minibatch_size);


    // D??finition des param??tres d'appel des fonctions sur GPU
    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((double)nn->layers[0]->activations->columns) / blockDim.x), ceil(((double)nn->layers[0]->activations->rows) / blockDim.y));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = ((10*minibatch_size)+threadsPerBlock-1) / threadsPerBlock ;
    
    for (int epoch = 0; epoch < 40; epoch ++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate( &start );
        cudaEventCreate( &stop );
        cudaEventRecord( start, 0 );

        printf("Start learning epoch %d\n", epoch);

        shuffle(shuffled_idx, datasize, datasize);
        unsigned *d_shuffled_idx;
        cudaMalloc((void **) &d_shuffled_idx, datasize*sizeof(unsigned));
        cudaMemcpy(d_shuffled_idx, shuffled_idx, datasize*sizeof(unsigned), cudaMemcpyHostToDevice);

        for (int i = 0; i < datasize - minibatch_size ; i+= minibatch_size)
        {
            populate_minibatch_kernel_x<<<gridDim, blockDim>>>(nn->layers[0]->activations->m, d_shuffled_idx+i, minibatch_size, train_img_kernel, 28*28);
            
            forward(nn);

            populate_minibatch_kernel_y<<<blocksPerGrid, threadsPerBlock>>>(out->m, d_shuffled_idx+i, minibatch_size, train_label_kernel);

            backward(nn, out);            
        }     
        printf("epoch %d accuracy %lf\n", epoch, accuracy(test_img, test_label, ntest, minibatch_size, nn));

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        time += elapsedTime;
        printf("Temps Epoch %d : %f ms \n\n", epoch, elapsedTime);
    }

    printf("Temps total : %f", time);

    free(shuffled_idx);
    destroy_matrix_kernel(out);   
    
    return 0;
}

