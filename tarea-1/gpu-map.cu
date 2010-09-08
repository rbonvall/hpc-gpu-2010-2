#include "gpu-map.hpp"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>

// Los kernels d_f y d_g implementan las funciones f y g
// aplicadas a un valor x cualquiera.
// Estos kernels están declarados con el calificador __device__,
// por lo que sólo pueden ser llamados por código
// que está siendo ejecutado en la GPU.

__device__ float d_f(float x) {
    float s = 0.0;
    for (int k = 1; k <= 10000; ++k) {
        s += sinf(2 * float(M_PI) * k * x);
    }
    return s;
}
__device__ float d_g(float x) {
    return x * x;
}

// Los kernels map_f y map_g se encargan de que cada hebra
// ejecute las funciones d_f y d_g con el argumento apropiado.

__global__ void map_f(float x[]) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i] = d_f(x[i]);
}

__global__ void map_g(float x[]) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    x[i] = d_g(x[i]);
}


// gpu_map hace toda la burocracia de ejecutar código en la GPU:
// reserva memoria, copia el arreglo, lanza el kernel,
// copia el resultado de vuelta y libera la memoria.
//
// La función está parametrizada con el nombre de la función
// representado como un char. Es posible encapsular los kernels
// usando functores para usar plantillas, pero ahora preferí
// que el código CUDA fuera lo más simple posible.

void gpu_map(char function_name, float x[], unsigned n) {

    unsigned mem_size = sizeof(float) * n;
     
    float *d_x;
    cudaMalloc((void **) &d_x, n * sizeof(float));
    cudaMemcpy(d_x, x, mem_size, cudaMemcpyHostToDevice);

    dim3 grid_size, block_size;
    block_size.x = 512;      // number of threads per block (<= 512)
    grid_size.x = n / 512;   // number of blocks

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    switch (function_name) {
        case 'f': map_f<<<grid_size, block_size>>>(d_x); break;
        case 'g': map_g<<<grid_size, block_size>>>(d_x); break;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(x, d_x, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    d_x = 0;
}

