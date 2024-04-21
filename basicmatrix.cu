import os
os.environ['PATH'] += ':/usr/local/cuda/bin'
os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/lib64'
!pip install pycuda

# Write CUDA C code to a file
cuda_code = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

__global__ void matrixMultiplication(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

int main() {
    int m = 8000;   // Number of rows in matrix A
    int n = 9000;   // Number of columns in matrix A
    int k = 10000;   // Number of columns in matrix B

    // Seed the random number generator
    srand(time(NULL));

    // Matrix A
    float* A = (float*)malloc(m * n * sizeof(float));
    for (int i = 0; i < m * n; i++) {
        A[i] = (float)rand() / RAND_MAX;  // Generate a random float value between 0 and 1
    }

    // Matrix B
    float* B = (float*)malloc(n * k * sizeof(float));
    for (int i = 0; i < n * k; i++) {
        B[i] = (float)rand() / RAND_MAX;  // Generate a random float value between 0 and 1
    }

    // Result matrix
    float* C = (float*)malloc(m * k * sizeof(float));
   
    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    int size_A = m * n * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * k * sizeof(float);
    clock_t start_time = clock();
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Transfer data from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 gridSize((k - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1, 1);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Launch the kernel
  
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();
    

    // Copy the result from device to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    clock_t end_time = clock();
    // Print the execution time
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution Time: %.7f",execution_time);
    printf(" seconds");

    // Free host and device memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

"""
with open("cuda1.cu", "w") as f:
    f.write(cuda_code)

# Compile CUDA code
!nvcc -o h cuda1.cu

# Run compiled binary
!./h
