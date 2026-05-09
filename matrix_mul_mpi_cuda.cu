#include <stdio.h>
#include <cuda.h>
#include <mpi.h>

#define N 2

__global__ void matrixMul(int *A, int *B, int *C)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;

    for(int k = 0; k < N; k++)
    {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int h_A[N*N] = {1,2,3,4};
    int h_B[N*N] = {5,6,7,8};
    int h_C[N*N];

    int *d_A, *d_B, *d_C;

    cudaSetDevice(rank);

    cudaMalloc((void**)&d_A, N*N*sizeof(int));
    cudaMalloc((void**)&d_B, N*N*sizeof(int));
    cudaMalloc((void**)&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(N, N);

    matrixMul<<<1, threads>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    if(rank == 0)
    {
        printf("Result Matrix:\n");

        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%d ", h_C[i*N + j]);
            }
            printf("\n");
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    MPI_Finalize();

    return 0;
}
