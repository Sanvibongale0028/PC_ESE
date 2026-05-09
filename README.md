# Parallel Matrix Multiplication Using MPI + CUDA

This experiment demonstrates Parallel Matrix Multiplication using MPI and CUDA on GPU systems. The main objective of the program is to understand how parallel computing improves computational performance by dividing matrix multiplication tasks among multiple processes and GPU threads.

MPI (Message Passing Interface) is used for process-level parallelism, where multiple processes are created and assigned computational tasks. CUDA is used for GPU-based parallelism, where thousands of GPU threads execute matrix multiplication operations simultaneously. Together, MPI and CUDA help in efficient workload distribution and faster execution of large matrix computations.

In this experiment:

* MPI creates parallel processes.
* CUDA assigns parallel GPU threads for computation.
* Each thread computes one element of the output matrix.
* GPU memory management is performed using CUDA functions like `cudaMalloc()` and `cudaMemcpy()`.

The experiment also helps in understanding:

* Parallel programming concepts
* Process-level and thread-level parallelism
* GPU acceleration
* Task distribution in matrix multiplication
* Performance improvement using parallel execution

The program is implemented using CUDA C/C++ and OpenMPI and executed on NVIDIA GPU systems.
