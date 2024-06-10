# BLAS-CUDA

CUDA based BLAS functions with complete loop unrolling and reduction for:
1. spMV_M: sparse matrix - vector multiplication with sparse matrix being diagonal (Jacobi) matrix
2. spMV: sparse matrix - vector multiplication with vector being extended in-place to facilitate coalesced memory access
3. spMV_thread: sparse matrix - vector multiplication for one thread per matrix row
4. spMV_warp: sparse matrix - vector multiplication for one warp (32 thread) per matrix row
5. dot_product: vector - vector dot product
6. nrm2 - norm2 of a vector without multiphase model of accumulation as did in cuBLAS implementation (https://docs.nvidia.com/cuda/cublas/)
7. axpy - axpy function
8. get_const_int_symbol - set the value for __constant__ d_Nnode, which is the length of the vector 

NOTE: sparse matrix is assummed to have CSR (compressed row) format
