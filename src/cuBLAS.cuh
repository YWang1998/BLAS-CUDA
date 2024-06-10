#pragma once
// CUDA Runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Utilities and system includes
#include <helper_functions.h>  // shared functions common to CUDA Samples
#include <helper_cuda.h>       // CUDA error checking

static __constant__ int d_Nnode; // length of the vector

static __constant__ double d_scale; // scale of vector-vector addition

namespace cuBLAS {

	void spMV_M(dim3& Grid, dim3& Block, const double* d_M, const double* d_V, double* d_Target);
	void spMV(dim3& Grid, dim3& Block,
		const double* d_a_expand,
		const int* d_ja_expand,
		const double* d_v, double* d_v_expanded, double* d_spMV);
	void spMV_thread(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const double* d_a,
		const double* d_v,
		double* d_spMV);
	void spMV_warp(dim3& Grid, dim3& Block,
		const int* d_ia_expand,
		const int* d_ja_expand,
		const double* d_a,
		const double* d_v,
		double* d_spMV);
	void dot_product(dim3& Grid, dim3& Block, const double* __restrict__ d_V_1, const double* __restrict__ d_V_2, double* product, double* __restrict__ d_product);
	void nrm2(dim3& Grid, dim3& Block, const double* __restrict__ V, double* sum, double* __restrict__ d_sum);
	void axpy(dim3& Grid, dim3& Block, const double* __restrict__ x, double* __restrict__ y, const double& scale);
	void get_const_int_symbol(const int& h_symbol);

}