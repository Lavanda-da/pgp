#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;
using output_type = unsigned char;

const int MAX_BLOCK_SIZE = 256;

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void hist(int n, unsigned char *in_arr, int *out_arr) {
    __shared__ int sdata[MAX_BLOCK_SIZE];
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int x = idx; x < MAX_BLOCK_SIZE; x += offsetx) {
        sdata[x] = 0;
        out_arr[x] = 0;
    }
    __syncthreads();

    for (int x = idx; x < n; x += offsetx) {
        atomicAdd(sdata + in_arr[x], 1);
    }
    __syncthreads();

    for (int x = idx; x < MAX_BLOCK_SIZE; x += offsetx) {
        atomicAdd(out_arr + x, *(sdata + x));
    }
    __syncthreads();
}

__global__ void scan(int *in_data, int *out_data) {
    __shared__ int sdata[MAX_BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = 1;

    /* sdata[2 * tid] = in_data[2 * tid];
    sdata[2 * tid + 1] = in_data[2 * tid + 1]; */

    sdata[tid] = in_data[index];

    __syncthreads();

    for (int s = MAX_BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
        __syncthreads();
        
    }

    if (tid == 0) {
        sdata[MAX_BLOCK_SIZE - 1] = 0;
    }

    for (int s = 1; s < MAX_BLOCK_SIZE; s <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < s) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int tmp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += tmp;
        }
    }
    __syncthreads();

    out_data[index] = sdata[tid];
}

__global__ void kernel(output_type *sorted_arr, int *scan_arr, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for(int x = idx; x < n; x += offsetx) {
        int end = (x != n - 1 ? scan_arr[x + 1] : n);
        for (int i = scan_arr[x]; i < end; ++i) {
            sorted_arr[i] = x;
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int n;
    fread(&n, sizeof(int), 1, stdin);
    unsigned char *arr = (unsigned char *)malloc(sizeof(unsigned char) * n);
    fread(arr, sizeof(unsigned char), n, stdin);

    /* int n;
    cin >> n;
    unsigned char *arr = (unsigned char *)malloc(sizeof(unsigned char) * n);
    for (int i = 0; i < n; ++i) {
        cin >> arr[i];
    } */

    unsigned char *in_arr;
	  CSC(cudaMalloc(&in_arr, sizeof(unsigned char) * MAX_BLOCK_SIZE));
    CSC(cudaMemcpy(in_arr, arr, sizeof(unsigned char) * n, cudaMemcpyHostToDevice));
    
    int *out_arr;
	  CSC(cudaMalloc(&out_arr, sizeof(int) * MAX_BLOCK_SIZE));
    CSC(cudaMemset(out_arr, 0, sizeof(int) * MAX_BLOCK_SIZE));

    hist<<<1, 128>>>(n, in_arr, out_arr);

    int *out_arr2;
	  CSC(cudaMalloc(&out_arr2, sizeof(int) * MAX_BLOCK_SIZE));

    scan<<<2, 256>>>(out_arr, out_arr2);

    output_type *res_arr;
	  CSC(cudaMalloc(&res_arr, sizeof(output_type) * n));

    kernel<<<128, 128>>>(res_arr, out_arr2, n);

    output_type *sorted_arr = (output_type *)malloc(sizeof(output_type) * n);
    cudaMemcpy(sorted_arr, res_arr, sizeof(output_type) * n, cudaMemcpyDeviceToHost);

    /* for (int i = 0; i < n; ++i) {
        cout << sorted_arr[i] << ' ';
    } */

    fwrite(sorted_arr, sizeof(unsigned char), n, stdout);

    return 0;
}
