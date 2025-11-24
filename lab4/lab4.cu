#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace std;

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
      return abs(a) < abs(b);
    }
};

__global__ void replace(double *arr, int n, int start, int end) {
    double tmp;
    for(int x = blockDim.x * blockIdx.x + threadIdx.x; x < n + 1; x += blockDim.x * gridDim.x) {
        tmp = arr[x * n + start];
        arr[x * n + start] = arr[x * n + end];
        arr[x * n + end] = tmp;
    }
}

__global__ void division(double *arr, int n, int i)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int x = idx + i; x < n + 1; x += offset) {
        arr[x * n + i] /= arr[i * n + i];
    }
}

__global__ void kernel(double *arr, int n, int now) {    
    for(int x = now + blockDim.x * blockIdx.x + threadIdx.x + 1; x < n; x += blockDim.x * gridDim.x) {
        for(int y = now + blockDim.y * blockIdx.y + threadIdx.y + 1; y < n + 1; y += blockDim.y * gridDim.y) {
            arr[y * n + x] -= arr[y * n + now] / arr[now * n + now] * arr[now * n + x];
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int n;
    cin >> n;
    double *arr = (double *)malloc(sizeof(double) * n * (n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> arr[i + j * n];
        }
    }

    for (int i = 0; i < n; ++i) {
        cin >> arr[n * n + i];
    }

    double *dev_arr;
    CSC(cudaMalloc(&dev_arr, sizeof(double) * n * (n + 1)));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n * (n + 1), cudaMemcpyHostToDevice));

    comparator cmp;

    thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(dev_arr);
    thrust::device_ptr<double> max_el;
    for (int i = 0; i < n - 1; ++i) {
        max_el = thrust::max_element(p_arr + i * n + i, p_arr + (i + 1) * n, cmp);
        // cout << res - p_arr << ' ' << arr[res - p_arr] << '\n';
        if (i * n + i != max_el - p_arr) {
            replace<<< 512, 512 >>>(dev_arr, n, i, max_el - p_arr - i * n);
        }

        division<<< 512, 512 >>>(dev_arr, n, i);
        kernel<<< dim3(32, 32), dim3(32, 32) >>>(dev_arr, n, i);

        /* for (int i = 0; i < n * (n + 1); ++i) {
            cout << arr[i] << ' ';
        }
        cout << '\n'; */
    }
    
    cudaMemcpy(arr, dev_arr, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost);
    double *res = (double *)malloc(sizeof(double) * n);
    double count;
    for (int i = n - 1; i >= 0; --i) {
        count = 0;
        for (int j = i + 1; j < n; ++j) {
            count += (arr[j * n + i] * res[j]);
        }
        res[i] = (arr[n * n + i] - count) / arr[i * n + i];
    }

    for (int i = 0; i < n; ++i) {
        cout << res[i] << ' ';
    }

	  CSC(cudaFree(dev_arr));

    free(arr);
    free(res);

    return 0;
}
