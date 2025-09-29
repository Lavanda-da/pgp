#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(double *arr1, double *arr2, double *res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while(idx < n) {
        res[idx] = arr1[idx] + arr2[idx];
        idx += offset;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    double *arr1 = (double *)malloc(sizeof(double) * n);
    double *arr2 = (double *)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++) {
        scanf("%lf", &arr1[i]);
    }
    for(int i = 0; i < n; i++) {
        scanf("%lf", &arr2[i]);
    }

    double *dev_arr1;
    cudaMalloc(&dev_arr1, sizeof(double) * n);
    cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice);
    double *dev_arr2;
    cudaMalloc(&dev_arr2, sizeof(double) * n);
    cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice);
    double *dev_res;
    cudaMalloc(&dev_res, sizeof(double) * n);

    kernel<<<1024, 1024>>>(dev_arr1, dev_arr2, dev_res, n);

    double *res = (double *)malloc(sizeof(double) * n);
    cudaMemcpy(res, dev_res, sizeof(double) * n, cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; i++) {
        printf("%.10lf ", res[i]);
    }
    printf("\n");

    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_res);
    free(arr1);
    free(arr2);
    free(res);
    return 0;
}
