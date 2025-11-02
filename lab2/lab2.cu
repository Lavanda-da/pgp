#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

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

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h, int new_w, int new_h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    int delta_w = w / new_w;
    int delta_h = h / new_h;
    int frame_x, frame_y;
    int r, g, b, a;
    uchar4 p;
    for(y = idy * delta_h; y < h; y += offsety) {
        for(x = idx * delta_w; x < w; x += offsetx) {
            r = 0, g = 0, b = 0, a = 0;
            for(frame_x = 0; frame_x < delta_w; ++frame_x) {
                for(frame_y = 0; frame_y < delta_h; ++frame_y) {
                    p = tex2D<uchar4>(tex, x + frame_x, y + frame_y);
                    r += p.x;
                    g += p.y;
                    b += p.z;
                    a += p.w;
                }
            }
            r /= (delta_w * delta_h);
            g /= (delta_w * delta_h);
            b /= (delta_w * delta_h);
            a /= (delta_w * delta_h);
            out[(y / delta_h) * new_w + x / delta_w] = make_uchar4(r, g, b, a);
        }
    }
}

int main() {
    string input, output;
    int new_w, new_h;
    cin >> input >> output >> new_w >> new_h;

    int w, h;
   	FILE *fp = fopen(input.c_str(), "rb");
 	  fread(&w, sizeof(int), 1, fp);
	  fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	  CSC(cudaMalloc(&dev_out, sizeof(uchar4) * new_w * new_h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h, new_w, new_h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    uchar4 *data2 = (uchar4 *)malloc(sizeof(uchar4) * new_w * new_h);
    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * new_w * new_h, cudaMemcpyDeviceToHost));

	  CSC(cudaDestroyTextureObject(tex));
	  CSC(cudaFreeArray(arr));
	  CSC(cudaFree(dev_out));

    fp = fopen(output.c_str(), "wb");
    fwrite(&new_w, sizeof(int), 1, fp);
    fwrite(&new_h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), new_w * new_h, fp);
    fclose(fp);

    free(data);
    return 0;
}
