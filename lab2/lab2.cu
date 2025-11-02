%%writefile image.cu
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

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h, int delta_w, int delta_h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y, x_frame, y_frame;
    uchar4 p;
    int r, g, b, a;
    for(y = idy; y < h; y += offsety) {
        for(x = idx; x < w; x += offsetx) {
            r = 0;
            g = 0;
            b = 0;
            a = 0;
            for (y_frame = 0; y_frame < delta_h; ++y_frame) {
                for (x_frame = 0; x_frame < delta_w; ++x_frame) {
                    p = tex2D< uchar4 >(tex, (x * delta_w + x_frame) / w, (y * delta_h + y_frame) / h);
                    r += p.x;
                    g += p.y;
                    b += p.z;
                    a += p.w;
                }
            }
            out[y * w + x] = make_uchar4((int) (r / (delta_h * delta_w)), (int) (g / (delta_h * delta_w)), (int) (b / (delta_h * delta_w)), (int) (a / (delta_h * delta_w)));
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

    int delta_w = w / new_w, delta_h = h / new_h;

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
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeMirror; // Clamp
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	  CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h, delta_w, delta_h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

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
