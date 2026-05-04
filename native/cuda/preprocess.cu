// GPU преподготовка кадра под нейросеть: BGR uint8 (HWC) -> planar RGB float32 NCHW [1,3,H,W].
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

__global__ void bgr_hwc_to_rgb_nchw_float_kernel(const unsigned char* __restrict__ bgr_hwc,
                                                  float* __restrict__ nchw,
                                                  int w,
                                                  int h,
                                                  float inv255) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }
  int idx_hwc = (y * w + x) * 3;
  unsigned char b = bgr_hwc[idx_hwc + 0];
  unsigned char g = bgr_hwc[idx_hwc + 1];
  unsigned char r = bgr_hwc[idx_hwc + 2];

  float rf = static_cast<float>(r) * inv255;
  float gf = static_cast<float>(g) * inv255;
  float bf = static_cast<float>(b) * inv255;

  int plane = h * w;
  int base = y * w + x;
  nchw[0 * plane + base] = rf;
  nchw[1 * plane + base] = gf;
  nchw[2 * plane + base] = bf;
}

}  // namespace

extern "C" cudaError_t integra_cuda_bgr_to_nchw_float(const unsigned char* d_bgr_hwc,
                                                       float* d_nchw,
                                                       int width,
                                                       int height,
                                                       cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  float inv255 = 1.0f / 255.0f;
  bgr_hwc_to_rgb_nchw_float_kernel<<<grid, block, 0, stream>>>(d_bgr_hwc, d_nchw, width, height, inv255);
  return cudaGetLastError();
}
