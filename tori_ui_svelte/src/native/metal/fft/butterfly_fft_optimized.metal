#include <metal_stdlib>
using namespace metal;

inline float2 cmul(float2 a, float2 b) { return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
inline float2 twiddle(uint k, uint m2); // supply via constant/texture

kernel void butterfly_fft_optimized(
  device float2* data [[buffer(0)]],
  constant uint& stage [[buffer(1)]],
  constant uint& width [[buffer(2)]],
  uint2 gid [[thread_position_in_grid]],
  uint tid  [[thread_index_in_threadgroup]]
){
  threadgroup float2 tg[256]; // 128-256 threads/tg sweet spot

  device float4* vec = reinterpret_cast<device float4*>(data);
  float4 v = vec[(gid.y * width + tid) >> 1];  // two float2 in a float4
  tg[tid] = float2(v.x, v.y);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = 1; s <= stage; ++s) {
    uint m  = 1u << s;
    uint m2 = m >> 1;
    uint k  = tid % m2;
    uint j1 = (tid / m2) * m + k;
    uint j2 = j1 + m2;
    float2 w = twiddle(k, m2);
    float2 t = cmul(w, tg[j2]);
    float2 u = tg[j1];
    tg[j1]   = u + t;
    tg[j2]   = u - t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  data[gid.y * width + tid] = tg[tid];
}