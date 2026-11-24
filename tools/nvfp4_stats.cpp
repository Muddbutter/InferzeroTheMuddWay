// How often the engine's per-32 fold of NVFP4 (src/quants.h nvfp4_block32_w8, zero/kernels/decode.cl QT 40) is exact
// over a GGUF's NVFP4 tensors: per 32-value block, its two 16-value scales equal, a power-of-two ratio up to 8 (or
// one 0), or neither (the smaller half then rounded, at most d / 2). Reads only the scale bytes (4 of every 36), no
// model run. usage: nvfp4_stats model.gguf
#include "../src/gguf.h"
#include "../src/quants.h"
#include <cstdio>
#include <map>

int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr, "usage: nvfp4_stats model.gguf\n"); return 1; }
  GgufFile f(argv[1]);
  uint64_t n[3] = {0, 0, 0};  // equal, power of two, folded
  int tensors = 0;
  for (const GgufTensor& t : f.tensors) {
    if (t.type != GGML_NVFP4) continue;
    ++tensors;
    uint64_t m[3] = {0, 0, 0};
    const uint64_t blocks = t.nbytes / 36;
    for (uint64_t b = 0; b < blocks; ++b) {
      const uint8_t* p = t.data + b * 36;
      for (int h = 0; h < 2; ++h) {
        const float s0 = nvfp4_scale(p[2 * h]), s1 = nvfp4_scale(p[2 * h + 1]);
        const float lo = std::min(s0, s1), hi = std::max(s0, s1);
        int k = 2;
        if (lo == hi) k = 0;
        else if (lo == 0.0f || lo == hi * 0.5f || lo == hi * 0.25f || lo == hi * 0.125f) k = 1;
        ++m[k];
      }
    }
    const double tot = (double)(m[0] + m[1] + m[2]);
    printf("%-36s equal %5.1f %%  power of two %5.1f %%  folded %5.1f %%\n", t.name.c_str(), 100 * m[0] / tot,
           100 * m[1] / tot, 100 * m[2] / tot);
    for (int i = 0; i < 3; ++i) n[i] += m[i];
  }
  const double tot = (double)(n[0] + n[1] + n[2]);
  if (!tensors) { printf("no NVFP4 tensors\n"); return 0; }
  printf("all %d NVFP4 tensors, %.0f 32-blocks: exact %.2f %% (equal %.2f, power of two %.2f), folded %.2f %%\n", tensors,
         tot, 100 * (n[0] + n[1]) / tot, 100 * n[0] / tot, 100 * n[1] / tot, 100 * n[2] / tot);
  return 0;
}
