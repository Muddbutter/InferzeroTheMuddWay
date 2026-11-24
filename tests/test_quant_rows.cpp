// With no arguments (or --self-test), runs the model-free Q5_1 regression used by CI.
// Prints the CPU dequantization of one row of a GGUF tensor (first values, sum, sum of squares), to compare the
// decoders in src/quants.h with an independent implementation. usage: test_quant_rows model.gguf TENSOR [row] [n]
// With PAD=1 and a small-block type (IQ4_NL / Q2_0) it also checks pad_small_block_rows: the padded row must
// dequantize to the same K values followed by zeros.
#include "../src/gguf.h"
#include "../src/quants.h"
#include <cstdio>
#include <vector>

static int q51_reference_test() {
  int checked = 0;
  for (int K : {32, 64, 128, 256, 512, 640, 768, 1024}) {
    std::vector<uint8_t> source((size_t)K / 32 * 24);
    std::vector<float> expected(K), actual(K + 16, -123456.0f);
    for (int b = 0; b < K / 32; ++b) {
      // Exact binary fractions: expected values do not use the production decoder.
      const float d = b % 4 == 0 ? 0.0f : (b % 4 == 1 ? 0.25f : (b % 4 == 2 ? 0.5f : 2.0f));
      const float m = (b % 3 - 1) * 0.5f;
      uint8_t* p = source.data() + b * 24;
      const uint16_t dh = f32_to_fp16(d), mh = f32_to_fp16(m);
      memcpy(p, &dh, 2); memcpy(p + 2, &mh, 2);
      uint32_t high = 0;
      for (int j = 0; j < 32; ++j) {
        const int q = (j * 13 + b * 7) % 32;
        high |= (uint32_t)(q >> 4) << j;
        p[8 + j % 16] |= (q & 15) << (j < 16 ? 0 : 4);
        expected[b * 32 + j] = d * q + m;
      }
      memcpy(p + 4, &high, 4);
    }
    dequant_row(GGML_Q5_1, source.data(), K, actual.data() + 8);
    for (int i = 0; i < K; ++i) {
      if (actual[8 + i] != expected[i]) {
        std::printf("FAIL K=%d i=%d got=%g expected=%g\n", K, i, actual[8 + i], expected[i]);
        return 1;
      }
      ++checked;
    }
    for (int i = 0; i < 8; ++i)
      if (actual[i] != -123456.0f || actual[K + 8 + i] != -123456.0f) return 2;
  }
  std::printf("test_quant_rows: all passed (%d Q5_1 values, eight row widths, output guards unchanged)\n", checked);
  return 0;
}

// NVFP4 (ggml type 40): the scale decode against ggml's ggml_ue4m3_to_fp32 written out literally, dequant_row against
// ggml's dequantize_row_nvfp4 loop, the GPU raw form's per-32 addressing (zero/kernels/decode.cl decode32 QT 40) and the
// per-32 fold nvfp4_block32_w8: exact for equal scales and power-of-two ratios up to 8, else within d / 2.
static float ggml_ue4m3_literal(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  const int exp = (x >> 3) & 0xF, man = x & 0x7;
  const float raw = exp == 0 ? std::ldexp((float)man, -9) : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
  return raw * 0.5f;
}
static int nvfp4_reference_test() {
  static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (int x = 0; x < 256; ++x)
    if (nvfp4_scale((uint8_t)x) != ggml_ue4m3_literal((uint8_t)x)) {
      std::printf("FAIL nvfp4 scale 0x%02X: %g, ggml %g\n", x, nvfp4_scale((uint8_t)x), ggml_ue4m3_literal((uint8_t)x));
      return 1;
    }
  // a row of 4 superblocks (16 blocks of 64): scale bytes from a walk over all 256 values, codes from a counter
  const int K = 1024, NB = K / 64;
  std::vector<uint8_t> row((size_t)NB * 36);
  for (int b = 0; b < NB; ++b)
    for (int i = 0; i < 36; ++i) row[(size_t)b * 36 + i] = (uint8_t)(i < 4 ? (b * 4 + i) * 37 + 11 : (b * 31 + i * 7) & 0xFF);
  std::vector<float> got(K), want(K);
  dequant_row(GGML_NVFP4, row.data(), K, got.data());
  for (int b = 0; b < NB; ++b)  // ggml's loop, as written there
    for (int s = 0; s < 4; ++s) {
      const float d = ggml_ue4m3_literal(row[(size_t)b * 36 + s]);
      for (int j = 0; j < 8; ++j) {
        const uint8_t c = row[(size_t)b * 36 + 4 + s * 8 + j];
        want[b * 64 + s * 16 + j] = kv[c & 0x0F] * d;
        want[b * 64 + s * 16 + j + 8] = kv[c >> 4] * d;
      }
    }
  for (int i = 0; i < K; ++i)
    if (got[i] != want[i]) { std::printf("FAIL nvfp4 dequant i=%d got=%g want=%g\n", i, got[i], want[i]); return 2; }
  // the GPU raw form: per superblock, 32-block ib's scales at [2 ib], its codes at 16 + 16 ib
  int exact = 0, folded = 0;
  for (int sb = 0; sb < K / 256; ++sb) {
    uint8_t g[144];
    gpu_raw_sb(GGML_NVFP4, row.data() + (size_t)sb * 144, g);
    for (int ib = 0; ib < 8; ++ib) {
      int8_t q[32];
      const float d = nvfp4_block32_w8(g + 2 * ib, g + 16 + 16 * ib, q);
      const float s0 = nvfp4_scale(g[2 * ib]), s1 = nvfp4_scale(g[2 * ib + 1]);
      const float lo = std::min(s0, s1), hi = std::max(s0, s1);
      bool pow2 = lo == hi || lo == 0.0f;
      for (float r = 0.5f; r >= 0.125f && !pow2; r *= 0.5f) pow2 = lo == hi * r;
      for (int j = 0; j < 32; ++j) {
        const float w = want[sb * 256 + ib * 32 + j], v = d * q[j];
        if (q[j] > 127 || q[j] < -127) { std::printf("FAIL nvfp4 fold: code %d\n", q[j]); return 3; }
        if (pow2 ? v != w : std::fabs(v - w) > d * 0.5f * (1.0f + 1e-6f)) {
          std::printf("FAIL nvfp4 fold sb=%d ib=%d j=%d: %g (d %g q %d), exact %g\n", sb, ib, j, v, d, q[j], w);
          return 4;
        }
      }
      (pow2 ? exact : folded) += 32;
    }
  }
  // every pair of scale bytes, the codes +-12 (the largest): the fold's bound and code range
  for (int a = 0; a < 256; ++a)
    for (int c = 0; c < 256; ++c) {
      const uint8_t sc[2] = {(uint8_t)a, (uint8_t)c}, qs[16] = {0x77, 0xFF, 0x77, 0xFF, 0x77, 0xFF, 0x77, 0xFF,
                                                                0xF7, 0x7F, 0xF7, 0x7F, 0xF7, 0x7F, 0xF7, 0x7F};
      int8_t q[32];
      const float d = nvfp4_block32_w8(sc, qs, q);
      for (int j = 0; j < 32; ++j) {
        const float w = (float)kv[(qs[(j / 16) * 8 + j % 8] >> (j % 16 < 8 ? 0 : 4)) & 15] * nvfp4_scale(sc[j / 16]);
        if (std::fabs(d * q[j] - w) > d * 0.5f * (1.0f + 1e-6f)) {
          std::printf("FAIL nvfp4 fold scales 0x%02X 0x%02X j=%d: %g, exact %g\n", a, c, j, d * q[j], w);
          return 5;
        }
      }
    }
  std::printf("test_quant_rows: nvfp4 passed (256 scales, %d values exact, %d values folded within d/2, 65536 scale pairs)\n",
              exact, folded);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--self-test")) {
    if (const int r = nvfp4_reference_test()) return 10 + r;
    return q51_reference_test();
  }
  if (argc < 3) { fprintf(stderr, "usage: test_quant_rows model.gguf TENSOR [row] [n]\n"); return 1; }
  GgufFile f(argv[1]);
  const GgufTensor& t = f.get(argv[2]);
  const uint64_t row = argc > 3 ? strtoull(argv[3], nullptr, 10) : 0;
  const int n = argc > 4 ? atoi(argv[4]) : 8;
  int blk = 0, bytes = 0;
  ggml_block_info(t.type, blk, bytes);
  const int K = (int)t.dims[0];
  const size_t row_bytes = (size_t)K / blk * bytes;
  std::vector<float> v(K);
  dequant_row(t.type, t.data + row * row_bytes, K, v.data());
  double s = 0, s2 = 0;
  for (float x : v) { s += x; s2 += (double)x * x; }
  printf("%s %s K=%d row %llu: sum %.6f sumsq %.6f first:", argv[2], ggml_type_name(t.type), K, (unsigned long long)row, s, s2);
  for (int i = 0; i < n && i < K; ++i) printf(" %.6f", v[i]);
  printf("\n");
  if (getenv("PAD") && is_small_block_type(t.type)) {
    const int Kp = padded_k(K);
    std::vector<uint8_t> padded((size_t)Kp / blk * bytes);
    pad_small_block_rows(t.type, t.data + row * row_bytes, 1, K, padded.data());
    std::vector<float> p(Kp);
    dequant_row(t.type, padded.data(), Kp, p.data());
    int bad = 0;
    for (int i = 0; i < Kp; ++i) {
      const float want = i < K ? v[i] : 0.0f;
      if (p[i] != want) ++bad;
    }
    printf("padded to K=%d (%zu bytes): %s\n", Kp, padded.size(), bad ? "MISMATCH" : "identical + zeros");
    return bad ? 1 : 0;
  }
  return 0;
}
