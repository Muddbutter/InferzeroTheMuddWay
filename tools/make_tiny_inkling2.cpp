// A small Inkling 2 (`inkling2_loop`) GGUF with the real layout and random weights, for building and checking the
// engine's path against tools/ref_forward without the 23 GB model (docs/inkling2.md, step I1.4). After
// tools/make_tiny_deepseek41.cpp: its random numbers, encoders and writer.
//
// The keys and tensor names of unsloth/Inkling-2-Mini-GGUF's mainline Q8_K_XL, smaller sizes, chosen so that a
// prompt of ~40 tokens reaches every case:
//  - 8 layers, attention-residual blocks of 2, the loop core [2, 6) run 3 times (a fixed loop, as the real file's);
//    sliding-window layers and global ones (1, 5 and 7: 5 inside the loop, so a global layer gets messages too);
//  - head size 128 as the real model's (kernels may be written for it), 6 query heads over 2 KV heads (GQA 3, as 24 /
//    8), 16 relative buckets;
//  - a sliding window of 16 and a relative extent of 32, so that distances pass both tables' ends; a log-scale floor of
//    8, so that the global layers' factor moves from position 8;
//  - 8 experts with 2 active, FFN 128; NVFP4 experts with their f32 .scale tensors in layers 0-5 and BF16 ones in 6-7
//    (Unsloth's file: NVFP4 in 0-19, BF16 in 20-23); every other matrix BF16, norms and small tensors F32, the router F32;
//  - a vocabulary of 500 padded to 512 rows (the real 200,058 / 201,024), tokenizer pre "inkling2".
// --no-loop: architecture inkling2 (the same layers once, no loop keys or controllers).
// --ggml-org: the types of ggml-org's Inkling-2-Mini-IQ4_NL.gguf: IQ4_NL experts in layers 0-5, Q8_0 ones in 6-7 (no
// expert scales), the other matrices Q8_0, the loop controller F32.
// --nvfp4-equal: every NVFP4 block's four scales equal (the per-32 fold then exact: zero/kernels/decode.cl QT 40).
// --scale S: every matrix drawn S times its default size (S < 1: the router's sigmoids and the controllers' tanh in range,
//   as a trained model's mostly are; the default saturates them, and int8 rounding is then amplified).
// --all-experts: every expert active (expert_used_count = expert_count), so no top-k choice: the random router's near
//   ties (its sigmoids near 1, the bias at 0.1) flip under int8 rounding and part a check from the reference for that
//   reason alone (2026-11-23, the first engine runs); with all experts the comparison measures the arithmetic.
//
// usage: make_tiny_inkling2 OUT.gguf [--seed N] [--no-loop] [--nvfp4-equal] [--ggml-org] [--dspark DRAFT.gguf] [--all-experts] [--passes N] [--scale S] [--byte-vocab] [--exps-type T] [--mats-type T]
#include "../src/log.h"  // (zlog: first, as gguf_pack.h's headers use it)
#include "../zero/gguf_pack.h"  // (zpack::Writer / put_kv)
#include "../src/quants.h"  // (f32_to_fp16, nvfp4_scale)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int E = 256, NL = 8, H = 6, HKV = 2, D = 128, NREL = 16, SWA = 16, EXT = 32, NC = 4, BLOCK = 2;
double g_init = 1.0;  // (--scale S)
bool g_byte_vocab = false;  // (--byte-vocab: tokens 0..255 are GPT-2's byte characters, so text encodes: serverz tests)
const uint32_t kNoType = 0xFFFFFFFFu;  // (no --exps-type / --mats-type: F32 is type 0)
uint32_t g_exps_type = kNoType;  // (--exps-type q4_k / q6_k / iq4_xs: every expert tensor of that type, the FFN 256 wide)
uint32_t g_mats_type = kNoType;  // (--mats-type T: the BF16 2D matrices of type T where their rows are whole blocks)
int NE = 8, NU = 2, FF = 128, RANK = 16, LSTART = 2, LEND = 6, PASSES = 3, NV = 500, NVP = 512;
const std::vector<int> GLOBAL = {1, 5, 7};
const int NVFP4_LAYERS = 6;  // (experts of layers 0..5 NVFP4, the rest BF16)

// ---- random numbers: xoshiro256** seeded through splitmix64; normals by Box-Muller (as make_tiny_deepseek41)
struct Rng {
  uint64_t s[4];
  bool have = false;
  double spare = 0;
  explicit Rng(uint64_t seed) {
    for (auto& x : s) {
      seed += 0x9E3779B97F4A7C15ull;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      x = z ^ (z >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    if (have) { have = false; return spare; }
    double u1;
    do u1 = uniform(); while (u1 <= 0);
    const double u2 = uniform(), r = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
    spare = r * std::sin(a);
    have = true;
    return r * std::cos(a);
  }
};

typedef std::vector<float> F;

uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
template <typename T> void put(std::vector<uint8_t>& o, T v) {
  const size_t n = o.size();
  o.resize(n + sizeof v);
  memcpy(&o[n], &v, sizeof v);
}

bool g_nvfp4_equal = false;
int g_ggml_org = 0;  // (--ggml-org: the types of ggml-org's Inkling-2-Mini-IQ4_NL.gguf; 1 its 2D matrices, 2 its experts)

// NVFP4 (ggml type 40): per 16 values a UE4M3 scale s (value = kvalues_mxfp4[code] x s, s = ggml_ue4m3_to_fp32),
// chosen as the representable s nearest to amax / 12 (the largest code 12 then reaches about amax), the codes the
// nearest values; --nvfp4-equal: one scale for the whole 64-block, from its largest magnitude
void encode_nvfp4(const float* v, std::vector<uint8_t>& o) {
  static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  auto pick_scale = [](double want) {  // the byte (0x00..0x7E) whose scale is nearest to want
    int best = 0;
    double bd = INFINITY;
    for (int c = 0; c < 0x7F; ++c) {
      const double d = std::fabs(nvfp4_scale((uint8_t)c) - want);
      if (d < bd) bd = d, best = c;
    }
    return (uint8_t)best;
  };
  uint8_t sc[4];
  double amax_all = 0;
  for (int i = 0; i < 64; ++i) amax_all = std::max(amax_all, (double)std::fabs(v[i]));
  for (int s = 0; s < 4; ++s) {
    double amax = 0;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, (double)std::fabs(v[16 * s + i]));
    sc[s] = pick_scale((g_nvfp4_equal ? amax_all : amax) / 12.0);
  }
  for (int s = 0; s < 4; ++s) put<uint8_t>(o, sc[s]);
  for (int s = 0; s < 4; ++s) {
    const double d = nvfp4_scale(sc[s]);
    uint8_t q[16];
    for (int i = 0; i < 16; ++i) {
      const double y = d > 0 ? v[16 * s + i] / d : 0.0;
      int best = 0;
      double bd = INFINITY;
      for (int c = 0; c < 16; ++c) {
        const double dd = std::fabs(y - kv[c]);
        if (dd < bd) bd = dd, best = c;
      }
      q[i] = (uint8_t)best;
    }
    for (int j = 0; j < 8; ++j) put<uint8_t>(o, (uint8_t)(q[j] | (q[j + 8] << 4)));
  }
}

// (--exps-type: K-quant and IQ4_XS superblocks of 256, simple quantizers (not ggml's search): a valid file of the
//  type, the test being the engine against ref_forward on the same bytes; each superblock checked through the
//  engine's own dequant_row, so a layout slip stops here)
void encode_q4k(const float* v, std::vector<uint8_t>& o) {
  float sc[8], mn[8];
  for (int j = 0; j < 8; ++j) {
    float lo = 0, hi = 0;
    for (int i = 0; i < 32; ++i) lo = std::min(lo, v[32 * j + i]), hi = std::max(hi, v[32 * j + i]);
    sc[j] = (hi - lo) / 15.0f;
    mn[j] = -lo;
  }
  float msc = 0, mmn = 0;
  for (int j = 0; j < 8; ++j) msc = std::max(msc, sc[j]), mmn = std::max(mmn, mn[j]);
  const uint16_t dh = f32_to_fp16(msc / 63.0f), mh = f32_to_fp16(mmn / 63.0f);
  const float d = fp16_to_f32(dh), dmin = fp16_to_f32(mh);
  uint8_t ls[8], lm[8];
  for (int j = 0; j < 8; ++j) {
    ls[j] = (uint8_t)std::clamp((int)std::lround(d > 0 ? sc[j] / d : 0.0f), 0, 63);
    lm[j] = (uint8_t)std::clamp((int)std::lround(dmin > 0 ? mn[j] / dmin : 0.0f), 0, 63);
  }
  uint8_t s12[12] = {};
  for (int j = 0; j < 4; ++j) {
    s12[j] = (uint8_t)(ls[j] | ((ls[j + 4] >> 4) << 6));
    s12[j + 4] = (uint8_t)(lm[j] | ((lm[j + 4] >> 4) << 6));
    s12[j + 8] = (uint8_t)((ls[j + 4] & 15) | ((lm[j + 4] & 15) << 4));
  }
  put<uint16_t>(o, dh);
  put<uint16_t>(o, mh);
  for (uint8_t b : s12) put<uint8_t>(o, b);
  auto q = [&](int j, int i) {
    const float s = d * ls[j];
    return std::clamp((int)std::lround(s > 0 ? (v[32 * j + i] + dmin * lm[j]) / s : 0.0f), 0, 15);
  };
  for (int c = 0; c < 4; ++c)
    for (int i = 0; i < 32; ++i) put<uint8_t>(o, (uint8_t)(q(2 * c, i) | (q(2 * c + 1, i) << 4)));
}
void encode_q6k(const float* v, std::vector<uint8_t>& o) {
  float sc[16], ms = 0;
  for (int s = 0; s < 16; ++s) {
    float a = 0;
    for (int i = 0; i < 16; ++i) a = std::max(a, std::fabs(v[16 * s + i]));
    sc[s] = a / 31.0f;
    ms = std::max(ms, sc[s]);
  }
  const uint16_t dh = f32_to_fp16(ms / 127.0f);
  const float d = fp16_to_f32(dh);
  int8_t is[16];
  for (int s = 0; s < 16; ++s) is[s] = (int8_t)std::clamp((int)std::lround(d > 0 ? sc[s] / d : 0.0f), -127, 127);
  uint8_t q[256];
  for (int e = 0; e < 256; ++e) {
    const float st = d * is[e / 16];
    q[e] = (uint8_t)(std::clamp((int)std::lround(st != 0 ? v[e] / st : 0.0f), -32, 31) + 32);
  }
  uint8_t ql[128], qh[64] = {};
  for (int n = 0; n < 2; ++n)  // (ggml's dequantize_row_q6_K order: halves of 128, four runs of 32)
    for (int l = 0; l < 32; ++l) {
      const uint8_t* y = q + 128 * n;
      ql[64 * n + l] = (uint8_t)((y[l] & 15) | ((y[l + 64] & 15) << 4));
      ql[64 * n + l + 32] = (uint8_t)((y[l + 32] & 15) | ((y[l + 96] & 15) << 4));
      qh[32 * n + l] = (uint8_t)((y[l] >> 4) | ((y[l + 32] >> 4) << 2) | ((y[l + 64] >> 4) << 4) | ((y[l + 96] >> 4) << 6));
    }
  for (uint8_t b : ql) put<uint8_t>(o, b);
  for (uint8_t b : qh) put<uint8_t>(o, b);
  for (int8_t b : is) put<int8_t>(o, b);
  put<uint16_t>(o, dh);
}
void encode_iq4xs(const float* v, std::vector<uint8_t>& o) {
  float sc[8], ms = 0;
  for (int j = 0; j < 8; ++j) {
    float a = 0;
    for (int i = 0; i < 32; ++i) a = std::max(a, std::fabs(v[32 * j + i]));
    sc[j] = a / 127.0f;
    ms = std::max(ms, sc[j]);
  }
  const uint16_t dh = f32_to_fp16(ms / 31.0f);
  const float d = fp16_to_f32(dh);
  uint8_t ls[8];
  uint16_t sh = 0;
  uint8_t sl[4] = {};
  for (int j = 0; j < 8; ++j) {
    ls[j] = (uint8_t)std::clamp((int)std::lround(d > 0 ? sc[j] / d : 0.0f) + 32, 0, 63);
    sl[j / 2] |= (uint8_t)((ls[j] & 15) << (4 * (j % 2)));
    sh |= (uint16_t)((ls[j] >> 4) << (2 * j));
  }
  put<uint16_t>(o, dh);
  put<uint16_t>(o, sh);
  for (uint8_t b : sl) put<uint8_t>(o, b);
  for (int j = 0; j < 8; ++j) {
    const float dl = d * (ls[j] - 32);
    uint8_t q[32];
    for (int i = 0; i < 32; ++i) {
      int best = 0;
      float bd = INFINITY;
      for (int c = 0; c < 16; ++c) {
        const float dd = std::fabs(v[32 * j + i] - dl * (float)kvalues_iq4nl[c]);
        if (dd < bd) bd = dd, best = c;
      }
      q[i] = (uint8_t)best;
    }
    for (int i = 0; i < 16; ++i) put<uint8_t>(o, (uint8_t)(q[i] | (q[i + 16] << 4)));
  }
}

// a type by its GGUF name, any case (q4_k, IQ2_XXS, mxfp4, f16, f32, ...); kNoType when unknown
uint32_t type_by_name(std::string s) {
  for (char& c : s) c = (char)toupper((unsigned char)c);
  for (uint32_t t = 0; t < 64; ++t)
    if (s == ggml_type_name(t)) return t;
  return kNoType;
}
// (--exps-type / --mats-type, the types with no quantizer here (the IQ1 / IQ2 / IQ3 grids, Q2_K / Q3_K / Q5_K, the
//  32-blocks, MXFP4): random code bytes, the block's scale fields set so that its values have the rms of x's block
//  (value linear in the scale; a second offset field, Q4_1's m or a K-quant's dmin, zero); a valid file of the type, the
//  test being the engine against ref_forward on the same bytes. Kinds: 'd' f16 scale, 'z' f16 zeroed, 'e' MXFP4's E8M0
//  exponent, 'm' IQ1_M's scale in the top nibbles of its 4 scale words)
struct ScaleAt { int off; char kind; };
std::vector<ScaleAt> scale_fields(uint32_t t) {
  switch (t) {
    case GGML_Q4_0: case GGML_Q5_0: case GGML_Q8_0: case GGML_IQ4_NL: case GGML_Q2_0: case GGML_IQ2_XXS: case GGML_IQ2_XS:
    case GGML_IQ2_S: case GGML_IQ3_XXS: case GGML_IQ3_S: case GGML_IQ1_S: case GGML_IQ4_XS: return {{0, 'd'}};
    case GGML_Q4_1: case GGML_Q5_1: case GGML_Q8_1: case GGML_Q4_K: case GGML_Q5_K: return {{0, 'd'}, {2, 'z'}};
    case GGML_Q2_K: return {{80, 'd'}, {82, 'z'}};
    case GGML_Q3_K: return {{108, 'd'}};
    case GGML_Q6_K: return {{208, 'd'}};
    case GGML_IQ1_M: return {{48, 'm'}};
    case GGML_MXFP4: return {{0, 'e'}};
    default: return {};
  }
}
void put_scale(uint8_t* b, const std::vector<ScaleAt>& fs, float d, int e, float zf = 0.0f) {
  for (const ScaleAt& s : fs) {
    if (s.kind == 'z') { const uint16_t z = f32_to_fp16(zf); memcpy(b + s.off, &z, 2); }
    if (s.kind == 'd') { const uint16_t h = f32_to_fp16(d); memcpy(b + s.off, &h, 2); }
    if (s.kind == 'e') b[s.off] = (uint8_t)std::clamp(e, 1, 254);
    if (s.kind == 'm') {
      const uint16_t h = f32_to_fp16(d);
      for (int i = 0; i < 4; ++i) {
        uint16_t w;
        memcpy(&w, b + s.off + 2 * i, 2);
        w = (uint16_t)((w & 0x0FFF) | (((h >> (4 * i)) & 15) << 12));
        memcpy(b + s.off + 2 * i, &w, 2);
      }
    }
  }
}
std::vector<uint8_t> encode_random(const F& x, uint32_t type) {
  int blk = 0, bytes = 0;
  ggml_block_info(type, blk, bytes);
  const std::vector<ScaleAt> fs = scale_fields(type);
  if (fs.empty() || blk <= 0) throw std::runtime_error(std::string("encode: no generic form for ") + ggml_type_name(type));
  if (x.size() % blk) throw std::runtime_error(std::string("encode: ") + ggml_type_name(type) + " needs whole blocks");
  static Rng rng(0xC0DE5EEDull);
  std::vector<uint8_t> o(x.size() / blk * bytes);
  std::vector<float> back(blk);
  for (size_t b = 0; b < x.size() / blk; ++b) {
    uint8_t* p = o.data() + b * bytes;
    for (int i = 0; i < bytes; ++i) p[i] = (uint8_t)(rng.next() >> 56);
    double want = 0;
    for (int i = 0; i < blk; ++i) want += (double)x[b * blk + i] * x[b * blk + i];
    want = std::sqrt(want / blk) + 1e-6;
    put_scale(p, fs, 1.0f, 127);  // (unit scale: the codes' own rms, then the scale that gives want)
    dequant_row(type, p, blk, back.data());
    double got = 0;
    for (float v : back) got += (double)v * v;
    got = std::sqrt(got / blk);
    if (!(got > 1e-12) || !std::isfinite(got)) {  // (all codes zero: a fixed pattern instead)
      for (int i = 0; i < bytes; ++i) p[i] = (uint8_t)(0x5A + 37 * i);
      put_scale(p, fs, 1.0f, 127);
      dequant_row(type, p, blk, back.data());
      got = 0;
      for (float v : back) got += (double)v * v;
      got = std::sqrt(got / blk) + 1e-12;
    }
    // (the offset field (Q4_1 / Q5_1's m, a K-quant's dmin): the one that makes the block's mean zero, values being
    //  linear in it; with it 0 every value had one sign, the matrices biased and the tiny model flat)
    float zc = 0.0f;
    bool hz = false;
    for (const ScaleAt& s : fs) hz = hz || s.kind == 'z';
    if (hz) {
      std::vector<float> v1(blk);
      put_scale(p, fs, 1.0f, 127, 0.0f);
      dequant_row(type, p, blk, back.data());
      put_scale(p, fs, 1.0f, 127, 1.0f);
      dequant_row(type, p, blk, v1.data());
      double m0 = 0, dm = 0;
      for (int i = 0; i < blk; ++i) m0 += back[i], dm += v1[i] - back[i];
      if (std::fabs(dm) > 1e-9) zc = (float)(-m0 / dm);
      put_scale(p, fs, 1.0f, 127, zc);
      dequant_row(type, p, blk, back.data());
      got = 0;
      for (float v : back) got += (double)v * v;
      got = std::sqrt(got / blk) + 1e-12;
    }
    put_scale(p, fs, (float)(want / got), 127 + (int)std::lround(std::log2(want / got)), (float)(zc * want / got));
    dequant_row(type, p, blk, back.data());
    for (float v : back)
      if (!std::isfinite(v)) throw std::runtime_error(std::string("encode: ") + ggml_type_name(type) + " gave a non-finite value");
  }
  return o;
}

std::vector<uint8_t> encode(const F& x, uint32_t type) {
  std::vector<uint8_t> o;
  if (type == GGML_F16) {
    for (float v : x) put<uint16_t>(o, f32_to_fp16(v));
    return o;
  }
  if (!scale_fields(type).empty() && type != GGML_Q4_K && type != GGML_Q6_K && type != GGML_IQ4_XS &&
      type != GGML_Q8_0 && type != GGML_IQ4_NL)
    return encode_random(x, type);
  if (type == GGML_Q4_K || type == GGML_Q6_K || type == GGML_IQ4_XS) {
    if (x.size() % 256) throw std::runtime_error("K-quants / IQ4_XS need a multiple of 256 values");
    const size_t sb = type == GGML_Q4_K ? 144 : type == GGML_Q6_K ? 210 : 136;
    float back[256];
    for (size_t b = 0; b < x.size() / 256; ++b) {
      const size_t at = o.size();
      if (type == GGML_Q4_K) encode_q4k(&x[b * 256], o);
      else if (type == GGML_Q6_K) encode_q6k(&x[b * 256], o);
      else encode_iq4xs(&x[b * 256], o);
      if (o.size() - at != sb) throw std::runtime_error("encode: superblock size");
      dequant_row(type, o.data() + at, 256, back);
      double err = 0, ref = 0;
      for (int i = 0; i < 256; ++i) err += (back[i] - x[b * 256 + i]) * (back[i] - x[b * 256 + i]), ref += x[b * 256 + i] * x[b * 256 + i];
      if (err > 0.05 * ref + 1e-12) throw std::runtime_error("encode: type " + std::to_string(type) + " round trip off (" +
                                                             std::to_string(std::sqrt(err / (ref + 1e-30))) + ")");
    }
    return o;
  }
  if (type == GGML_F32) {
    o.resize(x.size() * 4);
    memcpy(o.data(), x.data(), o.size());
  } else if (type == GGML_BF16) {
    for (float v : x) put<uint16_t>(o, f2bf(v));
  } else if (type == GGML_NVFP4) {
    if (x.size() % 64) throw std::runtime_error("NVFP4 needs a multiple of 64 values");
    for (size_t b = 0; b < x.size() / 64; ++b) encode_nvfp4(&x[b * 64], o);
  } else if (type == GGML_Q8_0 || type == GGML_IQ4_NL) {
    // (--ggml-org: blocks of 32, an f16 scale then 32 int8 (Q8_0) or 16 bytes of 4-bit codes into kvalues_iq4nl, value j
    //  in byte j's low nibble and value j + 16 in its high one (IQ4_NL); nearest values, the scale as stored)
    if (x.size() % 32) throw std::runtime_error("Q8_0 / IQ4_NL need a multiple of 32 values");
    for (size_t b = 0; b < x.size() / 32; ++b) {
      const float* v = &x[b * 32];
      float amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
      const uint16_t dh = f32_to_fp16(amax / 127.0f);
      const float d = fp16_to_f32(dh);
      put<uint16_t>(o, dh);
      if (type == GGML_Q8_0) {
        for (int i = 0; i < 32; ++i)
          put<int8_t>(o, (int8_t)std::clamp((int)std::lround(d > 0 ? v[i] / d : 0.0f), -127, 127));
      } else {
        uint8_t q[32];
        for (int i = 0; i < 32; ++i) {
          int best = 0;
          float bd = INFINITY;
          for (int c = 0; c < 16; ++c) {
            const float dd = std::fabs(v[i] - d * (float)kvalues_iq4nl[c]);
            if (dd < bd) bd = dd, best = c;
          }
          q[i] = (uint8_t)best;
        }
        for (int j = 0; j < 16; ++j) put<uint8_t>(o, (uint8_t)(q[j] | (q[j + 16] << 4)));
      }
    }
  } else {
    throw std::runtime_error("encode: type " + std::to_string(type));
  }
  return o;
}

struct Kv { std::string key; uint32_t t, et; GgufValue v; };
struct Tensor {
  std::string name;
  std::vector<uint64_t> dims;  // (ne0 first)
  uint32_t type;
  std::vector<uint8_t> data;
};

struct Model {
  std::vector<Kv> kv;
  std::vector<Tensor> ts;
  void u32(const std::string& k, int64_t v) { kv.push_back({k, 4, 0, v}); }
  void f32(const std::string& k, double v) { kv.push_back({k, 6, 0, v}); }
  void b(const std::string& k, bool v) { kv.push_back({k, 7, 0, v}); }
  void str(const std::string& k, const std::string& v) { kv.push_back({k, 8, 0, v}); }
  void arr(const std::string& k, uint32_t et, const GgufValue& v) { kv.push_back({k, 9, et, v}); }
  // data in row-major [.., ne1, ne0]; shape outermost first
  void tensor(const std::string& name, const F& data, uint32_t type, std::vector<uint64_t> shape) {
    // (--ggml-org: its IQ4_NL file's types: every BF16 matrix Q8_0 where its rows are whole 32-blocks, the loop
    //  controller F32)
    if ((g_ggml_org & 1) && type == GGML_BF16 && name.find("_exps.") == std::string::npos)
      type = name.find("loop_") != std::string::npos ? (uint32_t)GGML_F32 : shape.back() % 32 == 0 ? (uint32_t)GGML_Q8_0 : type;
    // (--mats-type T: every BF16 2D matrix of type T where its rows are whole blocks of it, the rest BF16)
    if (g_mats_type != kNoType && type == GGML_BF16 && name.find("_exps.") == std::string::npos && shape.size() == 2) {
      int blk = 0, bytes = 0;
      ggml_block_info(g_mats_type, blk, bytes);
      if (blk > 0 && shape.back() % (uint64_t)blk == 0) type = g_mats_type;
    }
    Tensor t;
    t.name = name;
    t.dims.assign(shape.rbegin(), shape.rend());
    t.type = type;
    t.data = encode(data, type);
    ts.push_back(std::move(t));
  }
  void write(const std::string& path) {
    zpack::Writer w(path);
    w.put(std::string("GGUF").data(), 4);
    w.put<uint32_t>(3);
    w.put<uint64_t>(ts.size());
    w.put<uint64_t>(kv.size());
    for (auto& k : kv) zpack::put_kv(w, k.key, k.t, k.et, k.v);
    uint64_t off = 0;
    for (auto& t : ts) {
      w.str(t.name);
      w.put<uint32_t>((uint32_t)t.dims.size());
      for (uint64_t d : t.dims) w.put<uint64_t>(d);
      w.put<uint32_t>(t.type);
      w.put<uint64_t>(off);
      off = (off + t.data.size() + 31) / 32 * 32;
    }
    w.pad(32);
    for (auto& t : ts) {
      w.put(t.data.data(), t.data.size());
      w.pad(32);
    }
    w.close();
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    printf("usage: make_tiny_inkling2 OUT.gguf [--seed N] [--no-loop] [--nvfp4-equal] [--ggml-org] [--dspark DRAFT.gguf] [--all-experts] [--passes N] [--scale S] [--byte-vocab] [--exps-type T] [--mats-type T]\n");
    return argc < 2 ? 1 : 0;
  }
  const std::string out = argv[1];
  std::string dspark_out;  // (--dspark FILE: a tiny DSpark drafter for it, the real one's names and keys)
  uint64_t seed = 2;
  bool loop = true;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
    else if (a == "--no-loop") loop = false;
    else if (a == "--nvfp4-equal") g_nvfp4_equal = true;
    else if (a == "--ggml-org") g_ggml_org = 3;
    else if (a == "--ggml-org-mats") g_ggml_org = 1;  // (tests: only its 2D matrices' types)
    else if (a == "--ggml-org-exps") g_ggml_org = 2;  // (only its experts' types)
    else if (a == "--all-experts") NU = NE;
    else if (a == "--scale" && i + 1 < argc) g_init = std::max(0.01, atof(argv[++i]));
    else if (a == "--dspark" && i + 1 < argc) dspark_out = argv[++i];
    else if (a == "--byte-vocab") g_byte_vocab = true;
    else if (a == "--mats-type" && i + 1 < argc) {
      g_mats_type = type_by_name(argv[++i]);
      if (g_mats_type == kNoType) { fprintf(stderr, "--mats-type: a GGUF type name\n"); return 2; }
    }
    else if (a == "--exps-type" && i + 1 < argc) {
      const std::string t = argv[++i];
      g_exps_type = type_by_name(t);
      if (g_exps_type == kNoType) { fprintf(stderr, "--exps-type: a GGUF type name (q4_k, iq2_xxs, mxfp4, f16, ...)\n"); return 2; }
      FF = 256;  // (superblocks of 256 along the down rows too)
    }
    else if (a == "--passes" && i + 1 < argc) PASSES = std::max(1, atoi(argv[++i]));  // (the loop's count)
    else { fprintf(stderr, "make_tiny_inkling2: unknown option %s\n", a.c_str()); return 2; }
  }
  try {
    Rng rng(seed);
    Model w;
    const std::string ARCH = loop ? "inkling2_loop" : "inkling2", p = ARCH + ".";
    auto mat = [&](int n_out, int n_in, double scale = 1.0) {  // [n_out][n_in], outputs of about unit size
      F v((size_t)n_out * n_in);
      const double s = scale * g_init / std::sqrt((double)n_in);  // (--scale: every matrix's size times S)
      for (auto& x : v) x = (float)(rng.normal() * s);
      return v;
    };
    auto normw = [&](size_t n, double c = 1.0, double sd = 0.1) {
      F v(n);
      for (auto& x : v) x = (float)(c + sd * rng.normal());
      return v;
    };
    auto randn = [&](size_t n, double scale = 1.0) {
      F v(n);
      for (auto& x : v) x = (float)(scale * rng.normal());
      return v;
    };
    typedef std::vector<uint64_t> S;
    auto is_global = [](int il) { return std::find(GLOBAL.begin(), GLOBAL.end(), il) != GLOBAL.end(); };

    w.str("general.architecture", ARCH);
    w.str("general.name", "Inkling 2 tiny (random weights, inferzero test)");
    w.u32("general.alignment", 32);
    w.u32(p + "block_count", NL);
    w.u32(p + "context_length", 4096);
    w.u32(p + "embedding_length", E);
    w.u32(p + "feed_forward_length", FF);
    w.arr(p + "attention.head_count", 5, std::vector<int64_t>(NL, H));
    w.arr(p + "attention.head_count_kv", 5, std::vector<int64_t>(NL, HKV));
    w.f32(p + "attention.layer_norm_rms_epsilon", 1e-6);
    w.u32(p + "expert_count", NE);
    w.u32(p + "expert_used_count", NU);
    w.u32(p + "attention.key_length", D);
    w.u32(p + "attention.value_length", D);
    if (loop) {
      w.u32(p + "loop.count", PASSES);
      w.u32(p + "loop.start", LSTART);
      w.u32(p + "loop.end", LEND);
      w.u32(p + "loop.gate_rank", RANK);
      w.b(p + "loop.varying", false);
    }
    w.u32(p + "vocab_size", NVP);
    w.u32(p + "vocab_size_unpadded", NV);
    w.u32(p + "expert_feed_forward_length", FF);
    w.u32(p + "attention.key_length_swa", D);
    w.u32(p + "attention.value_length_swa", D);
    w.u32(p + "rope.dimension_count", 0);
    w.u32(p + "attention.sliding_window", SWA);
    std::vector<int64_t> pattern(NL);
    for (int il = 0; il < NL; ++il) pattern[il] = is_global(il) ? 0 : 1;
    w.arr(p + "attention.sliding_window_pattern", 7, pattern);
    w.u32(p + "shortconv.l_cache", NC);
    w.f32(p + "expert_weights_scale", 8.0);
    w.f32(p + "activation.situ_beta", 4.0);
    w.f32(p + "activation.situ_linear_beta", 25.0);
    w.u32(p + "attn_res.block_size", BLOCK);
    w.f32(p + "final_logit_softcapping", 30.0);
    w.u32(p + "attention.relative_length", NREL);
    w.u32(p + "attention.relative_extent", EXT);
    w.f32(p + "attention.qk_clip", 22.62742);
    w.f32(p + "attention.relative_clip", 8.0);
    w.u32(p + "attention.log_scale_floor", 8);
    w.f32(p + "attention.log_scale_alpha", 0.1);

    // a token list so that the tokenizer loads (prompts are fed as ids): the real file's kind of specials at the end
    std::vector<std::string> toks;
    std::vector<int64_t> ttype;
    const char* specials[] = {"<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>",
                              "<|message_tool|>", "<|content_text|>", "<|content_model_end_sampling|>",
                              "<|content_thinking|>", "<|end_message|>", "<|content_invoke_tool_json|>"};
    const int nsp = (int)(sizeof(specials) / sizeof(specials[0]));
    // (--byte-vocab: the first 256 named as GPT-2's byte-level characters, as src/tokenizer.h maps them: any text
    //  encodes byte by byte and decodes back; the weights are the same)
    for (int i = 0, n = 0; i < NV - nsp; ++i) {
      std::string s = "t" + std::to_string(i);
      if (g_byte_vocab && i < 256) {
        const bool pr = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
        const uint32_t cp = pr ? (uint32_t)i : 256u + (uint32_t)n++;
        s.clear();
        if (cp < 0x80) s.push_back((char)cp);
        else { s.push_back((char)(0xC0 | (cp >> 6))); s.push_back((char)(0x80 | (cp & 0x3F))); }
      }
      toks.push_back(s);
      ttype.push_back(1);
    }
    for (int i = 0; i < nsp; ++i) { toks.push_back(specials[i]); ttype.push_back(3); }
    w.str("tokenizer.ggml.model", "gpt2");
    w.str("tokenizer.ggml.pre", "inkling2");
    w.arr("tokenizer.ggml.tokens", 8, toks);
    w.arr("tokenizer.ggml.token_type", 5, ttype);
    w.arr("tokenizer.ggml.merges", 8, std::vector<std::string>{});
    w.u32("tokenizer.ggml.eos_token_id", NV - nsp + 6);  // (<|content_model_end_sampling|>, as 200006)
    w.u32("tokenizer.ggml.padding_token_id", NV - nsp);
    w.b("tokenizer.ggml.add_bos_token", false);

    const uint32_t F32 = GGML_F32, BF16 = GGML_BF16, NVFP4 = GGML_NVFP4;
    const uint64_t uE = E;
    w.tensor("output_res_query.weight", randn(E, 0.5), F32, S{uE});
    w.tensor("token_embd.weight", randn((size_t)NVP * E), BF16, S{(uint64_t)NVP, uE});
    w.tensor("token_embd_norm.weight", normw(E), F32, S{uE});
    w.tensor("output_norm.weight", normw(E), F32, S{uE});
    w.tensor("output.weight", mat(NVP, E, 1.5), BF16, S{(uint64_t)NVP, uE});
    for (int il = 0; il < NL; ++il) {
      const std::string b = "blk." + std::to_string(il) + ".";
      const bool glob = is_global(il), swa = !glob;
      const int ext = swa ? SWA : EXT, KG = HKV * D;
      if (il > 0) {
        const int nq = glob ? 3 : 1;
        w.tensor(b + "attn_res_query.weight", randn((size_t)nq * E, 0.5), F32, nq == 1 ? S{uE} : S{(uint64_t)nq, uE});
      }
      w.tensor(b + "ffn_res_query.weight", randn(E, 0.5), F32, S{uE});
      w.tensor(b + "attn_sink_proj.weight", mat(H, E, 0.5), BF16, S{(uint64_t)H, uE});
      w.tensor(b + "attn_gate.weight", mat(H * D, E), BF16, S{(uint64_t)H * D, uE});
      w.tensor(b + "attn_k_conv.weight", randn((size_t)KG * NC, 0.3), F32, S{(uint64_t)KG, (uint64_t)NC});
      w.tensor(b + "attn_q_clip.weight", normw((size_t)H * D), F32, S{(uint64_t)H, (uint64_t)D});
      w.tensor(b + "attn_rel_clip.weight", normw((size_t)H * NREL), F32, S{(uint64_t)H, (uint64_t)NREL});
      w.tensor(b + "attn_rel_bias.weight", randn((size_t)ext * NREL, 0.5), F32, S{(uint64_t)ext, (uint64_t)NREL});
      w.tensor(b + "attn_v_conv.weight", randn((size_t)KG * NC, 0.3), F32, S{(uint64_t)KG, (uint64_t)NC});
      if (glob) {
        w.tensor(b + "attn_k_conv_proj.weight", mat(NC * HKV, E, 0.3), BF16, S{(uint64_t)NC * HKV, uE});
        w.tensor(b + "attn_v_conv_proj.weight", mat(NC * HKV, E, 0.3), BF16, S{(uint64_t)NC * HKV, uE});
      }
      // (q scaled up so that the L2 clip of 22.6 bites on some rows)
      w.tensor(b + "attn_k.weight", mat(KG, E, 2.0), BF16, S{(uint64_t)KG, uE});
      w.tensor(b + "attn_output.weight", mat(E, H * D), BF16, S{uE, (uint64_t)H * D});
      w.tensor(b + "attn_q.weight", mat(H * D, E, 2.5), BF16, S{(uint64_t)H * D, uE});
      w.tensor(b + "attn_rel.weight", mat(NREL * H, E, 3.0), BF16, S{(uint64_t)NREL * H, uE});
      w.tensor(b + "attn_v.weight", mat(KG, E), BF16, S{(uint64_t)KG, uE});
      if (swa) {
        w.tensor(b + "attn_norm.weight", normw(E), F32, S{uE});
      } else {
        w.tensor(b + "attn_q_inp_norm.weight", normw(E), F32, S{uE});
        w.tensor(b + "attn_k_inp_norm.weight", normw(E), F32, S{uE});
        w.tensor(b + "attn_v_inp_norm.weight", normw(E), F32, S{uE});
      }
      w.tensor(b + "attn_post_conv.weight", randn((size_t)E * NC, 0.3), F32, S{uE, (uint64_t)NC});
      // (--ggml-org: IQ4_NL where the original has NVFP4, Q8_0 where it has MXFP8, no expert scales)
      const uint32_t XT = (g_ggml_org & 2) ? (il < NVFP4_LAYERS ? (uint32_t)GGML_IQ4_NL : (uint32_t)GGML_Q8_0)
                                     : g_exps_type != kNoType ? g_exps_type : il < NVFP4_LAYERS ? NVFP4 : BF16;
      const uint64_t nE = NE;
      w.tensor(b + "ffn_gate_up_exps.weight", randn((size_t)NE * 2 * FF * E, 3 / std::sqrt((double)E)), XT,
               S{nE, (uint64_t)2 * FF, uE});
      if (XT == NVFP4) w.tensor(b + "ffn_gate_up_exps.scale", normw(NE, 1.0, 0.2), F32, S{nE});
      w.tensor(b + "ffn_down_exps.weight", randn((size_t)NE * E * FF, 1 / std::sqrt((double)FF)), XT,
               S{nE, uE, (uint64_t)FF});
      if (XT == NVFP4) w.tensor(b + "ffn_down_exps.scale", normw(NE, 1.0, 0.2), F32, S{nE});
      w.tensor(b + "exp_probs_b.bias", randn(NE, 0.1), F32, S{nE});
      w.tensor(b + "ffn_gate_inp.weight", mat(NE, E, 4.0), F32, S{nE, uE});
      w.tensor(b + "ffn_norm.weight", normw(E), F32, S{uE});
      w.tensor(b + "ffn_post_conv.weight", randn((size_t)E * NC, 0.3), F32, S{uE, (uint64_t)NC});
      if (loop && il >= LSTART && il < LEND) {
        w.tensor(b + "loop_msg_query.weight", randn((size_t)4 * E, 0.5), F32, S{4, uE});
        for (const char* c : {"q", "k", "v", "ffn"}) {
          const std::string lp = b + "loop_" + c + "_ctrl_";
          w.tensor(lp + "down.weight", mat(RANK, 2 * E), BF16, S{(uint64_t)RANK, (uint64_t)2 * E});
          w.tensor(lp + "forget.weight", mat(E, RANK, 0.5), BF16, S{uE, (uint64_t)RANK});
          w.tensor(lp + "gain.weight", mat(E, RANK, 0.5), BF16, S{uE, (uint64_t)RANK});
        }
      }
    }
    w.write(out);
    printf("%s: %s, %zu tensors, %zu keys\n", out.c_str(), ARCH.c_str(), w.ts.size(), w.kv.size());
    if (!dspark_out.empty()) {
      // A tiny DSpark drafter for this model (I2 step 6, 2026-11-24), as unsloth's dspark-Inkling-2-Mini-Q8_0.gguf lays
      // its drafter out: arch dflash, 3 layers (sliding, global, sliding), a block of 8, the Markov head of rank DR;
      // no token embedding and no head (it borrows the target's). Taps: 4 of the target's layer visits, the last one
      // their count (the head's input). Random weights: its drafts are tests of the machinery, not good guesses
      const int DL = 3, DH = 2, DHKV = 1, DF = 512, DR = 16, DB = 8, DW = 16, KG = DHKV * D, NVIS = loop ? LSTART + (LEND - LSTART) * PASSES + (NL - LEND) : NL;
      Model d;
      const std::string q = "dflash.";
      d.str("general.architecture", "dflash");
      d.str("general.name", "Inkling 2 tiny DSpark drafter (random weights, inferzero test)");
      d.u32("general.alignment", 32);
      d.u32(q + "block_count", DL);
      d.u32(q + "context_length", 4096);
      d.u32(q + "embedding_length", E);
      d.u32(q + "feed_forward_length", DF);
      d.u32(q + "attention.head_count", DH);
      d.u32(q + "attention.head_count_kv", DHKV);
      d.f32(q + "rope.freq_base", 1e6);
      d.f32(q + "attention.layer_norm_rms_epsilon", 1e-6);
      d.u32(q + "attention.key_length", D);
      d.u32(q + "attention.value_length", D);
      d.u32(q + "vocab_size", NVP);
      d.u32(q + "vocab_size_unpadded", NV);
      d.u32(q + "block_size", DB);
      d.arr(q + "target_layers", 5, std::vector<int64_t>{1, NVIS * 3 / 8, NVIS * 11 / 16, NVIS});
      d.u32(q + "dspark.markov_rank", DR);
      d.b(q + "sample_from_anchor", false);
      d.b(q + "has_confidence_head", false);
      d.b(q + "attention.causal", false);
      d.u32(q + "attention.sliding_window", DW);
      d.arr(q + "attention.sliding_window_pattern", 5, std::vector<int64_t>{1, 0, 1});
      d.u32(q + "rope.dimension_count", D);
      d.f32(q + "final_logit_softcapping", 30.0);
      const uint32_t Q8 = GGML_Q8_0;
      d.tensor("dspark_fuse.weight", randn((size_t)(DL + 1) * 4, 1.0), F32, S{(uint64_t)DL + 1, 4});
      d.tensor("dspark_hidden_out.bias", randn(E, 0.1), F32, S{uE});
      d.tensor("dspark_hidden_out.weight", mat(E, DR), BF16, S{uE, (uint64_t)DR});
      d.tensor("dspark_norm.weight", normw(E), F32, S{uE});
      d.tensor("joint_proj.bias", randn((size_t)3 * DR, 0.1), F32, S{(uint64_t)3 * DR});
      d.tensor("joint_proj.weight", mat(3 * DR, 2 * (DR + E)), BF16, S{(uint64_t)3 * DR, (uint64_t)2 * (DR + E)});
      d.tensor("markov_w1.weight", randn((size_t)NVP * DR, 1.0), BF16, S{(uint64_t)NVP, (uint64_t)DR});
      d.tensor("mask_embd.weight", randn(E, 1.0), F32, S{uE});
      d.tensor("output_norm.weight", normw(E), F32, S{uE});
      for (int il = 0; il < DL; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        d.tensor(b + "attn_k.weight", mat(KG, E), Q8, S{(uint64_t)KG, uE});
        d.tensor(b + "attn_k_ctx.weight", mat(KG, E), Q8, S{(uint64_t)KG, uE});
        d.tensor(b + "attn_k_ctx_norm.weight", normw(D), F32, S{(uint64_t)D});
        d.tensor(b + "attn_k_norm.weight", normw(D), F32, S{(uint64_t)D});
        d.tensor(b + "attn_norm.weight", normw(E), F32, S{uE});
        d.tensor(b + "attn_output.weight", mat(E, DH * D), Q8, S{uE, (uint64_t)DH * D});
        d.tensor(b + "attn_q.weight", mat(DH * D, E), Q8, S{(uint64_t)DH * D, uE});
        d.tensor(b + "attn_q_norm.weight", normw(D), F32, S{(uint64_t)D});
        d.tensor(b + "attn_v.weight", mat(KG, E), Q8, S{(uint64_t)KG, uE});
        d.tensor(b + "attn_v_ctx.weight", mat(KG, E), Q8, S{(uint64_t)KG, uE});
        d.tensor(b + "dflare_norm.weight", normw(E), F32, S{uE});
        d.tensor(b + "ffn_down.weight", mat(E, DF), Q8, S{uE, (uint64_t)DF});
        d.tensor(b + "ffn_gate.weight", mat(DF, E), Q8, S{(uint64_t)DF, uE});
        d.tensor(b + "ffn_norm.weight", normw(E), F32, S{uE});
        d.tensor(b + "ffn_up.weight", mat(DF, E), Q8, S{(uint64_t)DF, uE});
      }
      d.write(dspark_out);
      printf("%s: dflash (DSpark), %zu tensors, %zu keys\n", dspark_out.c_str(), d.ts.size(), d.kv.size());
    }
  } catch (const std::exception& e) {
    fprintf(stderr, "make_tiny_inkling2: %s\n", e.what());
    return 1;
  }
  return 0;
}
