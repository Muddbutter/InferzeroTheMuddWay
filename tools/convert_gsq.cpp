// Converts the ISTA-DASLab GSQ checkpoint of Qwen3.6-35B-A3B (compressed-tensors "pack-quantized": 2-bit routed
// experts, int8 attention / DeltaNet / shared experts / lm_head) into a GGUF for inferzero (arch "qwen35moe").
// The C++ port of tools/convert_gsq.py (removed since): the same GGUF byte for byte, with no Python
// (setup -Models 35b-gsq).
//
//   int8 matrices (group 128, or per channel for lm_head) -> Q8_0: the same int8 values, the scale per 32 as f16
//   2-bit experts (q in {-2,-1,0,1} x bf16 scale per 128 inputs) -> the layout of kernels/moe.cl:
//       blk.N.ffn_{gate,up,down}_exps.weight  I32  [N*K/16, n_expert]   codes (q + 2), see repack_codes()
//       blk.N.ffn_{gate,up,down}_exps.scale   BF16 [N*K/128, n_expert]  scales, see repack_scales()
//   tensor transforms as llama.cpp's convert_hf_to_gguf.py (Qwen3_5MoeTextModel): norm weights + 1 (except the
//   DeltaNet gated norm), A = -exp(A_log), DeltaNet V heads reordered from grouped to tiled order.
// The vision tower and the MTP block (which this checkpoint ships without its routed experts) are not converted.
// Tokenizer metadata is copied from a GGUF with the same tokenizer (--vocab-from) after checking it against
// tokenizer.json; the chat template comes from the checkpoint.
//
// usage: convert_gsq <snapshot dir> <out.gguf> --vocab-from <qwen3.5/3.6 gguf> [--layers N]
// (written to <out.gguf>.part, then renamed over <out.gguf>, as the Python script did: setup converts again over a
//  file the server no longer reads)
#include "../src/log.h"  // (zlog: first, as gguf_pack.h's headers use it; build.ps1 force-includes it, g++ does not)
#include "../zero/gguf_pack.h"  // (GgufFile, zpack::Writer / put_kv)
#include "../src/json.h"
#include "../src/quants.h"  // (f32_to_fp16: round to nearest even, as NumPy's astype(float16))
#include <chrono>
#include <cmath>
#include <immintrin.h>  // (_mm_fmadd_ss: build.ps1 /arch:AVX2, build.sh -mfma)
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gsq {

using Bytes = std::vector<uint8_t>;
[[noreturn]] inline void fail(const std::string& msg) { throw std::runtime_error(msg); }

// ---------------------------------------------------------------- files
// (paths as UTF-8 strings: the programs run with the UTF-8 code page, tools/utf8.manifest, so a narrow path holds any
//  characters on Windows as on Linux)
inline std::string join(const std::string& dir, const std::string& name) {
  return (std::filesystem::path(dir) / std::filesystem::path(name)).string();
}
inline std::string read_file(const std::string& path) {
  FILE* f = plat::fopen_path(std::filesystem::path(path), "rb");
  if (!f) fail("cannot open " + path);
  std::string s;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  const bool err = ferror(f) != 0;
  fclose(f);
  if (err) fail("cannot read " + path);
  return s;
}
inline Json read_json(const std::string& path) {
  try {
    return Json::parse(read_file(path));
  } catch (const std::exception& e) {
    fail(path + ": " + e.what());
  }
}
// (Python's open(..., encoding="utf-8").read(): strict UTF-8, "\r\n" and a lone "\r" read as "\n")
inline std::string read_text_utf8(const std::string& path) {
  const std::string s = read_file(path);
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (!n || i + n > s.size()) fail(path + ": not UTF-8");
    uint32_t cp = n == 1 ? c : c & (0x7F >> n);
    for (int k = 1; k < n; ++k) {
      const unsigned char d = (unsigned char)s[i + k];
      if ((d & 0xC0) != 0x80) fail(path + ": not UTF-8");
      cp = cp << 6 | (d & 0x3F);
    }
    static const uint32_t lo[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < lo[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp < 0xE000)) fail(path + ": not UTF-8");
    i += n;
  }
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\r') { o.push_back(s[i]); continue; }
    o.push_back('\n');
    if (i + 1 < s.size() && s[i + 1] == '\n') ++i;
  }
  return o;
}

// ---------------------------------------------------------------- safetensors
struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t nbytes = 0, esize = 0;
  int64_t dim(size_t i) const { return i < shape.size() ? shape[i] : 1; }
};

class Checkpoint {
 public:
  explicit Checkpoint(const std::string& snap) : snap_(snap) {
    const Json idx = read_json(join(snap, "model.safetensors.index.json"));
    if (!idx["weight_map"].is_object()) fail("model.safetensors.index.json: no weight_map");
    for (const auto& [k, v] : idx["weight_map"].entries()) map_[k] = v.str();
  }
  Tensor get(const std::string& name) {
    auto it = map_.find(name);
    if (it == map_.end()) fail("missing tensor " + name);
    Shard& sh = shard(join(snap_, it->second));
    auto hm = sh.index.find(name);
    if (hm == sh.index.end() || !hm->second->is_object()) fail(it->second + ": no tensor " + name);
    const Json& m = *hm->second;
    Tensor t;
    t.name = name;
    t.dtype = m["dtype"].str();
    static const std::map<std::string, size_t> sizes = {{"BF16", 2}, {"F32", 4}, {"I32", 4}, {"I64", 8}};
    auto ds = sizes.find(t.dtype);
    if (ds == sizes.end()) fail(name + ": unsupported dtype " + t.dtype);
    t.esize = ds->second;
    const Json& off = m["data_offsets"];
    const int64_t a = off[0].as_int(-1), b = off[1].as_int(-1);
    if (a < 0 || b < a || (uint64_t)b > sh.size - sh.data0) fail(name + ": bad data_offsets");
    t.data = sh.base + sh.data0 + a;
    t.nbytes = (size_t)(b - a);
    uint64_t ne = 1;
    for (const Json& d : m["shape"].items()) {
      const int64_t v = d.as_int(-1);
      if (v < 0) fail(name + ": bad shape");
      t.shape.push_back(v);
      ne *= (uint64_t)v;
    }
    if (ne * t.esize != t.nbytes) fail(name + ": shape does not match its bytes");
    return t;
  }

 private:
  struct Shard {
    plat::FileMap fm;
    const uint8_t* base = nullptr;
    uint64_t size = 0, data0 = 0;
    Json hdr;
    std::unordered_map<std::string, const Json*> index;  // (name -> its entry: the header lists thousands)
  };
  std::string snap_;
  std::unordered_map<std::string, std::string> map_;
  std::unordered_map<std::string, std::unique_ptr<Shard>> shards_;
  Shard& shard(const std::string& fn) {
    auto& p = shards_[fn];
    if (p) return *p;
    auto s = std::make_unique<Shard>();
    if (!s->fm.open(fn.c_str()) || !s->fm.size(s->size) || s->size < 8) fail("cannot open " + fn);
    if (s->fm.map() != 0) fail("cannot map " + fn);
    s->base = s->fm.data();
    uint64_t n;
    memcpy(&n, s->base, 8);
    if (n > s->size - 8) fail(fn + ": bad safetensors header");
    try {
      s->hdr = Json::parse(std::string((const char*)s->base + 8, (size_t)n));
    } catch (const std::exception& e) {
      fail(fn + ": " + e.what());
    }
    s->data0 = 8 + n;
    for (const auto& [k, v] : s->hdr.entries()) s->index[k] = &v;
    p = std::move(s);
    return *p;
  }
};

// a tensor as f32 (BF16 widened, F32 as stored, integers converted: Python's get(f32=True).astype(np.float32))
inline std::vector<float> to_f32(const Tensor& t) {
  const size_t n = t.nbytes / t.esize;
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* p = t.data + i * t.esize;
    if (t.dtype == "BF16") { uint16_t h; memcpy(&h, p, 2); v[i] = bf16_to_f32(h); }
    else if (t.dtype == "F32") memcpy(&v[i], p, 4);
    else if (t.dtype == "I32") { int32_t x; memcpy(&x, p, 4); v[i] = (float)x; }
    else { int64_t x; memcpy(&x, p, 8); v[i] = (float)x; }
  }
  return v;
}
inline Bytes f32_bytes(const std::vector<float>& v) {
  Bytes b(v.size() * 4);
  if (!v.empty()) memcpy(b.data(), v.data(), b.size());
  return b;
}

// ---------------------------------------------------------------- transforms
// V heads from grouped (by K head) to tiled order along a dimension of D = nk * nvk * hd entries, each `inner` bytes,
// repeated `outer` times (llama.cpp _reorder_v_heads; Python reorder_v)
inline Bytes reorder_v(const uint8_t* x, size_t outer, size_t D, size_t inner, int64_t nk, int64_t nvk, int64_t hd) {
  if (nk <= 0 || nvk <= 0 || hd <= 0 || (uint64_t)(nk * nvk * hd) != D) fail("reorder_v: cannot reshape");
  Bytes o(outer * D * inner);
  const size_t blk = (size_t)hd * inner;
  for (size_t r = 0; r < outer; ++r)
    for (int64_t j = 0; j < nvk; ++j)
      for (int64_t i = 0; i < nk; ++i)
        memcpy(o.data() + (r * D + (size_t)(j * nk + i) * hd) * inner, x + (r * D + (size_t)(i * nvk + j) * hd) * inner, blk);
  return o;
}

// An int8 matrix [N][K] and its scales [N][G] as f32.
struct Int8Mat {
  int64_t N = 0, K = 0, G = 0;
  Bytes q;               // int8 values
  std::vector<float> s;  // scales
};
inline Int8Mat int8_codes(Checkpoint& ck, const std::string& p) {
  // (stored + 128, 4 per int32, LSB first)
  const Tensor tq = ck.get(p + ".weight_packed");
  Int8Mat m;
  m.N = tq.shape.empty() ? 1 : tq.shape[0];
  if (m.N <= 0 || tq.nbytes % (size_t)m.N) fail(tq.name + ": cannot reshape");
  m.K = (int64_t)(tq.nbytes / (size_t)m.N);
  m.q.resize(tq.nbytes);
  for (size_t i = 0; i < tq.nbytes; ++i) m.q[i] = (uint8_t)(tq.data[i] ^ 0x80);
  const Tensor ts = ck.get(p + ".weight_scale");
  if (ts.shape.size() != 2) fail(ts.name + ": expected a 2-d scale");
  if (ts.shape[0] != m.N) fail(ts.name + ": " + std::to_string(ts.shape[0]) + " rows, the matrix " + std::to_string(m.N));
  m.G = ts.shape[1];
  m.s = to_f32(ts);
  return m;
}
// Q8_0 blocks of an int8 matrix with group scales (the group a multiple of 32)
inline Bytes q8_0_bytes(const Int8Mat& m) {
  if (m.K % 32 || m.G <= 0) fail("q8_0: cannot reshape");
  const int64_t nb = m.K / 32, rep = nb / m.G;
  if (rep * m.G != nb) fail("q8_0: scale groups do not divide the row");
  Bytes o((size_t)(m.N * nb * 34));
  uint8_t* d = o.data();
  for (int64_t n = 0; n < m.N; ++n)
    for (int64_t j = 0; j < nb; ++j, d += 34) {
      const uint16_t h = f32_to_fp16(m.s[(size_t)(n * m.G + j / rep)]);
      memcpy(d, &h, 2);
      memcpy(d + 2, m.q.data() + n * m.K + j * 32, 32);
    }
  return o;
}

// Within each 32-bit word of 16 codes, code e goes to bit 8 * (e % 4) + 2 * (e // 4): then
// (word >> 2j) & 0x03030303 holds the codes 4j .. 4j+3 in its bytes 0..3 (four consecutive inputs, one DP4a operand).
// 2-bit codes [N][K/16] (LSB-first, 16 per word) -> words [N/16][K/128][8][16]: column group, 128-input group, word,
// lane (one output column per lane; a sub-group block read of 8 words gives each lane its column).
inline void repack_codes(const Tensor& t, Bytes& out) {
  if (t.shape.size() != 2) fail(t.name + ": expected a 2-d tensor");
  const int64_t N = t.shape[0];
  if (N <= 0 || t.nbytes % ((size_t)N * 4)) fail(t.name + ": cannot view as uint32");
  const int64_t W = (int64_t)(t.nbytes / 4 / (size_t)N);
  if (N % 16 || W % 8) fail(t.name + ": cannot reshape");
  const size_t base = out.size();
  out.resize(base + t.nbytes);
  uint8_t* o = out.data() + base;
  for (int64_t n16 = 0; n16 < N / 16; ++n16)
    for (int64_t w8 = 0; w8 < W / 8; ++w8)
      for (int wi = 0; wi < 8; ++wi)
        for (int lane = 0; lane < 16; ++lane) {
          uint32_t u, r = 0;
          memcpy(&u, t.data + ((size_t)(n16 * 16 + lane) * W + w8 * 8 + wi) * 4, 4);
          for (int e = 0; e < 16; ++e) r |= ((u >> (2 * e)) & 3u) << (8 * (e % 4) + 2 * (e / 4));
          memcpy(o, &r, 4);
          o += 4;
        }
}
// scales [N][G] (bf16 as stored) -> [N/16][G][16]
inline void repack_scales(const Tensor& t, Bytes& out) {
  if (t.shape.size() != 2) fail(t.name + ": expected a 2-d tensor");
  const int64_t N = t.shape[0], G = t.shape[1];
  if (N % 16) fail(t.name + ": cannot reshape");
  const size_t es = t.esize, base = out.size();
  out.resize(base + t.nbytes);
  uint8_t* o = out.data() + base;
  for (int64_t n16 = 0; n16 < N / 16; ++n16)
    for (int64_t g = 0; g < G; ++g)
      for (int lane = 0; lane < 16; ++lane, o += es) memcpy(o, t.data + ((size_t)(n16 * 16 + lane) * G + g) * es, es);
}

// NumPy's float32 exp (numpy/_core/src/umath/loops_exponent_log.dispatch.c.src, simd_exp_FLOAT: the AVX2 / AVX-512
// kernel np.exp runs on every x86-64 CPU with AVX2 + FMA, as the Python script's -np.exp(A_log) did), lane by lane.
// Not correctly rounded (up to 2.52 ulp): std::exp gives other bits for ~9% of the floats.
#ifdef _MSC_VER
#pragma float_control(precise, on, push)  // (build.ps1 uses /fp:fast: the rounding steps must stay as written)
#pragma fp_contract(off)
#endif
// (one rounding: the FMA instruction, as the NumPy kernel's _mm256_fmadd_ps; MSVC's std::fma for floats gave other bits)
inline float fmaf_hw(float a, float b, float c) {
  return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
inline float numpy_expf(float x) {
  const float xmax = 88.72283935546875f, xmin = -103.97208404541015625f;
  if (x != x) return std::numeric_limits<float>::quiet_NaN();
  if (x >= xmax) return std::numeric_limits<float>::infinity();
  if (x <= xmin) return 0.0f;
  // (round to nearest: + and - 1.5 * 2^23. NumPy's wheels are compiled with the multiply and the add contracted into
  //  one FMA: a product that rounds to a tie then rounds by its exact value; with a separate multiply 84 floats differ)
  const float magic = 0x1.800000p+23f;
  volatile float quadrant = fmaf_hw(x, 1.442695040888963407359924681001892137f, magic);
  quadrant = quadrant - magic;
  const float q = quadrant;
  float r = fmaf_hw(q, -6.93145752e-1f, x);
  r = fmaf_hw(q, -1.42860677e-6f, r);
  r = fmaf_hw(q, 0.0f, r);
  float num = fmaf_hw(5.082762527590693718096e-04f, r, 6.757896990527504603057e-03f);
  num = fmaf_hw(num, r, 5.114512081637298353406e-02f);
  num = fmaf_hw(num, r, 2.473615434895520810817e-01f);
  num = fmaf_hw(num, r, 7.257664613233124478488e-01f);
  num = fmaf_hw(num, r, 9.999999999980870924916e-01f);
  float den = fmaf_hw(2.159509375685829852307e-02f, r, -2.742335390411667452936e-01f);
  den = fmaf_hw(den, r, 1.000000000000000000000e+00f);
  float poly = num / den;
  // poly * 2^q by adding q to its exponent (fma_scalef_ps); a denormal result (q <= -125) through a division
  uint32_t pb;
  memcpy(&pb, &poly, 4);
  if (q <= -125.0f) {
    const int32_t diff = (int32_t)-(q - -125.0f);
    const uint32_t two_pow = diff >= 0 && diff < 32 ? 1u << diff : 0u;  // (vpsllvd: a shift past 31 gives 0)
    pb += (uint32_t)(-125 * (1 << 23));
    memcpy(&poly, &pb, 4);
    return poly / (float)(int32_t)two_pow;
  }
  pb += (uint32_t)((int32_t)q * (1 << 23));
  memcpy(&poly, &pb, 4);
  return poly;
}
#ifdef _MSC_VER
#pragma float_control(pop)
#endif

// ---------------------------------------------------------------- GGUF metadata
// One metadata entry, written later in the order given (the stored GGUF type, as gguf_pack's put_kv writes it)
struct Kv {
  std::string key;
  uint32_t type, etype;
  GgufValue value;
};
inline Kv kv_str(const std::string& k, const std::string& s) { return {k, 8, 0, GgufValue(s)}; }
inline Kv kv_u32(const std::string& k, int64_t v) {
  if (v < 0 || v > 0xFFFFFFFFll) fail(k + ": " + std::to_string(v) + " does not fit a uint32");
  return {k, 4, 0, GgufValue(v)};
}
inline Kv kv_f32(const std::string& k, double v) { return {k, 6, 0, GgufValue(v)}; }
inline Kv kv_bool(const std::string& k, bool v) { return {k, 7, 0, GgufValue(v)}; }
inline Kv kv_arr_i32(const std::string& k, const std::vector<int64_t>& v) { return {k, 9, 5, GgufValue(v)}; }

// ---------------------------------------------------------------- the conversion
struct Plan {
  std::string name;
  std::vector<uint64_t> dims;  // (ne0 first)
  uint32_t type;
  uint64_t nbytes;
  std::function<Bytes()> make;
};

inline const Json& need(const Json& o, const char* k) {
  if (!o.has(k)) fail(std::string("config.json: no ") + k);
  return o[k];
}

inline int convert(const std::string& snapshot, const std::string& out_path, const std::string& vocab_from, int64_t layers) {
  const auto t_start = std::chrono::steady_clock::now();
  Checkpoint ck(snapshot);
  const Json cfg = read_json(join(snapshot, "config.json"));
  const Json& tc = cfg.has("text_config") ? cfg["text_config"] : cfg;
  auto I = [&](const char* k) { return need(tc, k).as_int(); };
  auto get_or = [&](const Json& o, const char* k, const Json& def) -> const Json& { return o.has(k) ? o[k] : def; };
  const int64_t E = I("hidden_size");
  int64_t nl = I("num_hidden_layers");
  if (layers) nl = layers;
  const int64_t nexp = I("num_experts"), nused = I("num_experts_per_tok"), ffe = I("moe_intermediate_size"),
                ffs = I("shared_expert_intermediate_size");
  const int64_t nh = I("num_attention_heads"), nkv = I("num_key_value_heads"), hd = I("head_dim");
  const int64_t lk = I("linear_key_head_dim"), lv = I("linear_value_head_dim"), nk = I("linear_num_key_heads"),
                nv = I("linear_num_value_heads");
  const Json empty_obj = Json::object();
  const Json& rp = tc["rope_parameters"].is_object() ? tc["rope_parameters"] : empty_obj;
  const double rope_theta = get_or(rp, "rope_theta", get_or(tc, "rope_theta", Json(1e7))).as_number();
  const double prf = get_or(rp, "partial_rotary_factor", get_or(tc, "partial_rotary_factor", Json(0.25))).as_number();
  std::vector<int64_t> sect;
  if (rp.has("mrope_section")) for (const Json& v : rp["mrope_section"].items()) sect.push_back(v.as_int());
  else sect = {11, 11, 10};
  sect.push_back(0);
  const int64_t vocab = I("vocab_size");
  const int64_t interval = get_or(tc, "full_attention_interval", Json(4)).as_int();
  const int64_t conv_k = I("linear_conv_kernel_dim");
  std::vector<std::string> types;
  for (const Json& v : tc["layer_types"].items()) types.push_back(v.str());
  if (types.empty())
    for (int64_t i = 0; i < nl; ++i) types.push_back((i + 1) % interval == 0 ? "full_attention" : "linear_attention");
  if ((int64_t)types.size() < nl) fail("config.json: layer_types lists " + std::to_string(types.size()) + " layers");
  if (nk <= 0) fail("config.json: linear_num_key_heads must be positive");
  const int64_t nvk = nv / nk;

  // ---- metadata
  std::vector<Kv> kv = {kv_str("general.architecture", "qwen35moe"),
                        kv_str("general.name", "Qwen3.6-35B-A3B GSQ 2-bit (ISTA-DASLab), inferzero"),
                        kv_str("inferzero.expert_format", "gsq2-v1")};
  const std::string a = "qwen35moe.";
  for (Kv& e : std::vector<Kv>{
           kv_u32(a + "block_count", nl), kv_u32(a + "context_length", get_or(tc, "max_position_embeddings", Json(262144)).as_int()),
           kv_u32(a + "embedding_length", E), kv_u32(a + "attention.head_count", nh),
           kv_u32(a + "attention.head_count_kv", nkv), kv_arr_i32(a + "rope.dimension_sections", sect),
           kv_f32(a + "rope.freq_base", rope_theta),
           kv_f32(a + "attention.layer_norm_rms_epsilon", get_or(tc, "rms_norm_eps", Json(1e-6)).as_number()),
           kv_u32(a + "expert_count", nexp), kv_u32(a + "expert_used_count", nused),
           kv_bool(a + "expert_weights_norm", true),
           kv_u32(a + "attention.key_length", hd), kv_u32(a + "attention.value_length", hd),
           kv_u32(a + "expert_feed_forward_length", ffe), kv_u32(a + "expert_shared_feed_forward_length", ffs),
           kv_u32(a + "ssm.conv_kernel", conv_k), kv_u32(a + "ssm.state_size", lk),
           kv_u32(a + "ssm.group_count", nk), kv_u32(a + "ssm.time_step_rank", nv),
           kv_u32(a + "ssm.inner_size", lv * nv), kv_u32(a + "full_attention_interval", interval),
           kv_u32(a + "rope.dimension_count", (int64_t)((double)hd * prf))})
    kv.push_back(std::move(e));

  GgufFile ref(vocab_from);
  const std::vector<std::string>* toks = ref.get_strs("tokenizer.ggml.tokens");
  if (!toks) fail("--vocab-from has no tokenizer.ggml.tokens");
  if ((int64_t)toks->size() != vocab)
    fail("--vocab-from has " + std::to_string(toks->size()) + " tokens, the model " + std::to_string(vocab));
  {  // check_vocab: tokenizer.json's ids (vocab, then added_tokens over them) against the GGUF's tokens
    const Json tj = read_json(join(snapshot, "tokenizer.json"));
    std::vector<int64_t> order;
    std::unordered_map<int64_t, std::string> idt;
    auto put = [&](int64_t id, const std::string& t) {
      if (idt.emplace(id, t).second) order.push_back(id);
      else idt[id] = t;
    };
    for (const auto& [t, id] : tj["model"]["vocab"].entries()) put(id.as_int(), t);
    for (const Json& at : tj["added_tokens"].items()) put(at["id"].as_int(), at["content"].str());
    std::vector<int64_t> bad;
    const int64_t nt = (int64_t)toks->size();
    for (int64_t id : order) {
      const int64_t j = id < 0 ? id + nt : id;  // (a negative id: Python's tokens[-1] counts from the end)
      if (id >= nt || j < 0 || (*toks)[(size_t)j] != idt[id]) bad.push_back(id);
    }
    if (!bad.empty()) {
      std::string first = "[";
      for (size_t i = 0; i < bad.size() && i < 5; ++i) first += (i ? ", " : "") + std::to_string(bad[i]);
      fail("tokenizer mismatch: " + std::to_string(bad.size()) + " of " + std::to_string(order.size()) +
           " tokens differ (first ids " + first + "])");
    }
    printf("tokenizer: %zu tokens match tokenizer.json\n", order.size());
    fflush(stdout);
  }
  {  // the GGUF's tokenizer.* entries in its order (a repeated key: its first place, its last value)
    std::unordered_set<std::string> seen;
    for (const std::string& k : ref.kv_order)
      if (k.rfind("tokenizer.", 0) == 0 && k != "tokenizer.chat_template" && seen.insert(k).second) {
        const auto& ty = ref.kv_types.at(k);
        kv.push_back({k, ty.first, ty.second, ref.kv.at(k)});
      }
  }
  const std::string tpl = join(snapshot, "chat_template.jinja");
  if (std::filesystem::exists(std::filesystem::path(tpl))) kv.push_back(kv_str("tokenizer.chat_template", read_text_utf8(tpl)));
  else if (ref.has("tokenizer.chat_template")) {
    const auto& ty = ref.kv_types.at("tokenizer.chat_template");
    kv.push_back({"tokenizer.chat_template", ty.first, ty.second, ref.kv.at("tokenizer.chat_template")});
  }

  // ---- tensor plan
  std::vector<Plan> plan;
  const std::string L = "model.language_model.";
  auto add = [&](const std::string& name, std::vector<uint64_t> dims, uint32_t type, int64_t nb, std::function<Bytes()> fn) {
    plan.push_back({name, std::move(dims), type, (uint64_t)nb, std::move(fn)});
  };
  using F32Fn = std::function<std::vector<float>(const Tensor&, std::vector<float>)>;
  auto f32 = [&](const std::string& src, F32Fn transform = nullptr, bool plus1 = false) {
    return [&ck, src, transform, plus1]() {
      const Tensor t = ck.get(src);
      std::vector<float> x = to_f32(t);
      if (plus1) for (float& v : x) v = v + 1.0f;
      if (transform) x = transform(t, std::move(x));
      return f32_bytes(x);
    };
  };
  using RowPerm = std::function<Bytes(const uint8_t*, size_t rows, size_t row_bytes)>;
  using ColBlocks = std::function<void(Int8Mat&)>;
  auto q8 = [&](const std::string& src, RowPerm row_perm = nullptr, ColBlocks col_blocks = nullptr) {
    return [&ck, src, row_perm, col_blocks]() {
      Int8Mat m = int8_codes(ck, src);
      if (row_perm) {
        m.q = row_perm(m.q.data(), (size_t)m.N, (size_t)m.K);
        const Bytes s = row_perm((const uint8_t*)m.s.data(), (size_t)m.N, (size_t)m.G * 4);
        memcpy(m.s.data(), s.data(), s.size());
      }
      if (col_blocks) col_blocks(m);
      return q8_0_bytes(m);
    };
  };
  auto q8_size = [](int64_t N, int64_t K) { return N * K / 32 * 34; };
  const uint64_t uE = (uint64_t)E, uvocab = (uint64_t)vocab;

  add("token_embd.weight", {uE, uvocab}, GGML_BF16, E * vocab * 2, [&ck, L]() {
    const Tensor t = ck.get(L + "embed_tokens.weight");
    return Bytes(t.data, t.data + t.nbytes);
  });
  add("output.weight", {uE, uvocab}, GGML_Q8_0, q8_size(vocab, E), [&ck]() { return q8_0_bytes(int8_codes(ck, "lm_head")); });
  add("output_norm.weight", {uE}, GGML_F32, E * 4, f32(L + "norm.weight", nullptr, true));

  const int64_t qk = 2 * lk * nk;  // q and k channels of the DeltaNet projection
  // (reorder along dim 0 of a [rows][row_bytes] array, the heads hd rows each)
  auto rows_v = [nk, nvk](const uint8_t* x, size_t rows, size_t row_bytes, int64_t h) {
    return reorder_v(x, 1, rows, row_bytes, nk, nvk, h);
  };
  RowPerm v_rows = [=](const uint8_t* x, size_t rows, size_t rb) {  // in_proj_qkv: reorder the V rows only
    if ((int64_t)rows < qk) fail("v_rows: cannot reshape");
    Bytes o(x, x + (size_t)qk * rb);
    const Bytes v = rows_v(x + (size_t)qk * rb, rows - (size_t)qk, rb, lv);
    o.insert(o.end(), v.begin(), v.end());
    return o;
  };
  RowPerm z_rows = [=](const uint8_t* x, size_t rows, size_t rb) { return rows_v(x, rows, rb, lv); };
  ColBlocks out_cols = [=](Int8Mat& m) {  // out_proj: reorder 128-column blocks (head_v_dim == group size)
    if (!(lv == 128 && m.G * 128 == m.K)) fail("out_proj: expected 128-column scale groups (linear_value_head_dim 128)");
    m.q = reorder_v(m.q.data(), (size_t)m.N, (size_t)m.K, 1, nk, nvk, lv);
    const Bytes s = reorder_v((const uint8_t*)m.s.data(), (size_t)m.N, (size_t)m.G, 4, nk, nvk, 1);
    memcpy(m.s.data(), s.data(), s.size());
  };
  auto rv1_vec = [=](const Tensor& t, std::vector<float> x) {  // reorder along dim 0, one row per head
    const size_t D = (size_t)t.dim(0), rb = D ? x.size() / D * 4 : 0;
    const Bytes o = rows_v((const uint8_t*)x.data(), D, rb, 1);
    memcpy(x.data(), o.data(), o.size());
    return x;
  };

  for (int64_t il = 0; il < nl; ++il) {
    const std::string p = L + "layers." + std::to_string(il) + ".", g = "blk." + std::to_string(il) + ".";
    add(g + "attn_norm.weight", {uE}, GGML_F32, E * 4, f32(p + "input_layernorm.weight", nullptr, true));
    add(g + "post_attention_norm.weight", {uE}, GGML_F32, E * 4, f32(p + "post_attention_layernorm.weight", nullptr, true));
    if (types[(size_t)il] == "linear_attention") {
      const std::string la = p + "linear_attn.";
      const int64_t C = 2 * lk * nk + lv * nv;
      add(g + "attn_qkv.weight", {uE, (uint64_t)C}, GGML_Q8_0, q8_size(C, E), q8(la + "in_proj_qkv", v_rows));
      add(g + "attn_gate.weight", {uE, (uint64_t)(lv * nv)}, GGML_Q8_0, q8_size(lv * nv, E), q8(la + "in_proj_z", z_rows));
      add(g + "ssm_out.weight", {(uint64_t)(lv * nv), uE}, GGML_Q8_0, q8_size(E, lv * nv), q8(la + "out_proj", nullptr, out_cols));
      add(g + "ssm_alpha.weight", {uE, (uint64_t)nv}, GGML_F32, nv * E * 4, f32(la + "in_proj_a.weight", rv1_vec));
      add(g + "ssm_beta.weight", {uE, (uint64_t)nv}, GGML_F32, nv * E * 4, f32(la + "in_proj_b.weight", rv1_vec));
      add(g + "ssm_a", {(uint64_t)nv}, GGML_F32, nv * 4, f32(la + "A_log", [=](const Tensor& t, std::vector<float> x) {
            for (float& v : x) v = -numpy_expf(v);
            return rv1_vec(t, std::move(x));
          }));
      add(g + "ssm_dt.bias", {(uint64_t)nv}, GGML_F32, nv * 4, f32(la + "dt_bias", rv1_vec));
      add(g + "ssm_conv1d.weight", {(uint64_t)conv_k, (uint64_t)C}, GGML_F32, C * conv_k * 4,
          f32(la + "conv1d.weight", [=](const Tensor& t, std::vector<float> x) {  // ([C, 1, k] as [C, k])
            const size_t rows = (size_t)t.dim(0);
            const Bytes o = v_rows((const uint8_t*)x.data(), rows, rows ? x.size() / rows * 4 : 0);
            if (o.size() != x.size() * 4) fail("conv1d: cannot reshape");
            memcpy(x.data(), o.data(), o.size());
            return x;
          }));
      add(g + "ssm_norm.weight", {(uint64_t)lv}, GGML_F32, lv * 4, f32(la + "norm.weight"));
    } else {
      const std::string sa = p + "self_attn.";
      add(g + "attn_q.weight", {uE, (uint64_t)(2 * nh * hd)}, GGML_Q8_0, q8_size(2 * nh * hd, E), q8(sa + "q_proj"));
      add(g + "attn_k.weight", {uE, (uint64_t)(nkv * hd)}, GGML_Q8_0, q8_size(nkv * hd, E), q8(sa + "k_proj"));
      add(g + "attn_v.weight", {uE, (uint64_t)(nkv * hd)}, GGML_Q8_0, q8_size(nkv * hd, E), q8(sa + "v_proj"));
      add(g + "attn_output.weight", {(uint64_t)(nh * hd), uE}, GGML_Q8_0, q8_size(E, nh * hd), q8(sa + "o_proj"));
      add(g + "attn_q_norm.weight", {(uint64_t)hd}, GGML_F32, hd * 4, f32(sa + "q_norm.weight", nullptr, true));
      add(g + "attn_k_norm.weight", {(uint64_t)hd}, GGML_F32, hd * 4, f32(sa + "k_norm.weight", nullptr, true));
    }
    const std::string m = p + "mlp.";
    add(g + "ffn_gate_inp.weight", {uE, (uint64_t)nexp}, GGML_F32, nexp * E * 4, f32(m + "gate.weight"));
    add(g + "ffn_gate_inp_shexp.weight", {uE}, GGML_F32, E * 4, f32(m + "shared_expert_gate.weight"));
    add(g + "ffn_gate_shexp.weight", {uE, (uint64_t)ffs}, GGML_Q8_0, q8_size(ffs, E), q8(m + "shared_expert.gate_proj"));
    add(g + "ffn_up_shexp.weight", {uE, (uint64_t)ffs}, GGML_Q8_0, q8_size(ffs, E), q8(m + "shared_expert.up_proj"));
    add(g + "ffn_down_shexp.weight", {(uint64_t)ffs, uE}, GGML_Q8_0, q8_size(E, ffs), q8(m + "shared_expert.down_proj"));
    const struct { const char* w; int64_t N, K; } mats[3] = {{"gate", ffe, E}, {"up", ffe, E}, {"down", E, ffe}};
    for (const auto& mt : mats) {
      const int64_t words = mt.N * mt.K / 16, groups = mt.N * mt.K / 128;
      const std::string w = mt.w;
      add(g + "ffn_" + w + "_exps.weight", {(uint64_t)words, (uint64_t)nexp}, GGML_I32, words * nexp * 4, [&ck, m, w, nexp]() {
        Bytes o;
        for (int64_t e = 0; e < nexp; ++e) repack_codes(ck.get(m + "experts." + std::to_string(e) + "." + w + "_proj.weight_packed"), o);
        return o;
      });
      add(g + "ffn_" + w + "_exps.scale", {(uint64_t)groups, (uint64_t)nexp}, GGML_BF16, groups * nexp * 2, [&ck, m, w, nexp]() {
        Bytes o;
        for (int64_t e = 0; e < nexp; ++e) repack_scales(ck.get(m + "experts." + std::to_string(e) + "." + w + "_proj.weight_scale"), o);
        return o;
      });
    }
  }

  // ---- write: header, KVs, tensor infos, then the data (each tensor aligned to 32)
  const uint64_t ALIGN = 32;
  const std::string tmp = out_path + ".part";
  auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
  try {
    zpack::Writer f(tmp);
    f.put<uint32_t>(0x46554747);  // "GGUF"
    f.put<uint32_t>(3);
    f.put<uint64_t>(plan.size());
    f.put<uint64_t>(kv.size());
    for (const Kv& e : kv) zpack::put_kv(f, e.key, e.type, e.etype, e.value);
    uint64_t off = 0;
    for (const Plan& t : plan) {
      f.str(t.name);
      f.put<uint32_t>((uint32_t)t.dims.size());
      for (uint64_t d : t.dims) f.put<uint64_t>(d);
      f.put<uint32_t>(t.type);
      f.put<uint64_t>(off);
      off += (t.nbytes + ALIGN - 1) / ALIGN * ALIGN;
    }
    f.pad(ALIGN);
    const uint64_t data0 = f.pos();
    for (Plan& t : plan) {
      Bytes b = t.make();
      if (b.size() != t.nbytes)
        fail(t.name + ": " + std::to_string(b.size()) + " bytes, expected " + std::to_string(t.nbytes));
      f.put(b.data(), b.size());
      f.pad(ALIGN);
      const std::string tail = "ffn_down_exps.scale";
      if (t.name.size() >= tail.size() && t.name.compare(t.name.size() - tail.size(), tail.size(), tail) == 0) {
        const size_t d1 = t.name.find('.');
        printf("  layer %s done (%.2f GiB, %.0f s)\n", t.name.substr(d1 + 1, t.name.find('.', d1 + 1) - d1 - 1).c_str(),
               f.pos() / 1073741824.0, seconds());
        fflush(stdout);
      }
    }
    if (f.pos() - data0 != off) fail("internal error: data size");
    f.close();
    if (!plat::replace_file(tmp.c_str(), out_path.c_str()))
      fail("cannot rename " + tmp + " to " + out_path + " (error " + std::to_string(plat::last_error()) + ")");
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path(tmp), ec);  // (no half model left behind)
    throw;
  }
  std::error_code ec;
  const uint64_t size = std::filesystem::file_size(std::filesystem::path(out_path), ec);
  printf("wrote %s: %zu tensors, %.2f GiB in %.0f s\n", out_path.c_str(), plan.size(), size / 1073741824.0, seconds());
  fflush(stdout);
  return 0;
}

}  // namespace gsq

#ifndef CONVERT_GSQ_NO_MAIN
int main(int argc, char** argv) {
  const char* usage = "usage: convert_gsq [-h] --vocab-from VOCAB_FROM [--layers LAYERS] snapshot out\n";
  std::vector<std::string> pos;
  std::string vocab_from;
  bool have_vocab = false;
  int64_t layers = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto value = [&](const std::string& opt) -> std::string {
      if (s.size() > opt.size() && s.compare(0, opt.size() + 1, opt + "=") == 0) return s.substr(opt.size() + 1);
      if (i + 1 >= argc) {
        zlog::eprintf("%sconvert_gsq: error: argument %s: expected one argument\n", usage, opt.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (s == "-h" || s == "--help") {
      printf("%s\n  snapshot                 the checkpoint's folder (config.json, tokenizer.json, *.safetensors)\n"
             "  out                      the GGUF to write\n"
             "  --vocab-from VOCAB_FROM  GGUF with the same tokenizer (Qwen3.5 / Qwen3.6)\n"
             "  --layers LAYERS          convert only the first N layers (testing)\n", usage);
      return 0;
    } else if (s == "--vocab-from" || s.rfind("--vocab-from=", 0) == 0) {
      vocab_from = value("--vocab-from");
      have_vocab = true;
    } else if (s == "--layers" || s.rfind("--layers=", 0) == 0) {
      const std::string v = value("--layers");
      char* end = nullptr;
      layers = strtoll(v.c_str(), &end, 10);
      if (v.empty() || *end || layers < 0) {
        zlog::eprintf("%sconvert_gsq: error: argument --layers: invalid value: '%s'\n", usage, v.c_str());
        return 2;
      }
    } else if (s.size() > 1 && s[0] == '-') {
      zlog::eprintf("%sconvert_gsq: error: unrecognized arguments: %s\n", usage, s.c_str());
      return 2;
    } else {
      pos.push_back(s);
    }
  }
  if (pos.size() > 2) {
    zlog::eprintf("%sconvert_gsq: error: unrecognized arguments: %s\n", usage, pos[2].c_str());
    return 2;
  }
  if (pos.size() < 2 || !have_vocab) {
    std::string miss;
    if (pos.size() < 1) miss += "snapshot, ";
    if (pos.size() < 2) miss += "out, ";
    if (!have_vocab) miss += "--vocab-from, ";
    miss.resize(miss.size() - 2);
    zlog::eprintf("%sconvert_gsq: error: the following arguments are required: %s\n", usage, miss.c_str());
    return 2;
  }
  try {
    return gsq::convert(pos[0], pos[1], vocab_from, layers);
  } catch (const std::exception& e) {
    zlog::eprintf("%s\n", e.what());
    return 1;
  }
}
#endif
