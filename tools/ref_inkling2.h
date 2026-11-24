// Inkling 2 (GGUF architectures inkling2 and inkling2_loop) for tools/ref_forward.cpp: the f32 reference forward pass
// of one sequence from position 0 (dequantized weights, f32 activations, double sums), mirroring llama.cpp's graph
// operation by operation (src/models/inkling2.cpp and inkling2-loop.cpp, master as of 2026-11-23; PR #32704 re-shares
// the loop's layers after the scale pass and changes nothing here). Included by ref_forward.cpp after its helpers (F,
// A, hp, matmul, vecf, rmsnorm). In: x the token embeddings (T x E); out: x the output norm's result (T x E), the
// output head's input (its softcap and its unpadded vocabulary are applied by ref_forward's head).
//
// The model, in llama.cpp's terms:
//  - layers run in "cache slots"; inkling2_loop runs the core [loop.start, loop.end) loop.count times, every pass from
//    the prelude's output, a later pass's layers getting "messages" from the previous pass (a mix of its blocks after
//    the first core one and its output) through low-rank controllers on their q / k / v / ffn inputs;
//  - the residual stream restarts at every attn_res.block_size-th layer, the earlier streams kept as "blocks", and a
//    layer's inputs are learned softmax mixes over the RMS-normed blocks and the current stream (attention over depth);
//  - attention without RoPE: q / k / r clipped to an L2 norm, a relative-position bias r . bias[distance] minus a
//    per-query sink logit, a zero sink in the softmax, K and V through short causal convolutions (global layers: with
//    input-dependent taps per head), q and r of the global layers times a log-length factor, V RMS-normed, the output
//    gated by sigmoid(gate(x)) and convolved again;
//  - MoE in every layer: sigmoid router + selection bias, top-k, weights normalized and scaled, gate and up fused, each
//    expert's matmul outputs times its f32 .scale, situ activation, the FFN output convolved.
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ref_inkling2 {

// a per-layer int key (an array, or one value for every layer)
inline int per_layer(const std::string& key, int il, int def = -1) {
  if (const auto* v = F->get_ints(A + "." + key)) return il < (int)v->size() ? (int)(*v)[il] : def;
  return hp(key, def);
}
inline float fkey(const std::string& key, double def) { return (float)F->get_float(A + "." + key, def); }
inline double sigm(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// llama's build_lora_mm: a 2D weight W may come with W.scale (one f32, shape {1}) multiplying its output, as an
// NVFP4-everywhere file would have (the Q8_K_XL has none on 2D weights). Activation scales (.input_scale, for FP8
// activations) are refused at the start (inkling2_layers).
// (REF_INK_Q8=1, a check of the engine: the activations of the matmuls the engine runs in int8 quantized as it does,
//  per 32 values with src/quants.h quantize_act_cpu, then taken back to floats; the weights as before. The engine's
//  rule (zero/engine_inkling.h ink_mat): int8 unless the router, N % 16 or K % 256. The engine's int8 drift then
//  leaves the comparison, and a stage that still parts shows a bug, not a rounding)
// (REF_INK_W8=1: a float matrix the engine keeps as Q8_0 rows (ink_mat: not the router, N % 16 == 0, K % 256 == 0)
//  multiplied with those rows, made by the engine's own requant_rows_q8_0; otherwise ref_forward's matmul)
inline void mm_rows(const std::string& name, const std::vector<float>& X, int T, std::vector<float>& Y, bool router) {
  static const bool w8 = getenv("REF_INK_W8") && atoi(getenv("REF_INK_W8")) != 0;
  const GgufTensor& t = F->get(name);
  const int K = (int)t.dims[0], N = t.dims.size() > 1 ? (int)t.dims[1] : 1;
  const bool flt = t.type == GGML_BF16 || t.type == GGML_F16 || t.type == GGML_F32;
  if (!(w8 && flt && !router && N % 16 == 0 && K % 256 == 0)) return matmul(name, X, T, Y);
  const size_t rb = t.nbytes / (size_t)N;
  Y.assign((size_t)T * N, 0.0f);
#pragma omp parallel
  {
    std::vector<float> w(K);
#pragma omp for schedule(dynamic, 16)
    for (int n = 0; n < N; ++n) {
      const std::vector<uint8_t> q = requant_rows_q8_0(t.type, t.data + (size_t)n * rb, rb, 1, K);
      dequant_row(GGML_Q8_0, q.data(), K, w.data());
      for (int i = 0; i < T; ++i) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += (double)w[k] * X[(size_t)i * K + k];
        Y[(size_t)i * N + n] = (float)acc;
      }
    }
  }
}
inline void mm(const std::string& name, const std::vector<float>& X, int T, std::vector<float>& Y) {
  static const bool q8 = getenv("REF_INK_Q8") && atoi(getenv("REF_INK_Q8")) != 0;
  const GgufTensor& wt = F->get(name);
  const int K = (int)wt.dims[0], N = wt.dims.size() > 1 ? (int)wt.dims[1] : 1;
  static const std::string kRouter = "ffn_gate_inp.weight";
  const bool router = name.size() >= kRouter.size() && name.compare(name.size() - kRouter.size(), kRouter.size(), kRouter) == 0;
  if (q8 && !router && N % 16 == 0 && K % 256 == 0) {
    std::vector<int8_t> aq((size_t)T * K);
    std::vector<float> asc((size_t)(K / 32) * T), xq((size_t)T * K);
    std::vector<uint16_t> ab((size_t)(K / 16) * T);
    quantize_act_cpu(X.data(), T, K, aq.data(), asc.data(), ab.data(), true);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < K; ++k) xq[(size_t)t * K + k] = (float)aq[(size_t)t * K + k] * asc[(size_t)(k / 32) * T + t];
    mm_rows(name, xq, T, Y, router);
  } else {
    mm_rows(name, X, T, Y, router);
  }
  const std::string sn = name.substr(0, name.size() - 7) + ".scale";  // (name ends in ".weight")
  if (const GgufTensor* s = F->find(sn)) {
    if (s->nelements() != 1) { zlog::eprintf("inkling2: %s is not one value\n", sn.c_str()); exit(1); }
    const float c = vecf(sn)[0];
    for (auto& y : Y) y *= c;
  }
}

// (REF_INK_W8=1, a check of the engine: a float weight row the engine keeps as Q8_0 (load_w's float matrices, the
//  streamed experts' float records) rounded the same way, with the engine's own requant_rows_q8_0)
inline bool ink_w8() {
  static const bool on = getenv("REF_INK_W8") && atoi(getenv("REF_INK_W8")) != 0;
  return on;
}
inline void row_w8(uint32_t type, const uint8_t* row, size_t rb, int K, float* w) {
  if (ink_w8() && (type == GGML_BF16 || type == GGML_F16 || type == GGML_F32) && K % 32 == 0) {
    const std::vector<uint8_t> q = requant_rows_q8_0(type, row, rb, 1, K);
    dequant_row(GGML_Q8_0, q.data(), K, w);
  } else {
    dequant_row(type, row, K, w);
  }
}
// rows of a [K][N][n_expert] tensor: expert e's N x K matrix times x
inline void expert_rows(const GgufTensor& t, int e, int N, int K, const float* x, float* y) {
  const size_t rb = t.nbytes / ((size_t)t.dims[2] * N);
  std::vector<float> w(K);
  for (int n = 0; n < N; ++n) {
    row_w8(t.type, t.data + ((size_t)e * N + n) * rb, rb, K, w.data());
    double acc = 0;
    for (int k = 0; k < K; ++k) acc += (double)w[k] * x[k];
    y[n] = (float)acc;
  }
}

// llama's mix: the states (each block, then cur) RMS-normed (eps 1e-6, no weight); per query j a softmax over the
// states of (state . query_j) / sqrt(E); out_j = the softmax-weighted sum of the normed states. q: [nq][E]
inline void mix(const std::vector<const std::vector<float>*>& blocks, const std::vector<float>& cur, const float* q, int nq,
                int T, int E, std::vector<float>& out) {
  out.assign((size_t)T * nq * E, 0.0f);
  const int S = (int)blocks.size() + 1;
#pragma omp parallel for
  for (int t = 0; t < T; ++t) {
    std::vector<float> st((size_t)S * E);
    for (int s = 0; s < S; ++s) {
      const float* src = s + 1 < S ? &(*blocks[s])[(size_t)t * E] : &cur[(size_t)t * E];
      memcpy(&st[(size_t)s * E], src, E * 4);
      rmsnorm(&st[(size_t)s * E], E, nullptr, 1e-6f);
    }
    for (int j = 0; j < nq; ++j) {
      std::vector<double> sc(S);
      double mx = -1e300;
      for (int s = 0; s < S; ++s) {
        double d = 0;
        for (int i = 0; i < E; ++i) d += (double)st[(size_t)s * E + i] * q[(size_t)j * E + i];
        sc[s] = d / std::sqrt((double)E);
        mx = std::max(mx, sc[s]);
      }
      double z = 0;
      for (auto& v : sc) { v = std::exp(v - mx); z += v; }
      float* o = &out[((size_t)t * nq + j) * E];
      for (int i = 0; i < E; ++i) {
        double acc = 0;
        for (int s = 0; s < S; ++s) acc += sc[s] * st[(size_t)s * E + i];
        o[i] = (float)(acc / z);
      }
    }
  }
}

// row j (of nq) of a mix's output, T x E
inline std::vector<float> take(const std::vector<float>& m, int j, int nq, int T, int E) {
  std::vector<float> o((size_t)T * E);
  for (int t = 0; t < T; ++t) memcpy(&o[(size_t)t * E], &m[((size_t)t * nq + j) * E], E * 4);
  return o;
}

// llama's convolve for a sequence from position 0 (the conv state starts at zero): out[t][c] = cur[t][c] +
// sum_i w[c][i] x[t - 3 + i][c] (+ sum_lag x[t - lag][c] dyn[t][h(c)][lag]), divided by sqrt((dyn ? 2 : 1) n_conv + 1)
inline std::vector<float> convolve(const std::vector<float>& cur, int T, int C, const std::vector<float>& w, int NC,
                                   const std::vector<float>* dyn) {
  std::vector<float> out((size_t)T * C);
  const int nh = dyn ? (int)(dyn->size() / T / NC) : 1, hw = C / nh;
  const double sc = 1.0 / std::sqrt((double)((dyn ? 2 : 1) * NC + 1));
  for (int t = 0; t < T; ++t)
    for (int c = 0; c < C; ++c) {
      double acc = 0;
      for (int i = 0; i < NC; ++i) {
        const int tt = t - (NC - 1) + i;
        if (tt >= 0) acc += (double)w[(size_t)c * NC + i] * cur[(size_t)tt * C + c];
      }
      if (dyn)
        for (int lag = 0; lag < NC; ++lag) {
          const int tt = t - lag;
          if (tt >= 0) acc += (double)cur[(size_t)tt * C + c] * (*dyn)[(size_t)t * nh * NC + (size_t)(c / hw) * NC + lag];
        }
      out[(size_t)t * C + c] = (float)((cur[(size_t)t * C + c] + acc) * sc);
    }
  return out;
}

// llama's clip on rows of n: times the per-row scale (nullptr: none), then scaled down to an L2 norm of at most thr
// (factor = clamp(thr / (norm + 1e-6), 0, 1))
inline void clip(float* v, int n, const float* scale, float thr) {
  if (scale)
    for (int i = 0; i < n; ++i) v[i] *= scale[i];
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += (double)v[i] * v[i];
  const float norm = (float)std::sqrt(ss) + 1e-6f;
  const float f = std::clamp(thr / norm, 0.0f, 1.0f);
  for (int i = 0; i < n; ++i) v[i] *= f;
}

struct Model {
  int nl = 0, E = 0, D = 0, NREL = 0, EXT = 0, SWA = 0, NC = 0, BLOCK = 0, NE = 0, NU = 0;
  int start = 0, end = 0, passes = 1;
  float eps = 1e-6f, qk_clip = 0, rel_clip = 0, log_floor = 1, log_alpha = 0, wscale = 1, situ_b = 1, situ_lb = 0;
  std::vector<int64_t> swa;
};

// one layer in cache slot il of physical layer ip (llama's build_layer); msgs: [T][4][E] for a later loop pass, or
// nullptr. cur: the stream in and out; blocks: the earlier streams (pushed to at a block's first layer)
// (REF_INK_DUMP=DIR, debugging: each layer's stages of its first run as DIR/<stage>_<layer>.bin, f32; runz's
//  ARC_INK_DUMP writes the same names from the engine, zero/engine_inkling.h)
inline void ink_dump(const char* stage, int ip, const std::vector<float>& v) {
  static const char* dir = getenv("REF_INK_DUMP");
  if (!dir) return;
  std::string fn;
  for (int k = 0;; ++k) {  // (each run of the layer its own file: _0 the first, a loop's later passes _1, _2)
    fn = std::string(dir) + "/" + stage + "_" + std::to_string(ip) + "_" + std::to_string(k) + ".bin";
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) break;
    fclose(f);
  }
  if (FILE* f = fopen(fn.c_str(), "wb")) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
}
inline void layer(const Model& M, int ip, int T, std::vector<float>& cur, std::vector<std::vector<float>>& blocks,
                  const std::vector<float>* msgs) {
  const int E = M.E, D = M.D;
  const std::string p = "blk." + std::to_string(ip) + ".";
  const bool is_swa = ip < (int)M.swa.size() && M.swa[ip] != 0;
  const int H = per_layer("attention.head_count", ip), HKV = per_layer("attention.head_count_kv", ip);
  const int KG = HKV * D;
  const int FF = per_layer("expert_feed_forward_length", ip);
  const int NU = per_layer("expert_used_count", ip, M.NU);  // (llama reads it per layer)
  std::vector<const std::vector<float>*> bp;
  for (const auto& b : blocks) bp.push_back(&b);
  const std::vector<float> inpSA = cur;

  // the attention inputs: one mix (SWA) or one per q / k / v (global), from the second layer on
  std::vector<float> in[3] = {cur, cur, cur};
  if (F->find(p + "attn_res_query.weight")) {
    const std::vector<float> q = vecf(p + "attn_res_query.weight");
    const int nq = (int)(q.size() / E);
    std::vector<float> m;
    mix(bp, cur, q.data(), nq, T, E, m);
    for (int j = 0; j < 3; ++j) in[j] = take(m, is_swa ? 0 : j, nq, T, E);
  }
  // a later loop pass: (1 - forget) x + gain message, gated on both (llama's control), per input
  static const char* kCtrl[4] = {"q", "k", "v", "ffn"};
  auto control = [&](std::vector<float>& x, int c) {
    const std::vector<float> msg = take(*msgs, c, 4, T, E);
    std::vector<float> g((size_t)T * 2 * E), lo, gain, forget;
    for (int t = 0; t < T; ++t) {
      memcpy(&g[(size_t)t * 2 * E], &x[(size_t)t * E], E * 4);
      memcpy(&g[(size_t)t * 2 * E + E], &msg[(size_t)t * E], E * 4);
      rmsnorm(&g[(size_t)t * 2 * E], E, nullptr, 1e-6f);
      rmsnorm(&g[(size_t)t * 2 * E + E], E, nullptr, 1e-6f);
    }
    const std::string lp = p + "loop_" + kCtrl[c] + "_ctrl_";
    mm(lp + "down.weight", g, T, lo);
    mm(lp + "gain.weight", lo, T, gain);
    mm(lp + "forget.weight", lo, T, forget);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = (float)(x[i] - std::tanh((double)forget[i]) * x[i] + std::tanh((double)gain[i]) * msg[i]);
  };
  if (msgs)
    for (int j = 0; j < 3; ++j) control(in[j], j);
  if (is_swa) {
    const auto an = vecf(p + "attn_norm.weight");
    for (int j = 0; j < (msgs ? 3 : 1); ++j)
      for (int t = 0; t < T; ++t) rmsnorm(&in[j][(size_t)t * E], E, an.data(), M.eps);
    if (!msgs) in[1] = in[2] = in[0];
  } else {
    static const char* kNorm[3] = {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight"};
    for (int j = 0; j < 3; ++j) {
      const auto w = vecf(p + kNorm[j]);
      for (int t = 0; t < T; ++t) rmsnorm(&in[j][(size_t)t * E], E, w.data(), M.eps);
    }
  }

  // projections; K and V through their convolutions (global: with the input-dependent taps)
  std::vector<float> Q, K0, V0, R, S, G, kd, vd;
  mm(p + "attn_q.weight", in[0], T, Q);
  mm(p + "attn_k.weight", in[1], T, K0);
  mm(p + "attn_v.weight", in[2], T, V0);
  mm(p + "attn_rel.weight", in[0], T, R);
  mm(p + "attn_sink_proj.weight", in[0], T, S);
  mm(p + "attn_gate.weight", in[0], T, G);
  if (!is_swa) {
    mm(p + "attn_k_conv_proj.weight", in[1], T, kd);
    mm(p + "attn_v_conv_proj.weight", in[2], T, vd);
  }
  std::vector<float> Kc = convolve(K0, T, KG, vecf(p + "attn_k_conv.weight"), M.NC, is_swa ? nullptr : &kd);
  std::vector<float> Vc = convolve(V0, T, KG, vecf(p + "attn_v_conv.weight"), M.NC, is_swa ? nullptr : &vd);
  const auto qc = vecf(p + "attn_q_clip.weight"), rc = vecf(p + "attn_rel_clip.weight"), rb = vecf(p + "attn_rel_bias.weight");
  const int ext = is_swa ? M.SWA : M.EXT;  // (attn_rel_bias: [ext][NREL])
  for (int t = 0; t < T; ++t) {
    // log-scaled positions: 1 + alpha log(max(1, (pos + 1) / floor)), on q and r of the global layers
    const float tau = is_swa ? 1.0f : 1.0f + M.log_alpha * std::log(std::max(1.0f, (float)(t + 1) / M.log_floor));
    for (int h = 0; h < H; ++h) {
      float* q = &Q[((size_t)t * H + h) * D];
      clip(q, D, &qc[(size_t)h * D], M.qk_clip);
      float* r = &R[((size_t)t * H + h) * M.NREL];
      clip(r, M.NREL, &rc[(size_t)h * M.NREL], M.rel_clip);
      if (!is_swa) {
        for (int i = 0; i < D; ++i) q[i] *= tau;
        for (int i = 0; i < M.NREL; ++i) r[i] *= tau;
      }
    }
    for (int h = 0; h < HKV; ++h) {
      clip(&Kc[((size_t)t * HKV + h) * D], D, nullptr, M.qk_clip);
      rmsnorm(&Vc[((size_t)t * HKV + h) * D], D, nullptr, M.eps);
    }
  }
  ink_dump("q", ip, Q);
  ink_dump("r", ip, R);
  ink_dump("kc", ip, Kc);
  ink_dump("vc", ip, Vc);
  // attention: score = q . k / sqrt(D) + r . bias[distance] - sink (distance past the table: 0 - sink); causal, the
  // SWA layers within the window; a zero sink in the softmax: p_u = exp(s_u) / (1 + sum exp(s))
  std::vector<float> att((size_t)T * H * D);
#pragma omp parallel for
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      const int g = h / (H / HKV);
      const float* q = &Q[((size_t)t * H + h) * D];
      const float* r = &R[((size_t)t * H + h) * M.NREL];
      const double sink = S[(size_t)t * H + h];
      const int u0 = is_swa ? std::max(0, t - M.SWA + 1) : 0;
      std::vector<double> sc(t + 1 - u0);
      double mx = 0;  // (the sink's logit 0 is in the softmax)
      for (int u = u0; u <= t; ++u) {
        double s = 0;
        for (int i = 0; i < D; ++i) s += (double)q[i] * Kc[((size_t)u * HKV + g) * D + i];
        s /= std::sqrt((double)D);
        const int dist = t - u;
        double b = 0;
        if (dist < ext)
          for (int i = 0; i < M.NREL; ++i) b += (double)rb[(size_t)dist * M.NREL + i] * r[i];
        sc[u - u0] = s + b - sink;
        mx = std::max(mx, sc[u - u0]);
      }
      double z = std::exp(-mx);
      for (auto& s : sc) { s = std::exp(s - mx); z += s; }
      for (int i = 0; i < D; ++i) {
        double acc = 0;
        for (int u = u0; u <= t; ++u) acc += sc[u - u0] * Vc[((size_t)u * HKV + g) * D + i];
        att[((size_t)t * H + h) * D + i] = (float)(acc / z * sigm(G[((size_t)t * H + h) * D + i]));
      }
    }
  ink_dump("att", ip, att);
  std::vector<float> ao;
  mm(p + "attn_output.weight", att, T, ao);
  ao = convolve(ao, T, E, vecf(p + "attn_post_conv.weight"), M.NC, nullptr);
  ink_dump("ao", ip, ao);
  // a block's first layer starts a new stream (the old one joins the blocks); else the residual
  if (ip % M.BLOCK == 0) {
    blocks.push_back(inpSA);
    cur = ao;
  } else {
    for (size_t i = 0; i < cur.size(); ++i) cur[i] = inpSA[i] + ao[i];
  }
  const std::vector<float> ffn_inp = cur;

  // MoE input: a mix over the blocks (with the new one) and cur, the controller, ffn_norm
  bp.clear();
  for (const auto& b : blocks) bp.push_back(&b);
  std::vector<float> xm;
  {
    const auto q = vecf(p + "ffn_res_query.weight");
    mix(bp, cur, q.data(), 1, T, E, xm);
  }
  if (msgs) control(xm, 3);
  {
    const auto fn = vecf(p + "ffn_norm.weight");
    for (int t = 0; t < T; ++t) rmsnorm(&xm[(size_t)t * E], E, fn.data(), M.eps);
  }
  ink_dump("xm", ip, xm);
  std::vector<float> rl;
  mm(p + "ffn_gate_inp.weight", xm, T, rl);
  ink_dump("rl", ip, rl);
  const std::vector<float> pb = F->find(p + "exp_probs_b.bias") ? vecf(p + "exp_probs_b.bias") : std::vector<float>(M.NE, 0.0f);
  const GgufTensor& tgu = F->get(p + "ffn_gate_up_exps.weight");
  const GgufTensor& tdn = F->get(p + "ffn_down_exps.weight");
  const std::vector<float> sgu = F->find(p + "ffn_gate_up_exps.scale") ? vecf(p + "ffn_gate_up_exps.scale") : std::vector<float>(M.NE, 1.0f);
  const std::vector<float> sdn = F->find(p + "ffn_down_exps.scale") ? vecf(p + "ffn_down_exps.scale") : std::vector<float>(M.NE, 1.0f);
  std::vector<float> mo((size_t)T * E, 0.0f);
#pragma omp parallel for
  for (int t = 0; t < T; ++t) {
    std::vector<double> prob(M.NE), sel(M.NE);
    for (int e = 0; e < M.NE; ++e) {
      prob[e] = sigm(rl[(size_t)t * M.NE + e]);
      sel[e] = prob[e] + pb[e];
    }
    std::vector<int> id(M.NE);
    for (int e = 0; e < M.NE; ++e) id[e] = e;
    std::stable_sort(id.begin(), id.end(), [&](int a, int b) { return sel[a] > sel[b]; });
    double sum = 0;
    for (int k = 0; k < NU; ++k) sum += prob[id[k]];
    std::vector<float> gu(2 * FF), act(FF), de(E);
    // (REF_INK_Q8=2: the experts' inputs rounded too, per 32 values as moe_quant does: the engine's int8 drift in full)
    static const bool q8x = getenv("REF_INK_Q8") && atoi(getenv("REF_INK_Q8")) >= 2;
    auto q8r = [](const float* x, int K, std::vector<float>& o) {
      std::vector<int8_t> aq((size_t)K);
      std::vector<float> asc((size_t)K / 32);
      std::vector<uint16_t> ab((size_t)K / 16);
      quantize_act_cpu(x, 1, K, aq.data(), asc.data(), ab.data(), true);
      o.resize((size_t)K);
      for (int k = 0; k < K; ++k) o[(size_t)k] = (float)aq[(size_t)k] * asc[(size_t)k / 32];
    };
    std::vector<float> xin, ain;
    if (q8x && E % 32 == 0) q8r(&xm[(size_t)t * E], E, xin);
    for (int k = 0; k < NU; ++k) {
      const int e = id[k];
      double w = prob[e] / (sum + 1e-20);
      if (M.wscale != 0.0f && M.wscale != 1.0f) w *= M.wscale;
      expert_rows(tgu, e, 2 * FF, E, xin.empty() ? &xm[(size_t)t * E] : xin.data(), gu.data());
      for (int i = 0; i < FF; ++i) {
        const double g = (double)gu[i] * sgu[e], u = (double)gu[FF + i] * sgu[e];
        // situ(gate, up) = beta tanh(gate / beta) sigmoid(gate) * lb tanh(up / lb)
        const double a = M.situ_b * std::tanh(g / M.situ_b) * sigm(g);
        act[i] = (float)(a * (M.situ_lb > 0 ? M.situ_lb * std::tanh(u / M.situ_lb) : u));
      }
      if (q8x && FF % 32 == 0) q8r(act.data(), FF, ain);
      expert_rows(tdn, e, E, FF, ain.empty() ? act.data() : ain.data(), de.data());
      for (int i = 0; i < E; ++i) mo[(size_t)t * E + i] += (float)(w * de[i] * sdn[e]);
    }
  }
  ink_dump("mo", ip, mo);
  mo = convolve(mo, T, E, vecf(p + "ffn_post_conv.weight"), M.NC, nullptr);
  for (size_t i = 0; i < cur.size(); ++i) cur[i] = ffn_inp[i] + mo[i];
  ink_dump("out", ip, cur);
}

}  // namespace ref_inkling2

// The whole pass: x in = the token embeddings, out = the output norm's result
static void inkling2_layers(const std::vector<int>& toks, std::vector<float>& x) {
  using namespace ref_inkling2;
  Model M;
  const int T = (int)toks.size();
  M.nl = hp("block_count"); M.E = hp("embedding_length"); M.D = hp("attention.key_length");
  M.NREL = hp("attention.relative_length"); M.EXT = hp("attention.relative_extent"); M.SWA = hp("attention.sliding_window");
  M.NC = hp("shortconv.l_cache"); M.BLOCK = hp("attn_res.block_size"); M.NE = hp("expert_count"); M.NU = hp("expert_used_count");
  M.eps = fkey("attention.layer_norm_rms_epsilon", 1e-6); M.qk_clip = fkey("attention.qk_clip", 0);
  M.rel_clip = fkey("attention.relative_clip", 0); M.log_floor = (float)hp("attention.log_scale_floor", 1);
  M.log_alpha = fkey("attention.log_scale_alpha", 0); M.wscale = fkey("expert_weights_scale", 0);
  M.situ_b = fkey("activation.situ_beta", 1); M.situ_lb = fkey("activation.situ_linear_beta", 0);
  if (const auto* v = F->get_ints(A + ".attention.sliding_window_pattern")) M.swa = *v;
  for (const GgufTensor& t : F->tensors)  // (activation scales: FP8 activations, which this reference does not model)
    if (t.name.size() > 12 && t.name.compare(t.name.size() - 12, 12, ".input_scale") == 0) {
      zlog::eprintf("inkling2: %s: activation scales are not supported\n", t.name.c_str());
      exit(1);
    }
  const bool loop = A == "inkling2_loop";
  M.start = loop ? hp("loop.start") : M.nl;
  M.end = loop ? hp("loop.end") : M.nl;
  M.passes = loop ? hp("loop.count") : 1;
  const int E = M.E;
  {
    const auto tn = vecf("token_embd_norm.weight");
    for (int t = 0; t < T; ++t) rmsnorm(&x[(size_t)t * E], E, tn.data(), M.eps);
  }
  std::vector<std::vector<float>> blocks;
  for (int il = 0; il < M.start; ++il) {
    layer(M, il, T, x, blocks, nullptr);
    zlog::eprintf("\rlayer %d", il + 1);
  }
  const std::vector<float> prelude = x;
  const std::vector<std::vector<float>> prelude_blocks = blocks;
  for (int pass = 0; pass < M.passes && M.start < M.end; ++pass) {
    // a later pass: the bank of the previous one (its blocks after the first core block, and its output)
    std::vector<std::vector<float>> bank;
    std::vector<float> bank_out;
    if (pass > 0) {
      bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
      bank_out = x;
      x = prelude;
      blocks = prelude_blocks;
    }
    std::vector<const std::vector<float>*> bp;
    for (const auto& b : bank) bp.push_back(&b);
    for (int ip = M.start; ip < M.end; ++ip) {
      if (pass == 0) {
        layer(M, ip, T, x, blocks, nullptr);
      } else {
        const auto q = vecf("blk." + std::to_string(ip) + ".loop_msg_query.weight");
        std::vector<float> msgs;
        mix(bp, bank_out, q.data(), 4, T, E, msgs);
        layer(M, ip, T, x, blocks, &msgs);
      }
      zlog::eprintf("\rpass %d layer %d", pass + 1, ip + 1);
    }
  }
  for (int il = M.end; il < M.nl; ++il) {
    layer(M, il, T, x, blocks, nullptr);
    zlog::eprintf("\rlayer %d", il + 1);
  }
  // the output: a mix over the blocks and the stream, then output_norm
  std::vector<const std::vector<float>*> bp;
  for (const auto& b : blocks) bp.push_back(&b);
  const auto q = vecf("output_res_query.weight");
  std::vector<float> m;
  mix(bp, x, q.data(), 1, T, E, m);
  ink_dump("fin", 99, m);
  const auto on = vecf("output_norm.weight");
  for (int t = 0; t < T; ++t) rmsnorm(&m[(size_t)t * E], E, on.data(), M.eps);
  x = m;
}
