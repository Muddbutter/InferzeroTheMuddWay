// Inkling 2 (inkling2 / inkling2_loop): the operations no other model has, as llama.cpp's graph computes them
// (src/models/inkling2.cpp; the f32 reference: tools/ref_inkling2.h). f32 throughout; the K / V cache in f16. Written
// for correctness first (docs/inkling2.md, I1.5): the speed work (I2 / I3) comes after the KL checks.
//
// Work-group sizes: every kernel here with local memory takes an explicit local size, INK_LS work-items (a power of
// two, at most 256; ink_attn: exactly INK_D), and every barrier is reached by all work-items in the same control flow
// (loop counts uniform across the group; AGENTS.md, docs/lessons-intel-arc-igpu.md 2026-11-23). The element-wise
// kernels (ink_ctrl_apply, ink_conv, ink_conv_state, ink_kv_store) take any size.

#ifndef INK_LS
#define INK_LS 256
#endif
#ifndef INK_D
#define INK_D 128  // (the head size: Inkling 2 Mini's 128)
#endif
#ifndef INK_MAXS
#define INK_MAXS 16  // (states a mix may hold: the blocks + the stream)
#endif
#ifndef INK_EXT
#define INK_EXT 1024  // (the longest relative-bias table: Inkling 2 Mini's window and extent, both 1024)
#endif

// sum of v over the work-group (every work-item gets it); red: INK_LS floats of local memory
inline float ink_wg_sum(float v, __local float* red) {
  const int l = get_local_id(0), n = get_local_size(0);
  red[l] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] += red[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}
inline float ink_wg_max(float v, __local float* red) {
  const int l = get_local_id(0), n = get_local_size(0);
  red[l] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = n / 2; s > 0; s >>= 1) {
    if (l < s) red[l] = fmax(red[l], red[l + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}
inline float ink_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// llama's mix (attention over depth), one work-group per token t: the states (nb blocks, block s at blocks + s bstride,
// then cur), each RMS-normed (eps 1e-6, no weight); per query j (q: nq rows of E) a softmax over the states of
// (normed state . q_j) / sqrt(E); out[t][j] = the softmax-weighted sum of the normed states. Rows of E per token.
// (I2 step 1, 2026-11-24: sep 1 writes query j's rows to outj ([T][E] each) instead of out's [T][nq][E]; sep 2 (nq 1)
//  writes the one query's rows to out, out1 and out2: what ink_take copied out after it. The same values)
__kernel void ink_mix(__global const float* blocks, int nb, long bstride, __global const float* cur,
                      __global const float* q, int nq, __global float* out, int E, __global float* out1,
                      __global float* out2, int sep) {
  __local float red[INK_LS];
  __local float rs[INK_MAXS];
  __local float sc[4 * INK_MAXS];
  const int t = get_group_id(0), l = get_local_id(0), n = get_local_size(0), S = nb + 1;
#define INK_STATE(s) ((s) < nb ? blocks + (s) * bstride + (size_t)t * E : cur + (size_t)t * E)
  for (int s = 0; s < S; ++s) {
    __global const float* x = INK_STATE(s);
    float ss = 0.0f;
    for (int i = l; i < E; i += n) ss += x[i] * x[i];
    ss = ink_wg_sum(ss, red);
    if (l == 0) rs[s] = rsqrt(ss / (float)E + 1e-6f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv = rsqrt((float)E);
  for (int j = 0; j < nq; ++j)
    for (int s = 0; s < S; ++s) {
      __global const float* x = INK_STATE(s);
      float d = 0.0f;
      for (int i = l; i < E; i += n) d += x[i] * q[(size_t)j * E + i];
      d = ink_wg_sum(d, red);
      if (l == 0) sc[j * INK_MAXS + s] = d * rs[s] * inv;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l < nq) {  // (the softmax per query, by one work-item each)
    float mx = -INFINITY, z = 0.0f;
    for (int s = 0; s < S; ++s) mx = fmax(mx, sc[l * INK_MAXS + s]);
    for (int s = 0; s < S; ++s) z += sc[l * INK_MAXS + s] = exp(sc[l * INK_MAXS + s] - mx);
    for (int s = 0; s < S; ++s) sc[l * INK_MAXS + s] /= z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int j = 0; j < nq; ++j)
    for (int i = l; i < E; i += n) {
      float acc = 0.0f;
      for (int s = 0; s < S; ++s) acc += sc[j * INK_MAXS + s] * rs[s] * INK_STATE(s)[i];
      if (sep == 0) {
        out[((size_t)t * nq + j) * E + i] = acc;
      } else {
        const size_t o = (size_t)t * E + i;
        if (sep == 2 || j == 0) out[o] = acc;
        if (sep == 2 || j == 1) out1[o] = acc;
        if (sep == 2 || j == 2) out2[o] = acc;
      }
    }
#undef INK_STATE
}

// the controller's input, one work-group per token: g[t] = [rms(x[t]), rms(m[t])] (eps 1e-6, no weight), 2E values;
// x and m rows at xstride / mstride floats, m from element moff (one message of a [T][4][E] mix output: moff c E)
__kernel void ink_rms2(__global const float* x, int xstride, __global const float* m, int moff, int mstride, __global float* g,
                       int E) {
  __local float red[INK_LS];
  const int t = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  __global const float* a = x + (size_t)t * xstride;
  __global const float* b = m + moff + (size_t)t * mstride;
  float sa = 0.0f, sb = 0.0f;
  for (int i = l; i < E; i += n) { sa += a[i] * a[i]; sb += b[i] * b[i]; }
  const float ra = rsqrt(ink_wg_sum(sa, red) / (float)E + 1e-6f), rb = rsqrt(ink_wg_sum(sb, red) / (float)E + 1e-6f);
  __global float* o = g + (size_t)t * 2 * E;
  for (int i = l; i < E; i += n) { o[i] = a[i] * ra; o[E + i] = b[i] * rb; }
}

// (I2 step 1, 2026-11-24) ink_rms2, then ctrl_down's quantization of its 2E rows, in one launch: the row as ink_rms2
// writes it (work-groups of 256), then quantized from g as swiglu_quant does; pad rows (t >= M_valid) quantize as
// zeros. grid (256, Mp), local (256, 1); 2E a multiple of 256
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ink_rms2_quant(__global const float* x, int xstride, __global const float* m, int moff, int mstride,
                             __global float* g, int E, int M, int M_valid, __global char* aq, __global float* asc,
                             __global ushort* absum, int act32) {
  __local float red[INK_LS];
  const int t = get_group_id(1), l = get_local_id(0), n = get_local_size(0), lane = get_sub_group_local_id();
  const bool valid = t < M_valid;
  __global float* o = g + (size_t)t * 2 * E;
  if (valid) {
    __global const float* a = x + (size_t)t * xstride;
    __global const float* b = m + moff + (size_t)t * mstride;
    float sa = 0.0f, sb = 0.0f;
    for (int i = l; i < E; i += n) { sa += a[i] * a[i]; sb += b[i] * b[i]; }
    const float ra = rsqrt(ink_wg_sum(sa, red) / (float)E + 1e-6f), rb = rsqrt(ink_wg_sum(sb, red) / (float)E + 1e-6f);
    for (int i = l; i < E; i += n) { o[i] = a[i] * ra; o[E + i] = b[i] * rb; }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  for (int s = get_sub_group_id(); s < 2 * E / 256; s += get_num_sub_groups()) {
    float v[16];
    if (!valid) {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    } else {
      const float16 q = vload16(0, o + s * 256 + 16 * lane);
      const float qv[16] = {q.s0, q.s1, q.s2, q.s3, q.s4, q.s5, q.s6, q.s7, q.s8, q.s9, q.sa, q.sb, q.sc, q.sd, q.se, q.sf};
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = qv[i];
    }
    quant_store16(v, s, t, M, 2 * E, aq, asc, absum, act32);
  }
}

// the controller's result, element-wise (any size): x = x - tanh(f) x + tanh(gn) m, for T x E (m at mstride floats)
__kernel void ink_ctrl_apply(__global float* x, __global const float* f, __global const float* gn, __global const float* m,
                             int moff, int mstride, int E, int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  const int t = i / E, e = i % E;
  const float v = x[i];
  x[i] = v - tanh(f[i]) * v + tanh(gn[i]) * m[moff + (size_t)t * mstride + e];
}

// (I2 step 1, 2026-11-24) the controller's gain, forget and apply in one launch, for f32 gain / forget matrices (the
// rank-64 ones, which ink_mm_f32 ran): per value (t, e) its forget and gain rows' dot products with lo[t] (K values),
// the same loop as ink_mm_f32's, then ink_ctrl_apply's step. The same values bit for bit
__kernel void ink_ctrl_fused(__global float* x, __global const float* Wf, __global const float* Wg,
                             __global const float* lo, __global const float* m, int moff, int mstride, int E, int K,
                             int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  const int t = i / E, e = i % E;
  __global const float* xr = lo + (size_t)t * K;
  __global const float* wf = Wf + (size_t)e * K;
  __global const float* wg = Wg + (size_t)e * K;
  float f = 0.0f, g = 0.0f;
  for (int k = 0; k < K; ++k) f += wf[k] * xr[k];
  for (int k = 0; k < K; ++k) g += wg[k] * xr[k];
  const float v = x[i];
  x[i] = v - tanh(f) * v + tanh(g) * m[moff + (size_t)t * mstride + e];
}

// llama's convolve on a chunk of T tokens after the slot's history (any size; one work-item per value):
// out[t][c] = (cur[t][c] + sum_i w[c][i] x(t - (NC - 1) + i) + [dyn] sum_lag x(t - lag) dyn[t][h(c)][lag]) * scale,
// x(tt) the chunk's input for tt >= 0, else the slot's state (the NC - 1 inputs before the chunk, oldest first, per
// channel). dyn: T x (nh NC) or none (has_dyn 0). Call ink_conv_state afterwards (this reads the old state).
// (I2 step 6, DSpark's verification: has_save also copies the chunk's inputs to save, T x C, for ink_conv_commit to
//  move the state on by the accepted rows only; the caller then skips ink_conv_state)
__kernel void ink_conv(__global const float* cur, __global const float* w, __global const float* dyn, int has_dyn, int nh,
                       __global const float* state, __global float* out, int C, int T, int NC, float scale,
                       __global float* save, int has_save) {
  const int i = get_global_id(0);
  if (i >= C * T) return;
  if (has_save) save[i] = cur[i];
  const int t = i / C, c = i % C;
#define INK_X(tt) ((tt) >= 0 ? cur[(size_t)(tt) * C + c] : state[(size_t)c * (NC - 1) + (NC - 1) + (tt)])
  float acc = 0.0f;
  for (int k = 0; k < NC; ++k) acc += w[(size_t)c * NC + k] * INK_X(t - (NC - 1) + k);
  if (has_dyn) {
    __global const float* dt = dyn + (size_t)t * nh * NC + (size_t)(c / (C / nh)) * NC;
    for (int lag = 0; lag < NC; ++lag) acc += INK_X(t - lag) * dt[lag];
  }
  out[i] = (cur[i] + acc) * scale;
#undef INK_X
}

// (I2 step 1, 2026-11-24) one token (generation): ink_conv and ink_conv_state in one launch, one work-item per channel:
// the conv as ink_conv's (reading the old state), then the state moved on as ink_conv_state's. The same values
// (channel c's conv of its one input xc; returns the output and moves the state on)
inline float ink_conv1_c(float xc, __global const float* w, __global const float* dyn, int has_dyn, int nh,
                         __global float* state, int c, int C, int NC, float scale) {
  __global float* s = state + (size_t)c * (NC - 1);
#define INK_X1(tt) ((tt) >= 0 ? xc : s[(NC - 1) + (tt)])
  float acc = 0.0f;
  for (int k = 0; k < NC; ++k) acc += w[(size_t)c * NC + k] * INK_X1(-(NC - 1) + k);
  if (has_dyn) {
    __global const float* dt = dyn + (size_t)(c / (C / nh)) * NC;
    for (int lag = 0; lag < NC; ++lag) acc += INK_X1(-lag) * dt[lag];
  }
  const float o = (xc + acc) * scale;
#undef INK_X1
  for (int k = 0; k < NC - 1; ++k) s[k] = k == NC - 2 ? xc : s[k + 1];
  return o;
}
// (has_add: also add[c] = add[c] + out[c], ink_add's step: the stream's residual in the same launch)
__kernel void ink_conv1(__global const float* cur, __global const float* w, __global const float* dyn, int has_dyn, int nh,
                        __global float* state, __global float* out, int C, int NC, float scale, __global float* add,
                        int has_add) {
  const int c = get_global_id(0);
  if (c >= C) return;
  const float o = ink_conv1_c(cur[c], w, dyn, has_dyn, nh, state, c, C, NC, scale);
  out[c] = o;
  // (the add reads the stored, rounded output back (volatile): with -cl-mad-enable the compiler would otherwise fuse
  //  the conv's last multiply and this add into one fma, another rounding than ink_conv1 + ink_add)
  if (has_add) add[c] = add[c] + ((volatile __global float*)out)[c];
}
// (I2 step 1) one token's MoE output, its FFN conv and the residual in one launch, one work-item per channel: the
// experts' sum as moe_combine forms it for Inkling (x zeroed, x += v + sgate sh with the zero gate and sh = x = 0),
// written to mo; then ink_conv1's conv into out; then cur[c] = cur[c] + out[c] (ink_add). The same values
__kernel void ink_moe_out(__global const float* y, __global const int* inv, __global const float* sgate,
                          __global float* mo, __global const float* w, __global float* state, __global float* out,
                          __global float* cur, int C, int NC, float scale, int NU) {
  const int c = get_global_id(0);
  if (c >= C) return;
  float v = 0.0f;
  for (int k = 0; k < NU; ++k) v += y[(size_t)inv[k] * C + c];
  float x = 0.0f;
  x += v + sgate[0] * 0.0f;
  mo[c] = x;
  const float o = ink_conv1_c(x, w, w, 0, 1, state, c, C, NC, scale);
  out[c] = o;
  cur[c] = cur[c] + ((volatile __global float*)out)[c];  // (as ink_conv1's add: no fma across the two)
}

// the slot's conv state after a chunk: the last NC - 1 inputs, oldest first (one work-item per channel; in place,
// ascending, reads only entries not yet written)
__kernel void ink_conv_state(__global const float* cur, __global float* state, int C, int T, int NC) {
  const int c = get_global_id(0);
  if (c >= C) return;
  __global float* s = state + (size_t)c * (NC - 1);
  for (int k = 0; k < NC - 1; ++k) {
    const int tt = T - (NC - 1) + k;  // (the input this entry now holds)
    s[k] = tt >= 0 ? cur[(size_t)tt * C + c] : s[k + T];
  }
}

// llama's clip on rows of n values (one work-group per row; rows: T x nrow, n <= INK_LS): times the row's scale
// (scale: nrow rows of n, or has_scale 0), scaled down to an L2 norm of at most thr (factor clamp(thr / (norm + 1e-6),
// 0, 1)); then, when use_tau (q and r of the global layers, whatever log_alpha's sign), times
// tau = 1 + log_alpha log(max(1, (pos0 + t + 1) / log_floor)). mode 1: an RMS norm instead (V: eps, no weight)
// (one row r of token t at v; returns this work-item's new value, written back)
inline float ink_row1(__global float* v, int r, int t, int n, __global const float* scale, int has_scale, float thr,
                      int use_tau, float log_alpha, float log_floor, int pos0, int mode, float eps, __local float* red) {
  const int l = get_local_id(0);
  float a = 0.0f;
  if (l < n) {
    a = v[l];
    if (has_scale) a *= scale[(size_t)r * n + l];
  }
  const float ss = ink_wg_sum(a * a, red);
  float f;
  if (mode == 1) f = rsqrt(ss / (float)n + eps);
  else f = clamp(thr / (sqrt(ss) + 1e-6f), 0.0f, 1.0f);
  if (use_tau) f *= 1.0f + log_alpha * log(fmax(1.0f, (float)(pos0 + t + 1) / log_floor));
  const float o = a * f;
  if (l < n) v[l] = o;
  return o;
}
__kernel void ink_rows(__global float* x, int nrow, int n, __global const float* scale, int has_scale, float thr,
                       int use_tau, float log_alpha, float log_floor, int pos0, int mode, float eps) {
  __local float red[INK_LS];
  const int g = get_group_id(0), t = g / nrow, r = g % nrow;
  ink_row1(x + (size_t)g * n, r, t, n, scale, has_scale, thr, use_tau, log_alpha, log_floor, pos0, mode, eps, red);
}
// (I2 step 1, 2026-11-24) ink_rows on q, r, k and v, then ink_kv_store, in one launch: work-groups 0 .. T H - 1 the q
// rows (D values, q_clip, qk_clip, tau), then T H rows of r (NREL, rel_clip, rel_clip's threshold, tau), T HKV of k
// (D, qk_clip), T HKV of v (D, the RMS norm); a k / v row also stored into the slot's f16 cache at its position, as
// ink_kv_store does. Each row ink_rows' own steps: the same values
__kernel void ink_rows_kv(__global float* q, __global float* rr, __global float* kc, __global float* vc, int H, int HKV,
                          int D, int NREL, __global const float* qsc, __global const float* rsc, float qthr, float rthr,
                          int tau, float log_alpha, float log_floor, int pos0, float eps, __global half* kcache,
                          __global half* vcache, int cap, int T) {
  __local float red[INK_LS];
  int g = get_group_id(0);
  const int l = get_local_id(0);
  if (g < T * H) {
    ink_row1(q + (size_t)g * D, g % H, g / H, D, qsc, 1, qthr, tau, log_alpha, log_floor, pos0, 0, eps, red);
    return;
  }
  g -= T * H;
  if (g < T * H) {
    ink_row1(rr + (size_t)g * NREL, g % H, g / H, NREL, rsc, 1, rthr, tau, log_alpha, log_floor, pos0, 0, eps, red);
    return;
  }
  g -= T * H;
  const int isv = g >= T * HKV;
  if (isv) g -= T * HKV;
  const int t = g / HKV, r = g % HKV;
  const float o = ink_row1((isv ? vc : kc) + (size_t)g * D, r, t, D, qsc, 0, qthr, 0, log_alpha, log_floor, pos0,
                           isv ? 1 : 0, eps, red);
  if (l < D) vstore_half(o, (size_t)((pos0 + t) % cap) * HKV * D + (size_t)r * D + l, isv ? vcache : kcache);
}

// K and V of a chunk into the slot's f16 cache (any size): row of position pos0 + t at (pos0 + t) % cap, HKV x D each
__kernel void ink_kv_store(__global const float* k, __global const float* v, __global half* kc, __global half* vc, int KG,
                           int T, int pos0, int cap) {
  const int i = get_global_id(0);
  if (i >= KG * T) return;
  const int t = i / KG, e = i % KG;
  const size_t dst = (size_t)((pos0 + t) % cap) * KG + e;
  vstore_half(k[i], dst, kc);
  vstore_half(v[i], dst, vc);
}

// Attention of a chunk's queries over the slot's cache, one work-group of INK_D work-items per (token, head):
// score(u) = q . k_u / sqrt(D) + r . bias[pos - u] - sink, the bias 0 at distances past the table (ext rows of NREL),
// over keys u = lo .. pos (lo = pos - window + 1 for a sliding layer, else 0), at (u % cap) in the cache. The softmax
// has a zero sink logit as well (llama: the sink subtracted from the bias, zero sinks in the attention), so it starts
// from max 0, sum 1. Online over tiles of INK_D keys. out[t][h] = the weighted V sum times sigmoid(gate[t][h]).
// q, gate: T x H x D; r: T x H x NREL; sink: T x H; kc, vc: cap x HKV x D (f16).
__kernel void ink_attn(__global const float* q, __global const float* r, __global const float* sink,
                       __global const float* gate, __global const half* kc, __global const half* vc,
                       __global const float* bias, int ext, int NREL, __global float* out, int H, int HKV, int pos0,
                       int window, int cap) {
  __local float qs[INK_D];
  __local float table[INK_EXT + 1];
  __local float p[INK_D];
  __local float red[INK_D];
  const int g = get_group_id(0), t = g / H, h = g % H, kvh = h / (H / HKV), d = get_local_id(0);
  const int pos = pos0 + t, lo = window > 0 ? max(0, pos - window + 1) : 0;
  const float sk = sink[(size_t)t * H + h];
  qs[d] = q[((size_t)t * H + h) * INK_D + d];
  __global const float* rr = r + ((size_t)t * H + h) * NREL;
  for (int e = d; e <= ext; e += INK_D) {  // (the bias by distance, minus the sink; past the table: 0 - sink)
    float b = 0.0f;
    if (e < ext)
      for (int j = 0; j < NREL; ++j) b += bias[(size_t)e * NREL + j] * rr[j];
    table[e] = b - sk;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv = rsqrt((float)INK_D);
  float m = 0.0f, z = 1.0f, acc = 0.0f;  // (the zero sink logit)
  for (int base = lo; base <= pos; base += INK_D) {
    const int u = base + d;
    float s = -INFINITY;
    if (u <= pos) {
      __global const half* kr = kc + ((size_t)(u % cap) * HKV + kvh) * INK_D;
      float dot = 0.0f;
      for (int i = 0; i < INK_D; ++i) dot += qs[i] * vload_half(i, kr);
      const int dist = pos - u;
      s = dot * inv + table[min(dist, ext)];
    }
    const float mt = ink_wg_max(s, red);
    const float mn = fmax(m, mt), corr = exp(m - mn);
    const float e = u <= pos ? exp(s - mn) : 0.0f;
    p[d] = e;
    z = z * corr + ink_wg_sum(e, red);  // (the barriers in it also publish p)
    acc *= corr;
    const int nk = min(INK_D, pos - base + 1);
    for (int k = 0; k < nk; ++k) acc += p[k] * vload_half(((size_t)((base + k) % cap) * HKV + kvh) * INK_D + d, vc);
    m = mn;
    barrier(CLK_LOCAL_MEM_FENCE);  // (p is rewritten by the next tile)
  }
  const size_t o = ((size_t)t * H + h) * INK_D + d;
  out[o] = acc / z * ink_sigmoid(gate[o]);
}

// situ on a fused gate-up row pair (any size): out[i] = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), g and u the
// expert's outputs times its scale (llama: the scale on the matmul's output, before the activation). gu: rows of 2F
// (gate F, then up F) per (token, slot); sc: the experts' gate-up scales (indexed by ids[row]), or has_scale 0
__kernel void ink_situ(__global const float* gu, __global const int* ids, __global const float* sc, int has_scale,
                       __global float* out, int F, int rows, float beta, float lb) {
  const int i = get_global_id(0);
  if (i >= F * rows) return;
  const int row = i / F, f = i % F;
  const float s = has_scale ? sc[ids[row]] : 1.0f;
  const float g = gu[(size_t)row * 2 * F + f] * s, u = gu[(size_t)row * 2 * F + F + f] * s;
  const float a = beta * tanh(g / beta) * ink_sigmoid(g);
  out[i] = a * (lb > 0.0f ? lb * tanh(u / lb) : u);
}

// The experts on their GGUF rows, read exactly (the correctness path, I1.5; the fast one goes through the engine's MoE
// kernels later): out[row][n] = sum_k W_e[n][k] x[row][k], e = ids[row], one work-item per (row, n) (any size). W: the
// tensor's raw bytes, expert e's row n at (e N + n) rb; type 40 NVFP4 (per 16 exactly, as ggml's dequantize_row_nvfp4,
// no per-32 fold), 30 BF16, 8 Q8_0, 0 F32. kv: kvalues_mxfp4 (16 bytes)
inline float ink_ue4m3(uint x) {
  if (x == 0u || x == 0x7Fu) return 0.0f;
  const uint e = (x >> 3) & 15u, m = x & 7u;
  return e == 0u ? ldexp((float)m, -10) : ldexp((float)(8u + m), (int)e - 11);
}
__kernel void ink_expert_mv(__global const uchar* W, int type, ulong rb, int N, int K, __global const int* ids,
                            __global const float* x, int xdiv, __global float* out, int rows,
                            __global const char* kv) {
  const size_t i = get_global_id(0);
  if (i >= (size_t)rows * N) return;
  const int row = (int)(i / N), n = (int)(i % N);
  __global const uchar* w = W + ((size_t)ids[row] * N + n) * rb;
  __global const float* xr = x + (size_t)(row / xdiv) * K;  // (xdiv: rows sharing an input, a token's NU experts)
  float acc = 0.0f;
  if (type == 40) {
    for (int b = 0; b < K / 64; ++b) {
      __global const uchar* p = w + (size_t)b * 36;
      for (int s = 0; s < 4; ++s) {
        const float d = ink_ue4m3(p[s]);
        float part = 0.0f;
        for (int j = 0; j < 8; ++j) {
          const uint c = p[4 + 8 * s + j];
          part += (float)kv[c & 15u] * xr[64 * b + 16 * s + j] + (float)kv[c >> 4] * xr[64 * b + 16 * s + 8 + j];
        }
        acc += d * part;
      }
    }
  } else if (type == 30) {
    __global const ushort* h = (__global const ushort*)w;
    for (int k = 0; k < K; ++k) acc += as_float((uint)h[k] << 16) * xr[k];
  } else if (type == 8) {
    for (int b = 0; b < K / 32; ++b) {
      __global const uchar* p = w + (size_t)b * 34;
      const float d = vload_half(0, (__global const half*)p);
      float part = 0.0f;
      for (int j = 0; j < 32; ++j) part += (float)(char)p[2 + j] * xr[32 * b + j];
      acc += d * part;
    }
  } else {
    __global const float* f = (__global const float*)w;
    for (int k = 0; k < K; ++k) acc += f[k] * xr[k];
  }
  out[i] = acc;
}

// the weighted sum of a token's experts (any size): y[t][i] = sum_k w[t][k] scale[ids[t][k]] de[t][k][i] (llama: the
// down matmul's output times the expert's scale, then times the router weight); scale: per expert, or has_scale 0
__kernel void ink_expert_sum(__global const float* de, __global const int* ids, __global const float* w,
                             __global const float* scale, int has_scale, __global float* y, int E, int NU, int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  const int t = i / E, e = i % E;
  float acc = 0.0f;
  for (int k = 0; k < NU; ++k) {
    const int id = ids[(size_t)t * NU + k];
    acc += w[(size_t)t * NU + k] * (has_scale ? scale[id] : 1.0f) * de[((size_t)t * NU + k) * E + e];
  }
  y[i] = acc;
}

// the router for one token per work-group (explicit local size INK_LS >= NE): sigmoid of the logits, the selection
// adds the bias, the top NU by selection (ties: the lower index, as a stable descending sort), weights
// prob / (sum + 1e-20) x wscale (wscale 0 or 1: none). ids / w: T x NU
__kernel void ink_route(__global const float* logits, __global const float* bias, int has_bias, int NE, int NU,
                        float wscale, __global int* ids, __global float* w) {
  __local float prob[INK_LS];
  __local float sel[INK_LS];
  const int t = get_group_id(0), l = get_local_id(0);
  if (l < NE) {
    prob[l] = ink_sigmoid(logits[(size_t)t * NE + l]);
    sel[l] = prob[l] + (has_bias ? bias[l] : 0.0f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0) {
    float sum = 0.0f;
    int pick[64];
    for (int k = 0; k < NU && k < 64; ++k) {
      int best = -1;
      for (int e = 0; e < NE; ++e) {
        bool used = false;
        for (int j = 0; j < k; ++j) used = used || pick[j] == e;
        if (!used && (best < 0 || sel[e] > sel[best])) best = e;
      }
      pick[k] = best;
      sum += prob[best];
    }
    for (int k = 0; k < NU && k < 64; ++k) {
      float wk = prob[pick[k]] / (sum + 1e-20f);
      if (wscale != 0.0f && wscale != 1.0f) wk *= wscale;
      ids[(size_t)t * NU + k] = pick[k];
      w[(size_t)t * NU + k] = wk;
    }
  }
}

// ---- small helpers

// RMS norm with a weight, one work-group per row (explicit local size INK_LS): out[t] = x[t] / rms(x[t]) * w (x rows at
// xstride floats; out rows of E; in place when out == x)
__kernel void ink_rmsnorm(__global const float* x, int xstride, __global const float* w, __global float* out, int E,
                          float eps) {
  __local float red[INK_LS];
  const int t = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  __global const float* a = x + (size_t)t * xstride;
  float ss = 0.0f;
  for (int i = l; i < E; i += n) ss += a[i] * a[i];
  const float r = rsqrt(ink_wg_sum(ss, red) / (float)E + eps);
  for (int i = l; i < E; i += n) out[(size_t)t * E + i] = a[i] * r * w[i];
}

// (I2 step 1, 2026-11-24) ink_rmsnorm, then the next GEMM's quantization of its rows, in one launch: row m's norm as
// ink_rmsnorm computes it (work-groups of 256, ink_wg_sum), written to out (in place allowed), then its superblocks
// quantized from out as swiglu_quant does (quant_store16, per-32 or per-256 scales by act32); rows >= M_valid quantize
// as zeros and write no f32 row, as quant_rows' pad rows. grid (256, Mp), local (256, 1)
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void ink_rmsnorm_quant(__global const float* x, int xstride, __global const float* w, __global float* out, int E,
                                float eps, int M, int M_valid, __global char* aq, __global float* asc,
                                __global ushort* absum, int act32) {
  __local float red[INK_LS];
  const int t = get_group_id(1), l = get_local_id(0), n = get_local_size(0), lane = get_sub_group_local_id();
  const bool valid = t < M_valid;
  if (valid) {  // (the work-group's own row: the same branch for all of it)
    __global const float* a = x + (size_t)t * xstride;
    float ss = 0.0f;
    for (int i = l; i < E; i += n) ss += a[i] * a[i];
    const float r = rsqrt(ink_wg_sum(ss, red) / (float)E + eps);
    for (int i = l; i < E; i += n) out[(size_t)t * E + i] = a[i] * r * w[i];
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  __global const float* gr = out + (size_t)t * E;
  for (int s = get_sub_group_id(); s < E / 256; s += get_num_sub_groups()) {
    float v[16];
    if (!valid) {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    } else {
      const float16 g = vload16(0, gr + s * 256 + 16 * lane);
      const float gv[16] = {g.s0, g.s1, g.s2, g.s3, g.s4, g.s5, g.s6, g.s7, g.s8, g.s9, g.sa, g.sb, g.sc, g.sd, g.se, g.sf};
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = gv[i];
    }
    quant_store16(v, s, t, M, E, aq, asc, absum, act32);
  }
}

// out = a + b, element-wise (any size; in place allowed)
__kernel void ink_add(__global const float* a, __global const float* b, __global float* out, int n) {
  const int i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}

// out[t] = src[t][j] of a [T][nq][E] mix output (any size)
__kernel void ink_take(__global const float* src, int nq, int j, __global float* out, int E, int T) {
  const int i = get_global_id(0);
  if (i >= E * T) return;
  out[i] = src[((size_t)(i / E) * nq + j) * E + i % E];
}

// y[t][n] = sum_k W[n][k] x[t][k] with an f32 W (any size; one work-item per (t, n)): the matrices whose shapes the
// int8 GEMM does not take (N not a multiple of 16 or K of 256: the sinks' 24 rows, the controllers' rank-64 inputs)
// and the router, kept f32 as llama.cpp keeps it (expert_router_f32)
// (I2 step 2, 2026-11-24) the same product with one 16-lane sub-group per output: the lanes take every 16th value of
// the row and the sub-group sums their parts (sub_group_reduce_add). Another order of the sums than ink_mm_f32's: KL,
// not bit for bit. For the few-row matrices (the 24-row sinks, the 128-row router), where one work-item per output left
// the GPU nearly idle: 24 / 128 work-items each running 2,304 multiply-adds in a row. grid: N T 16 (rounded up to the
// work-group), local 64
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ink_mm_f32_sg(__global const float* W, __global const float* x, __global float* y, int N, int K, int T) {
  const int o = get_global_id(0) / 16, lane = get_sub_group_local_id();
  if (o >= N * T) return;  // (a whole sub-group: o is the same for its 16 lanes)
  const int t = o / N, n = o % N;
  __global const float* w = W + (size_t)n * K;
  __global const float* xr = x + (size_t)t * K;
  float acc = 0.0f;
  for (int k = lane; k < K; k += 16) acc += w[k] * xr[k];
  acc = sub_group_reduce_add(acc);
  if (lane == 0) y[o] = acc;
}
__kernel void ink_mm_f32(__global const float* W, __global const float* x, __global float* y, int N, int K, int T) {
  const int i = get_global_id(0);
  if (i >= N * T) return;
  const int t = i / N, n = i % N;
  __global const float* w = W + (size_t)n * K;
  __global const float* xr = x + (size_t)t * K;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += w[k] * xr[k];
  y[i] = acc;
}

// the logits past the vocabulary (the head's padding rows, 201,024 for 200,058 tokens) to -inf, so that nothing samples
// them (llama.cpp drops them); rows of nl logits (any size)
__kernel void ink_mask_logits(__global float* logits, int nl, int nv, int rows) {
  const int i = get_global_id(0), pad = nl - nv;
  if (i >= pad * rows) return;
  logits[(size_t)(i / pad) * nl + nv + i % pad] = -INFINITY;
}

// ---- DSpark drafting (I2 step 6, docs/inkling2.md "DSpark drafting: design"; zero/engine_dspark.h)

// the conv states after a verification pass: each (state, its saved inputs) entry moved on by the first `rows` of the
// pass's inputs, as ink_conv_state would have moved it by those rows alone (in place, ascending, reads only entries
// not yet written). One launch for every visit's four states: grid (max C rounded, entries); tab: (state offset, save
// offset, C) per entry, in floats of the two arenas
__kernel void ink_conv_commit(__global float* states, __global const float* saves, __global const int* tab, int n,
                              int rows, int NC) {
  const int c = get_global_id(0), e = get_global_id(1);
  if (e >= n) return;
  const int C = tab[3 * e + 2];
  if (c >= C) return;
  __global float* s = states + tab[3 * e] + (size_t)c * (NC - 1);
  __global const float* cur = saves + tab[3 * e + 1];
  for (int k = 0; k < NC - 1; ++k) {
    const int tt = rows - (NC - 1) + k;
    s[k] = tt >= 0 ? cur[(size_t)tt * C + c] : s[k + rows];
  }
}

// rows 1 .. B-1 of the draft block: the mask embedding (row 0, the anchor, is written by the host)
__kernel void dsp_block_init(__global float* x, __global const float* mask, int E, int B) {
  const int i = get_global_id(0);
  if (i >= E * (B - 1)) return;
  x[(size_t)E + i] = mask[i % E];
}

// a head's RMS norm (weights w, eps) and its rotary position (NeoX halves, theta = pos base^(-2i / D)) in the
// work-group (INK_D items, one per value); returns this item's value
inline float dsp_norm_rope1(float v, __global const float* w, int pos, float eps, float base, __local float* red,
                            __local float* y) {
  const int d = get_local_id(0), h2 = INK_D / 2;
  const float ss = ink_wg_sum(v * v, red);
  y[d] = v * rsqrt(ss / (float)INK_D + eps) * w[d];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int i = d < h2 ? d : d - h2;
  const float th = (float)pos * pow(base, -2.0f * (float)i / (float)INK_D), cs = cos(th), sn = sin(th);
  const float o = d < h2 ? y[d] * cs - y[d + h2] * sn : y[d] * cs + y[d - h2] * sn;
  barrier(CLK_LOCAL_MEM_FENCE);
  return o;
}
// q or k rows of the block in place: [rows][nh][D], row r at position pos0 + r; one work-group of INK_D per (row, head)
__kernel void dsp_qk(__global float* x, int nh, __global const float* w, int pos0, float eps, float base) {
  __local float red[INK_D], y[INK_D];
  const int g = get_group_id(0), r = g / nh, d = get_local_id(0);
  __global float* p = x + (size_t)g * INK_D;
  p[d] = dsp_norm_rope1(p[d], w, pos0 + r, eps, base, red, y);
}
// the context's K (normed, rotated) and V of `rows` committed tokens into a layer's f16 cache, at (pos0 + r) % cap
__kernel void dsp_kv_store(__global const float* kk, __global const float* vv, int nkv, __global const float* w,
                           int pos0, float eps, float base, __global half* kc, __global half* vc, int cap) {
  __local float red[INK_D], y[INK_D];
  const int g = get_group_id(0), r = g / nkv, h = g % nkv, d = get_local_id(0);
  const float k = dsp_norm_rope1(kk[(size_t)g * INK_D + d], w, pos0 + r, eps, base, red, y);
  const size_t o = ((size_t)((pos0 + r) % cap) * nkv + h) * INK_D + d;
  vstore_half(k, o, kc);
  vstore_half(vv[(size_t)g * INK_D + d], o, vc);
}
// the block's attention, one work-group of INK_D per (row, head): its keys the context's cn - clo cached ones (u % cap)
// and the block's B rows, every row seeing all of them (not causal), scale 1 / sqrt(D); online softmax over tiles of
// INK_D keys, as ink_attn
__kernel void dsp_attn(__global const float* q, __global const float* kb, __global const float* vb,
                       __global const half* kc, __global const half* vc, int cap, int clo, int cn,
                       __global float* out, int H, int HKV, int B) {
  __local float qs[INK_D], p[INK_D], red[INK_D];
  const int g = get_group_id(0), t = g / H, h = g % H, kvh = h / (H / HKV), d = get_local_id(0);
  qs[d] = q[((size_t)t * H + h) * INK_D + d];
  barrier(CLK_LOCAL_MEM_FENCE);
  const int nctx = cn - clo, nk = nctx + B;
  const float inv = rsqrt((float)INK_D);
  float m = -INFINITY, z = 0.0f, acc = 0.0f;
  for (int base = 0; base < nk; base += INK_D) {
    const int j = base + d;
    float s = -INFINITY;
    if (j < nk) {
      float dot = 0.0f;
      if (j < nctx) {
        __global const half* kr = kc + ((size_t)((clo + j) % cap) * HKV + kvh) * INK_D;
        for (int i = 0; i < INK_D; ++i) dot += qs[i] * vload_half(i, kr);
      } else {
        __global const float* kr = kb + ((size_t)(j - nctx) * HKV + kvh) * INK_D;
        for (int i = 0; i < INK_D; ++i) dot += qs[i] * kr[i];
      }
      s = dot * inv;
    }
    const float mt = ink_wg_max(s, red);
    const float mn = fmax(m, mt), corr = m == -INFINITY ? 0.0f : exp(m - mn);
    const float e = j < nk ? exp(s - mn) : 0.0f;
    p[d] = e;
    z = z * corr + ink_wg_sum(e, red);
    acc *= corr;
    const int n = min(INK_D, nk - base);
    for (int k = 0; k < n; ++k) {
      const int jj = base + k;
      const float v = jj < nctx ? vload_half(((size_t)((clo + jj) % cap) * HKV + kvh) * INK_D + d, vc)
                                : vb[((size_t)(jj - nctx) * HKV + kvh) * INK_D + d];
      acc += p[k] * v;
    }
    m = mn;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[((size_t)t * H + h) * INK_D + d] = acc / z;
}
// the taps' mix of a row: out[r] = RMS(sum_t f[t] tap_t[r0 + r], w), f the fuse row (softmaxed at load); taps at
// tstride floats apart, rows of E. One work-group of 256 per row
__kernel void dsp_mix(__global const float* taps, int ntap, long tstride, int r0, __global const float* f,
                      __global const float* w, __global float* out, int E, float eps) {
  __local float red[INK_LS];
  const int r = get_group_id(0), l = get_local_id(0), n = get_local_size(0);
  float ss = 0.0f;
  for (int i = l; i < E; i += n) {
    float v = 0.0f;
    for (int t = 0; t < ntap; ++t) v += f[t] * taps[(size_t)t * tstride + (size_t)(r0 + r) * E + i];
    out[(size_t)r * E + i] = v;
    ss += v * v;
  }
  const float rs = rsqrt(ink_wg_sum(ss, red) / (float)E + eps);
  for (int i = l; i < E; i += n) out[(size_t)r * E + i] = out[(size_t)r * E + i] * rs * w[i];
}
// the Markov head's input: z = [state (R), markov_w1[last] (R), h (E), f (E)]
__kernel void dsp_cat(__global float* z, __global const float* state, __global const float* w1, __global const float* h,
                      __global const float* f, int R, int E) {
  const int i = get_global_id(0);
  if (i >= 2 * (R + E)) return;
  z[i] = i < R ? state[i] : i < 2 * R ? w1[i - R] : i < 2 * R + E ? h[i - 2 * R] : f[i - 2 * R - E];
}
// its gates (g: joint_proj z, b its bias): keep = sigmoid, update = tanh, latent = tanh; state = keep state +
// (1 - keep) update
__kernel void dsp_gate(__global const float* g, __global const float* b, __global float* state, __global float* lat,
                       int R) {
  const int i = get_global_id(0);
  if (i >= R) return;
  const float keep = 1.0f / (1.0f + exp(-(g[i] + b[i])));
  const float upd = tanh(g[R + i] + b[R + i]);
  lat[i] = tanh(g[2 * R + i] + b[2 * R + i]);
  state[i] = keep * state[i] + (1.0f - keep) * upd;
}
// ---- the chain on the GPU (2026-11-24): the head folded into hidden_out at load (F = lm_head hidden_out, V x R f16;
//  fb = lm_head b_h), each row's top-K candidates of its base logits, then per row the candidates' corrections
//  F[c] . latent + fb[c] and their argmax, which also fetches markov_w1[argmax] for the next row: no host wait a row

// F[c][r0 + j] = rows[j][c] (the head's outputs for columns r0 .. r0 + m - 1 of hidden_out), f16
__kernel void dsp_fold(__global const float* rows, int nl, int r0, int m, __global half* F, int R, int V) {
  const int i = get_global_id(0);
  if (i >= V * m) return;
  const int c = i / m, j = i % m;
  vstore_half(rows[(size_t)j * nl + c], (size_t)c * R + r0 + j, F);
}
// markov_w1[tok] (f16 rows of R) into w1row (f32)
__kernel void dsp_w1_gather(int tok, __global const half* w1, __global float* w1row, int R) {
  const int r = get_global_id(0);
  if (r < R) w1row[r] = vload_half((size_t)tok * R + r, w1);
}
// each row's K largest logits among the first nv (one work-group of 256 a row): every lane its DSP_TL largest of a
// strided share, then a bitonic sort of the 256 DSP_TL in local memory; values un-capped (cap tanh(x / cap) is the
// head's softcap: x = cap atanh(v / cap)), indices -1 past the candidates found. A candidate list, not exact: a lane
// holding more than DSP_TL of a row's top K would lose some (drafts only)
#define DSP_TL 8
#define DSP_TN (256 * DSP_TL)
__kernel void dsp_topk(__global const float* lg, int nl, int nv, float cap, int K, __global int* idx,
                       __global float* val) {
  __local float sv[DSP_TN];
  __local int si[DSP_TN];
  const int row = get_group_id(0), l = get_local_id(0);
  __global const float* x = lg + (size_t)row * nl;
  float tv[DSP_TL];
  int ti[DSP_TL];
  for (int k = 0; k < DSP_TL; ++k) tv[k] = -INFINITY, ti[k] = -1;
  for (int c = l; c < nv; c += 256) {
    const float v = x[c];
    if (!(v > tv[DSP_TL - 1])) continue;
    int k = DSP_TL - 1;
    while (k > 0 && tv[k - 1] < v) { tv[k] = tv[k - 1]; ti[k] = ti[k - 1]; --k; }
    tv[k] = v;
    ti[k] = c;
  }
  for (int k = 0; k < DSP_TL; ++k) sv[l * DSP_TL + k] = tv[k], si[l * DSP_TL + k] = ti[k];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int size = 2; size <= DSP_TN; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int t = l; t < DSP_TN / 2; t += 256) {
        const int i = 2 * stride * (t / stride) + t % stride, j = i + stride;
        const bool up = (i & size) == 0;  // (the last pass: every pair up, so the whole list descending)
        const float a = sv[i], b = sv[j];
        if (up ? a < b : a > b) {
          sv[i] = b, sv[j] = a;
          const int q = si[i];
          si[i] = si[j], si[j] = q;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  for (int k = l; k < K; k += 256) {
    const float v = sv[k];
    val[(size_t)row * K + k] = cap > 0.0f && v > -INFINITY ? cap * atanh(clamp(v / cap, -0.999999f, 0.999999f)) : v;
    idx[(size_t)row * K + k] = si[k];
  }
}
// row `row`'s draft: its K candidates' corrected logits (base + F[c] . lat + fb[c]), the argmax (the lowest k on a tie,
// the larger base logit), written to ids[row], and markov_w1[that token] into w1row. One work-group of 64 (K <= 64)
__kernel void dsp_select(__global const int* idx, __global const float* val, int row, int K, __global const half* F,
                         int R, __global const float* fb, __global const float* lat, __global int* ids,
                         __global const half* w1, __global float* w1row) {
  __local float bv[64];
  __local int bc[64], bk[64];
  const int k = get_local_id(0);
  float v = -INFINITY;
  int c = -1;
  if (k < K) {
    c = idx[(size_t)row * K + k];
    if (c >= 0) {
      float s = val[(size_t)row * K + k] + fb[c];
      for (int r = 0; r < R; ++r) s += vload_half((size_t)c * R + r, F) * lat[r];
      v = s;
    }
  }
  bv[k] = v, bc[k] = c, bk[k] = k;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (k < s) {
      const bool take = bv[k + s] > bv[k] || (bv[k + s] == bv[k] && bk[k + s] < bk[k]);
      if (take) bv[k] = bv[k + s], bc[k] = bc[k + s], bk[k] = bk[k + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const int best = bc[0] < 0 ? 0 : bc[0];
  if (k == 0) ids[row] = best;
  for (int r = k; r < R; r += 64) w1row[r] = vload_half((size_t)best * R + r, w1);
}

// u = u + b (+ h where has_h): the correction's vector, hidden_out lat + its bias (+ the row's own h: the exact chain)
__kernel void dsp_addb(__global float* u, __global const float* b, __global const float* h, int has_h, int E) {
  const int i = get_global_id(0);
  if (i >= E) return;
  u[i] = u[i] + b[i] + (has_h ? h[i] : 0.0f);
}
