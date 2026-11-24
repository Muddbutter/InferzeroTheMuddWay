// attn_fx: the prompt attention as flash attention on XMX for discrete Xe2 / Xe3 cards (16-lane int8 DPAS; issue #27,
// 2026-11-15; zero/xe2/engine_xe2.h attnfx_init). Reads the same int8 K / V cache, scales, int8 q and writes the same
// f32 output as attn_q8 (kernels/ops_attn.cl), with the work laid out the way FlashAttention-2 lays it out (vLLM's
// Xe2 chunk-prefill kernel, vllm-xpu-kernels chunk_policy_head256: 8 rows per sub-group, 32-key tiles):
//
//   a sub-group owns 8 query tokens of one q head (the 8 rows of a DPAS; attn_q8: 1 token x HS <= 4 heads, M = 4),
//   scores  S[8 tokens][16 keys]  += Q[8][32 dims] . K[32 dims][16 keys]   8 DPAS per 16-key block, K as the B operand
//                                                                          straight from the cache by block reads (its
//                                                                          [dim word][16 keys] layout is B's layout)
//   P . V   O[8 tokens][16 dims]  += P[8][32 keys] . V[32 keys][16 dims]   16 DPAS per 32 keys, V from the cache's
//                                                                          [dim][16 keys] bytes (B's layout too)
//
// The softmax is attn_q8's online softmax (log2 domain: log2(e) folded into the q scales), one step per 32 keys; P times
// the key's V scale is quantized to 0..255 against one bound per row and 32 keys (attn_q8: per 16 keys), so the result
// is not bit-identical to attn_q8 (KL against the plain path: 512-token chunks 0.00152 -> 0.00146, a 16K chunk
// 0.00141 -> 0.00138, docs/before-after.md). A work-group holds NT token groups x NH q heads of one KV head; its
// sub-groups read the same K / V blocks (through L1, no SLM, no barriers) and prefetch the next span's blocks into L1
// between them (without the prefetch 2.3x slower). The latest token groups (the most keys) start first. Plain causal
// prompts only: no window, no non-causal span, no sparse mask (attn_q8 keeps those).
// -D: HD (head dims, multiple of 32, <= 256), HPG (q heads per KV head), NT (token groups of 8 per work-group), NH (q
// heads per work-group, divides HPG), FX_PF (the L1 prefetch). grid (ceil(M / (8 NT)) * 16 NT NH, nkv * HPG / NH),
// local (16 NT NH, 1).
// Tried (a B70's 27B 16K prompt, attention 707 ms at 16 x 2): q read again every span instead of held (slower: 4K 64 ->
// 96 ms); the V loads in chunks behind sub-group barriers, V by block reads or vload4 (no change); prefetching 2 or 3
// spans ahead (no change). The ablations below (timing only) showed P . V's V loads, DPAS and float update each
// hidden behind the others: one removed 2-3 %, all three 36 %. Then at 618 ms: the 8 rows of P packed 4 to a word (8
// shuffles instead of fx_pair's 16, the bytes put together by upsample) 656 ms, slower; without exp2, without the row
// reductions: no faster; every span reading the same two blocks (L1 hits, the same bytes) 520-533 ms, so the misses
// beyond L1 cost ~15 % and the rest is the sub-group's own loads and arithmetic at 8 rows. What is left: more rows per
// sub-group for each K / V byte loaded (16 rows over half the head dims, the pairs adding their integer partial
// scores through SLM: 2.7-4x fewer bytes a row), at the edge of 256 registers (the output alone is 128): not tried.
// The work-group in step span by span (a split barrier around each span, as gemm_xw's XW_LOCK): 16K 629-632 ms
// against 619-624, slower; the sub-groups already run close together.
//
// IGC 2.41 (Battlemage) crashed ("Floating point exception") on this kernel with ushort block reads of q, and with a
// branch on "the running max grew" next to the row reduction: both avoided below.

#define FX_RT 8           // query tokens per sub-group (DPAS rows)
#define FX_KB (HD / 32)   // 32-dim slices of a row (scores: DPAS depth)
#define FX_DG (HD / 16)   // 16-dim groups of the output (P . V: DPAS columns)
#define FX_BU (4 * HD)    // uints in one 16-key block of K (and of V)
#define FX_NSG (NT * NH)  // sub-groups per work-group
#if HD % 32 || HD > 256 || HPG % NH
#error "attn_fx: HD a multiple of 32 up to 256, NH dividing HPG"
#endif

// The 8 rows' reduction over the 16 lanes at once (a transposing butterfly: 4 + 2 + 1 + 1 shuffles, then 8 to give
// every lane all rows; 8 separate sub-group reductions take ~32). After the butterfly lane l holds row (l >> 1)'s result.
#define FX_RED8_DEF(name, OP)                                                                     \
  inline float8 name(float8 v) {                                                                \
    const uint l = get_sub_group_local_id();                                                    \
    const bool b3 = (l & 8) != 0, b2 = (l & 4) != 0, b1 = (l & 2) != 0;                         \
    float c[4], d[2];                                                                           \
    c[0] = OP(b3 ? v.s4 : v.s0, intel_sub_group_shuffle_xor(b3 ? v.s0 : v.s4, 8));             \
    c[1] = OP(b3 ? v.s5 : v.s1, intel_sub_group_shuffle_xor(b3 ? v.s1 : v.s5, 8));             \
    c[2] = OP(b3 ? v.s6 : v.s2, intel_sub_group_shuffle_xor(b3 ? v.s2 : v.s6, 8));             \
    c[3] = OP(b3 ? v.s7 : v.s3, intel_sub_group_shuffle_xor(b3 ? v.s3 : v.s7, 8));             \
    d[0] = OP(b2 ? c[2] : c[0], intel_sub_group_shuffle_xor(b2 ? c[0] : c[2], 4));             \
    d[1] = OP(b2 ? c[3] : c[1], intel_sub_group_shuffle_xor(b2 ? c[1] : c[3], 4));             \
    float e = OP(b1 ? d[1] : d[0], intel_sub_group_shuffle_xor(b1 ? d[0] : d[1], 2));          \
    e = OP(e, intel_sub_group_shuffle_xor(e, 1));                                               \
    return (float8)(intel_sub_group_shuffle(e, 0), intel_sub_group_shuffle(e, 2), intel_sub_group_shuffle(e, 4), \
                    intel_sub_group_shuffle(e, 6), intel_sub_group_shuffle(e, 8), intel_sub_group_shuffle(e, 10), \
                    intel_sub_group_shuffle(e, 12), intel_sub_group_shuffle(e, 14));             \
  }
#define FX_ADD(a, b) ((a) + (b))
FX_RED8_DEF(fx_rmax8, fmax)
FX_RED8_DEF(fx_rsum8, FX_ADD)

// rows r, r + 1 of P (lane = key, 0..255) of the span's two 16-key blocks, as A operand halves: lane l < 8 keys 2l, 2l + 1
// of block 0, lanes 8..15 keys 2(l - 8), 2(l - 8) + 1 of block 1; low byte the even key, row r in the low 16 bits
inline uint fx_pair(int a0, int b0, int a1, int b1, int lane) {
  const uint w0 = (uint)a0 | ((uint)b0 << 16), w1 = (uint)a1 | ((uint)b1 << 16);
  const uint e0 = w0 | (intel_sub_group_shuffle_xor(w0, 1) << 8), e1 = w1 | (intel_sub_group_shuffle_xor(w1, 1) << 8);
  const int src = (2 * lane) & 15;
  const uint g0 = intel_sub_group_shuffle(e0, src), g1 = intel_sub_group_shuffle(e1, src);
  return lane < 8 ? g0 : g1;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void attn_fx(__global const char* qi, __global const float* qsc, __global const uint* kq,
                      __global const float* ksc, __global const uint4* vq, __global const float* vsc, int cap,
                      int nh, int pos0, int M_valid, __global float* out, int win, int nc_lo, int nc_hi) {
  (void)win; (void)nc_lo; (void)nc_hi;  // (attn_q8's arguments; the engine sends only plain causal prompts here)
  const int lid = get_local_id(0), sg = lid / 16, lane = lid % 16;
  const int tg = sg % NT, hh = sg / NT;
  const int wt = (int)get_num_groups(0) - 1 - (int)get_group_id(0);  // (the latest tokens, the most keys, first)
  const int g = (int)get_group_id(1) / (HPG / NH), h = g * HPG + ((int)get_group_id(1) % (HPG / NH)) * NH + hh;
  const int m0 = (wt * NT + tg) * FX_RT;
  if (m0 >= M_valid) return;
  const int nrow = min(FX_RT, M_valid - m0);
  const int pfirst = pos0 + m0, plast = pos0 + m0 + nrow - 1;
  const int nbc = cap / 16;
  const size_t blk0 = (size_t)g * nbc;

  // q as the scores' A operand, held in 32 registers: qa[kb] row r = token m0 + r's dims 32 kb + 2 lane, + 1 (one
  // ushort a lane per 32-dim slice); rows past M_valid zero
  short8 qa[FX_KB];
  {
    __global const ushort* qrow[FX_RT];
#pragma unroll
    for (int r = 0; r < FX_RT; ++r)
      qrow[r] = (__global const ushort*)(qi + ((size_t)(m0 + (r < nrow ? r : 0)) * nh + h) * HD) + lane;
#pragma unroll
    for (int kb = 0; kb < FX_KB; ++kb)
      qa[kb] = as_short8((ushort8)(qrow[0][kb * 16], 1 < nrow ? qrow[1][kb * 16] : (ushort)0,
                                   2 < nrow ? qrow[2][kb * 16] : (ushort)0, 3 < nrow ? qrow[3][kb * 16] : (ushort)0,
                                   4 < nrow ? qrow[4][kb * 16] : (ushort)0, 5 < nrow ? qrow[5][kb * 16] : (ushort)0,
                                   6 < nrow ? qrow[6][kb * 16] : (ushort)0, 7 < nrow ? qrow[7][kb * 16] : (ushort)0));
  }
  float8 qsv = (float8)0.0f;
  int8 posv = (int8)(-1);  // (a row past M_valid sees no key)
  {
    float qs_[FX_RT];
    int ps_[FX_RT];
#pragma unroll
    for (int r = 0; r < FX_RT; ++r) {
      const bool act = r < nrow;
      qs_[r] = act ? qsc[(size_t)(m0 + r) * nh + h] * M_LOG2E_F : 0.0f;
      ps_[r] = act ? pfirst + r : -1;
    }
    qsv = (float8)(qs_[0], qs_[1], qs_[2], qs_[3], qs_[4], qs_[5], qs_[6], qs_[7]);
    posv = (int8)(ps_[0], ps_[1], ps_[2], ps_[3], ps_[4], ps_[5], ps_[6], ps_[7]);
  }

  float8 o[FX_DG];
#pragma unroll
  for (int i = 0; i < FX_DG; ++i) o[i] = (float8)0.0f;
  float8 mx = (float8)(-INFINITY), lp = (float8)0.0f;
  const int bl = plast / 16;  // the last block any row of this sub-group sees
  const int nsp = plast / 32 + 1;

  for (int s = 0; s < nsp; ++s) {
    const int k0 = 32 * s, b0 = 2 * s, b1 = min(2 * s + 1, bl);  // (b1 past the last block: its keys are all masked)
    const size_t o0 = blk0 + (size_t)(b0 % nbc), o1 = blk0 + (size_t)(b1 % nbc);
    __global const uint* kp0 = kq + o0 * FX_BU;
    __global const uint* kp1 = kq + o1 * FX_BU;
#if defined(FX_PF) && defined(cl_intel_subgroup_buffer_prefetch)
    // the next span's K and V blocks (4 x 4 KB = 32 pieces of 512 B) into L1, the pieces shared by the sub-groups
    // (-DFX_PFD=N, 2026-11-23, ARC_ATTN_FX_PFD: the span N ahead, default 1 (the same blocks as before); a knob for the
    //  Max 1550, whose attn_fx runs at a third of a B70's rate, #89. Only which blocks go to L1: the same results)
#ifndef FX_PFD
#define FX_PFD 1
#endif
    if (s + FX_PFD < nsp) {
      const int nb0 = 2 * (s + FX_PFD), nb1 = min(2 * (s + FX_PFD) + 1, bl);
      for (int p = sg; p < 32; p += FX_NSG) {
        const size_t ob = blk0 + (size_t)(((p & 8) ? nb1 : nb0) % nbc);
        __global const uint* base = (p & 16) ? (__global const uint*)vq : kq;
        intel_sub_group_block_prefetch_ui8(base + ob * FX_BU + (p & 7) * 128);
      }
    }
#endif
    // scores: int8 DPAS sums of the two blocks, then q scale x k scale (log2 domain)
    int8 a0 = (int8)0, a1 = (int8)0;
#pragma unroll
    for (int kb = 0; kb < FX_KB; ++kb) {
#ifndef FX_ABL_QK  // (ablations, timing only, wrong results; ARC_ATTN_FX=2 skips the start-up check)
      a0 = intel_sub_group_i8_i8_matrix_mad_k32(qa[kb], as_int8(intel_sub_group_block_read8(kp0 + kb * 128)), a0);
      a1 = intel_sub_group_i8_i8_matrix_mad_k32(qa[kb], as_int8(intel_sub_group_block_read8(kp1 + kb * 128)), a1);
#else
      a0 ^= as_int8(intel_sub_group_block_read8(kp0 + kb * 128)) ^ convert_int8(qa[kb]);
      a1 ^= as_int8(intel_sub_group_block_read8(kp1 + kb * 128));
#endif
    }
    const float ks0 = ksc[o0 * 16 + lane], ks1 = ksc[o1 * 16 + lane];
    float8 s0 = convert_float8(a0) * qsv * ks0, s1 = convert_float8(a1) * qsv * ks1;
    const int key0 = k0 + lane, key1 = (b1 == 2 * s + 1) ? k0 + 16 + lane : INT_MAX;
    if (k0 + 31 > pfirst || b1 != 2 * s + 1) {  // (the causal edge, or a missing second block: mask per row)
      s0 = select((float8)(-INFINITY), s0, (int8)(key0) <= posv);
      s1 = select((float8)(-INFINITY), s1, (int8)(key1) <= posv);
    }
    // online softmax step per row. No branch on "the max grew" (the IGC crash): the rescale is folded into the P . V
    // update, o = o * corr + P . V, corr = 1 where the max stayed
    const float8 tm = fx_rmax8(fmax(s0, s1));
    const int8 grew = isgreater(tm, mx);
    const float8 mn = fmax(mx, tm);
    const float8 corr = select((float8)1.0f, native_exp2(mx - mn), grew);  // (mx = -inf: 0)
    lp *= corr;
    mx = mn;
    const float8 ms = select(mx, (float8)0.0f, isinf(mx));  // (a row that saw no key yet: its scores are -inf too)
    const float8 p0 = native_exp2(s0 - ms), p1 = native_exp2(s1 - ms);
    lp += p0 + p1;
    // P x V scale quantized to 0..255 against the row's bound over the span (tile max weight x largest V scale; the
    // reciprocal by native_recip: 16K attention 702 -> 618 ms, the same perplexity as the division)
    const float8 pb = native_exp2(tm - ms);
    const float vs0 = key0 <= plast ? vsc[o0 * 16 + lane] : 0.0f, vs1 = key1 <= plast ? vsc[o1 * 16 + lane] : 0.0f;
    const float vmt = sub_group_reduce_max(fmax(vs0, vs1));
    const float8 pmv = pb * vmt;
    const float8 inv = select((float8)0.0f, 255.0f * native_recip(pmv), isgreater(pmv, (float8)0.0f));
    const int8 q0 = min(convert_int8_rte(p0 * vs0 * inv), (int8)255), q1 = min(convert_int8_rte(p1 * vs1 * inv), (int8)255);
    const float8 ps = pmv * (1.0f / 255.0f);
    const uint pr0 = fx_pair(q0.s0, q0.s1, q1.s0, q1.s1, lane), pr1 = fx_pair(q0.s2, q0.s3, q1.s2, q1.s3, lane),
               pr2 = fx_pair(q0.s4, q0.s5, q1.s4, q1.s5, lane), pr3 = fx_pair(q0.s6, q0.s7, q1.s6, q1.s7, lane);
    const ushort8 pa = (ushort8)((ushort)pr0, (ushort)(pr0 >> 16), (ushort)pr1, (ushort)(pr1 >> 16), (ushort)pr2,
                                 (ushort)(pr2 >> 16), (ushort)pr3, (ushort)(pr3 >> 16));
    // P . V: lane = dim 16 i + lane, the 32 keys' bytes of that dim from the two blocks
    __global const uint4* vp0 = vq + o0 * (FX_BU / 4);
    __global const uint4* vp1 = vq + o1 * (FX_BU / 4);
#pragma unroll
    for (int i = 0; i < FX_DG; ++i) {
#if defined(FX_ABL_PV)  // (no V loads, no DPAS, no update)
      o[i] += ps;
#else
      const uint4 v0 = vp0[16 * i + lane], v1 = vp1[16 * i + lane];
#if defined(FX_ABL_PVD)  // (the loads, no DPAS)
      const int8 acc = as_int8((uint8)(v0, v1)) ^ convert_int8(pa);
#else
      const int8 acc = intel_sub_group_u8_i8_matrix_mad_k32(pa, as_int8((uint8)(v0, v1)), (int8)0);
#endif
#if defined(FX_ABL_UPD)  // (no rescale or float update)
      o[i] += as_float8(acc);
#else
      o[i] = fma(o[i], corr, convert_float8(acc) * ps);
#endif
#endif
    }
  }

  const float8 inv = (float8)1.0f / fx_rsum8(lp);
#define FX_WR(r, c)                                                                     \
  if (r < nrow) {                                                                       \
    __global float* dst = out + ((size_t)(m0 + r) * nh + h) * HD + lane;                \
    _Pragma("unroll") for (int i = 0; i < FX_DG; ++i) dst[16 * i] = o[i].c * inv.c;     \
  }
  FX_WR(0, s0) FX_WR(1, s1) FX_WR(2, s2) FX_WR(3, s3) FX_WR(4, s4) FX_WR(5, s5) FX_WR(6, s6) FX_WR(7, s7)
#undef FX_WR
}
