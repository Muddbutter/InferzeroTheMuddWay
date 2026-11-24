// H8 prompt GEMMs (QUEUE 31.12; zero/kernels/h8.cl, gemm.cl QTYPE 110): a part of class Engine (zero/engine.h),
// included inside the class body like zero/xe2/engine_xe2.h, not a header of its own.
//
// The text model's prompt GEMMs with weights and activations rotated by a Walsh-Hadamard transform in blocks of up to
// 1024 values along K, both rounded to int8 with one scale per 256-superblock: the GEMM is then pure int8 dot
// products, one float multiply-add per superblock (DP4a on the iGPU, int8 DPAS with -DXMX on Xe2 cards; the same
// kernel and launch as the W8 GEMM). Accuracy simulated against f32 (tools/ref_forward REF_HAD, per-row scales, so
// coarser than these per-256 ones): Qwen3.8 27B KL 0.00057 (the engine's own level), Gemma 4 E4B 0.00198.
//
// The weights come from the W8 / Q8_0 forms the engine already makes (decode_w8 just in time, or resident), converted
// in place once per decode (h8_weights); their H8 superblock scales go to a header buffer of their own (a resident
// weight's in h8_res_, a staged one's per role in stage_hh_). The activations stay as the fused norm / quantization
// kernels write them (aq_ / asc_, per-32 or per-256 scales) and are rotated into h8_aq_ / h8_asc_ before each GEMM
// (h8_act: 14 % of the GEMMs' time on a B60, 2026-11-06).
//
// H8R, per-column and per-token scales (gemm.cl QTYPE 111, kernels h8r_* of h8.cl, the default within ARC_H8=1;
// ARC_H8_SB=1 keeps the per-256 ones, QTYPE 110): a weight's column scales are computed once from its first decode
// (h8_col_, N floats) and each token's scale in the rotation's one pass (h8_asc_[token]), the rotations in registers,
// and a GEMM's dot products over the whole of K are one integer sum. See h8_float for the activations' source.
//
// Off unless ARC_H8=1 until it is measured; never with ARC_NPU (shared weight buffers); never with a two-pass model (q2mm_: Gemma 4 26B / 12B, Qwen3 ASR's text
// model), never for a mixture of experts' expert GEMMs (their own kernels) or the vision / audio encoders (only the
// text model's named matrix roles). On a split model each shard rotates in blocks dividing its own K: the rotation is
// per matrix input, so the shards' parts still sum to the product. ARC_H8_HB=256 / 512: smaller blocks (tests).

  bool h8_ = false;
  std::string h8_src_, x8_src_, rot_src_;
  cl_mem h8_aq_ = nullptr, h8_asc_ = nullptr;   // (one GEMM input rotated: int8 [M][K], per-256 scales [K / 256][M])
  cl_mem stage_hh_[kRoles] = {};                // (the H8 scales of the weight staged per role)
  bool stage_h8_[kRoles] = {};                  // (the staged weight is in its rotated H8 form, not W8: the 140T preset mixes both)
  int stage_packed_[kRoles] = {};               // (that H8 form in gemm_xw8's packed order, ARC_GEMM_XW_PACK: engine_xw.h; 1, or
                                                //  ARC_GEMM_XW_WGL's block of groups G > 1)
  size_t stage_hh_bytes_[kRoles] = {};
  std::map<const GpuWeight*, cl_mem> h8_res_;   // (a resident weight converted in place: its H8 scales)

  // ARC_H8_Q2=1 (2026-11-21, opt-in on every GPU where H8 runs; QUEUE "H8's flips", step 1, the user: "Add opt-in
  // pass"): H8's activations in two int8 planes, the value's code and its residual (kernels/h8.cl h8r_act -DQ2), each
  // H8 GEMM run a second time on the residual and added (h8_matmul). H8's own rounding doubled the KL against f32 on
  // every GPU (the laptop's 27B 0.00056 plain -> 0.00112 H8); this is for exact work, at ~2x the H8 GEMMs' time. The
  // fused forms that cannot take two passes fall back to their separate kernels while it is on: the norm and the
  // rotation in one (ARC_H8_NORM), the SwiGLU in the up GEMM's epilogue, gate + up in one launch. H8R only (one scale
  // per token and column), not ARC_H8_SB. Off: every kernel and its build options as before
  // (this engine's: EngineOptions::h8_q2 when a model-list variant set it, else ARC_H8_Q2; read once an engine: it is
  //  asked on every GEMM)
  static bool h8_q2_var() {
    static const bool on = getenv("ARC_H8_Q2") && atoi(getenv("ARC_H8_Q2")) != 0;
    return on;
  }
  mutable int h8q2_ = -1;
  bool h8_q2() const {
    if (h8q2_ < 0) h8q2_ = (opt.h8_q2 >= 0 ? opt.h8_q2 != 0 : h8_q2_var()) && h8_row() ? 1 : 0;
    return h8q2_ == 1;
  }
  cl_mem h8_aq2_ = nullptr, h8_asc2_ = nullptr;  // (ARC_H8_Q2: the residual plane, as h8_aq_ / h8_asc_)
  size_t h8_aq_bytes_ = 0;
  cl_mem h8_aq_un_ = nullptr;  // (ARC_GEMM_XW_WGL: an input in the block order copied back to the packed one, for a GEMM
                               //  the big tile does not take; made the first time one needs it, engine_xw.h xw_unwgl)
  bool h8_q2_pass_ = false;                      // (the second pass is running: the rotation is already made)
  static bool h8_env() {
    static const bool on = getenv("ARC_H8") && atoi(getenv("ARC_H8")) != 0;
    return on;
  }
  static bool h8_row() {  // (one scale per column and per token; ARC_H8_SB=1: per 256 values)
    static const bool sb = getenv("ARC_H8_SB") && atoi(getenv("ARC_H8_SB")) != 0;
    return !sb;
  }
  static int h8_qtype() { return h8_row() ? 111 : 110; }
  // H8R rotates a Qwen model's activations as they are before their first rounding: quant_store16 keeps a float copy
  // after the int8 rows of aq_, in the four kernels of build_kernels (rmsnorm_quant, swiglu_quant, gated_norm_quant,
  // attn_gate_quant; not the encoders', whose aq is a buffer of their own). Their int8 rows stay for the GEMMs H8 does
  // not cover and the generation path. Gemma and Flash Next have quantization kernels of their own and rotate the
  // int8 form. ARC_H8_INT=1: the int8 form here too.
  std::string h8f_defs_;
  size_t h8f_off_ = 0;
  bool h8_float() const {
    static const bool i8 = getenv("ARC_H8_INT") && atoi(getenv("ARC_H8_INT")) != 0;
    return h8_row() && !i8 && !hp.hc && !hp.gemma;
  }
  // (q2_init has run: q2mm_ known)
  void h8_init() {
    // (never with the NPU: its shares copy a weight (N.gate = L.gate) over the same buffers, so the in-place conversion
    //  ran twice on the shared rows, and the NPU read its own rows as W8 after they had become H8; kernel audit 2026-11-03)
    // (p140_: the Arc 140T preset; h8_auto_: the default on discrete cards for Qwen dense models; zero/xe2/engine_xe2.h)
    h8_ = (h8_env() || p140_ || h8_auto_) && !q2mm_ && !npu_wanted();
    if (h8_ && h8_float()) {
      h8f_off_ = (size_t)max_chunk * aq_max_k();
      h8f_defs_ = " -DH8F=" + std::to_string(h8f_off_);
    }
    if (h8_env() && q2mm_ && !opt.dry)
      zlog::eprintf("inferzero: ARC_H8 is not for a two-pass model: its prompt GEMMs stay as they are\n");
    if (h8_env() && !q2mm_ && npu_wanted() && !opt.dry)
      zlog::eprintf("inferzero: ARC_H8 is off with ARC_NPU (both convert the same weights)\n");
    // (H8 is this GPU's default, and the NPU turns it off: a B390's 27B 4K prompt 628 tok/s alone, 194-330 with the
    //  NPU at every share from 0.12 to 0.5; gee456, issue #31, 2026-11-18)
    if (!h8_env() && (p140_ || h8_auto_) && !q2mm_ && npu_wanted() && !opt.dry && first_for_device("h8_npu"))
      zlog::eprintf("inferzero: the NPU turns off H8, this GPU's prompt path: the GPU alone is likely faster "
                    "(an Arc B390's 27B 4K prompt 628 tok/s alone, at most 330 with the NPU)\n");
    // (once per kind of GPU: a split model's shards each init, one kind; a layer split of unlike cards says it for each.
    //  bench/gemm_sweep.ps1 reads this line)
    if (h8_ && !opt.dry && first_for_device("h8_on"))
      zlog::eprintf("inferzero: H8 on for the prompt GEMMs (Hadamard-rotated int8, one scale per %s)\n",
                    h8_row() ? "column and token" : "256 values");
    if (h8_ && h8f_off_ && !opt.dry && first_for_device("h8f"))
      zlog::eprintf("inferzero: H8 rotates the activations from f32\n");
  }
  // the rotation block for a row length K: the largest power of two up to 1024 (ARC_H8_HB) dividing it; K is a whole
  // number of 256-superblocks, so at least 256
  static int h8_block(int K) {
    static const int cap = [] {
      const int v = getenv("ARC_H8_HB") ? atoi(getenv("ARC_H8_HB")) : 1024;
      return v == 256 || v == 512 ? v : 1024;
    }();
    int hb = 256;
    while (hb * 2 <= cap && K % (hb * 2) == 0) hb *= 2;
    return hb;
  }
  bool h8_ok(const GpuWeight& w) const {
    return h8_ && w.role >= 0 && w.role <= 10 &&  // (weight_role's named matrices of the text model)
           (w.gemm_qtype == (int)GGML_Q8_0 || (w.gemm_qtype >= 100 && w.gemm_qtype <= 103)) && w.K % 256 == 0 && w.N % 16 == 0 &&
           (!h8_row() || (w.K / h8_block(w.K) <= 32 && w.K <= 130000));  // (H8R: red[32]; the sum over K within int32)
  }
  // its buffers with the others (alloc_buffers), so the memory plan counts them; a dry run too
  void h8_reserve(size_t M, int maxK) {
    if (!h8_) return;
    h8_aq_ = buf(M * (size_t)maxK);
    h8_aq_bytes_ = M * (size_t)maxK;
    h8_aq_un_ = nullptr;  // (ARC_GEMM_XW_WGL's copy: made again at the new size when a GEMM needs it)
    h8_asc_ = buf(M * (size_t)(maxK / 256) * 4);
    if (h8_q2()) {  // (the second plane: as big as the first; only with ARC_H8_Q2=1)
      h8_aq2_ = buf(M * (size_t)maxK);
      h8_asc2_ = buf(M * (size_t)(maxK / 256) * 4);
    }
    if (h8_row()) return;
    for (int r = 0; r < kRoles; ++r)  // (a role's H8 scales: N * K / 256 floats, an eighth of its W8 header at most)
      if (stage_need_h_[r]) stage_hh_[r] = buf(stage_hh_bytes_[r] = stage_need_h_[r] / 8 + 64);
  }
  cl_kernel h8_kernel(int hb, int src, const char* name) {
    return kernel("h8", "-DHB=" + std::to_string(hb) + (src ? " -DSRC=" + std::to_string(src) : std::string()), name);
  }
  // (H8R: 64 floats x 16 lanes in registers; ARC_H8_D=1: each value's sign flipped by its index first, which gained
  //  nothing on our models)
  // The register file of the H8R rotation kernels (h8r_colmax / h8r_weights / h8r_act, h8r_decode, rmsnorm_h8r): 256
  // registers by default. ARC_H8_GRF=128 (opt-in, 2026-11-17): 128 where a register holds 16 floats (Xe2 / Xe3, the
  // B-series and the B390), twice the threads per core; ocloc bmg / ptl: no spill in any of them at 128 (ptl takes 96).
  // GPUs whose registers hold 8 floats (Xe-LPG: the 125H, the 140T and 285H, told by their 8-lane sub-groups, cl_.sg8) keep
  // 256: 8-16 KB spilled at 128. The register file changes no arithmetic. ARC_H8_GRF=-128: 128 on any GPU (tests:
  // the 128-register builds' results on a GPU at hand, spills and all).
  // 128 by default on discrete cards with 16-lane registers since 2026-11-17: a Pro B70's 27B 4K prompt 2,725 ->
  // 2,772 tok/s (h8_decode 61.9 -> 43.7 ms, h8_act 95.0 -> 88.5), the same logits (joshdev-de, issue #27, release
  // 2026.11.17.03); the B60 is the same chip. Lunar / Panther Lake keep 256 until a B390 measures it. ARC_H8_GRF=256: 256
  std::string h8_grf() const {
    // (the switch read once; the default per call, so each shard of a split over unlike GPUs decides for its own, not
    //  the first one asking: the Researcher, 2026-11-20)
    static const int env = getenv("ARC_H8_GRF") ? atoi(getenv("ARC_H8_GRF")) : 0;
    const int e = env ? env : (zrt::integrated() ? 256 : 128);
    return (e == 128 && !cl_.sg8) || e == -128 ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
  }
  cl_kernel h8r_kernel(int hb, int src, const char* name, const std::string& extra = std::string()) {
    static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;
    return kernel("h8", h8_grf() + (d ? "-DH8_D " : "") + "-DHB=" + std::to_string(hb) +
                            (src ? " -DSRC=" + std::to_string(src) : std::string()) + extra, name);
  }
  // H8R: a weight's column scales (N floats), computed once from its decoded form, then the weight rotated in place
  std::map<const GpuWeight*, cl_mem> h8_col_;
  // (the resident copies (w.wq) rotated in place for H8R: their own record, not h8_col_'s entries, which the staged path
  //  also makes (its column scales, reused at each decode of the stage buffer). 2026-11-23, StealthDrifter's B580, #80:
  //  the start replay (h8_e2e_check) ran layer 0's qkv through the staged path, w8_part_auto then made it resident from a
  //  fresh decode, and the resident branch took the scales' entry for "rotated": rotated activations times unrotated
  //  weights, every prompt of 48+ tokens wrong. w8_part_auto drops a weight from this record when it decodes it anew)
  std::set<const GpuWeight*> h8_res_rot_;
  cl_mem h8r_weights(const GpuWeight& w, cl_mem wq, cl_mem wh, bool packed = false) {
    const int hb = h8_block(w.K);
    int K = w.K;
    size_t gws[2] = {(size_t)w.N / 16 * 256, (size_t)(w.K / hb)}, lws[2] = {256, 1};
    auto it = h8_col_.find(&w);
    if (it == h8_col_.end()) {
      const std::vector<uint32_t> zero((size_t)w.N, 0u);
      cl_mem col = buf(zero.size() * 4, zero.data());
      cl_kernel km = h8r_kernel(hb, w.gemm_qtype, "h8r_colmax");
      set_args(km, 0, wq, wh, col, K);
      run("h8_colmax", km, 2, gws, lws);
      it = h8_col_.emplace(&w, col).first;
    }
    cl_kernel k = h8r_kernel(hb, w.gemm_qtype, "h8r_weights", packed ? std::string(" -DXW_PACKW_OUT") : std::string());
    set_args(k, 0, wq, wh, it->second, K);
    run("h8_weights", k, 2, gws, lws);
    return it->second;
  }
  // H8R straight from a staged weight's raw rows (kernels/decode.cl h8r_decode): decode_w8 + h8r_weights in one pass,
  // the W8 copy neither written nor read again (a B60's 27B 4K prompt: decode_w8 189 ms + h8_weights 372 ms). The
  // same values bit for bit; for the types decode_w8 stores without requantizing and with int8 scales, rotation
  // blocks of 512 / 1024, once the column scales are known (the first decode keeps the two passes). ARC_H8_FUSE=0: off.
  bool h8_dec_ok(const GpuWeight& w) const {
    static const bool on = !getenv("ARC_H8_FUSE") || atoi(getenv("ARC_H8_FUSE")) != 0;
    // (since 2026-11-14 the per-32 float types too (W8 variant 4, the Q8_0 form, gemm_qtype 8: Q8_0, Q4_0, IQ4_NL, Q2_0,
    //  MXFP4), B60 prefill plan 13.3; the requantized ones (Q2_K / Q4_K / Q5_K) keep the two passes, kernels/decode.cl
    //  H8RDEC says why; ARC_H8_FUSE=1: only the types of before)
    static const bool old_set = getenv("ARC_H8_FUSE") && atoi(getenv("ARC_H8_FUSE")) == 1;
    const bool type_ok = !w8_requant(w.type) &&
                         ((w.gemm_qtype >= 100 && w.gemm_qtype <= 103) || (!old_set && w.gemm_qtype == GGML_Q8_0 && w8_variant(w.type) == 4));
    return on && w.raw && type_ok && w.type != GGML_IQ1_M && w.type != GGML_IQ1_S && h8_block(w.K) >= 512 && w.N % 16 == 0;
  }
  void h8r_decode(const GpuWeight& w, cl_mem wq, cl_mem col, int packed = 0) {  // (packed: as h8_prepare's)
    static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;
    // (ARC_H8_DEC_PRIV=1: each lane loads its superblock into registers, as before 2026-11-14; the default stages the
    //  work-group's superblocks in local memory once, B60 prefill plan 13.3, the same values bit for bit)
    static const bool priv = getenv("ARC_H8_DEC_PRIV") && atoi(getenv("ARC_H8_DEC_PRIV")) != 0;
    // (ARC_H8_DEC_R=n: n rotation blocks per work-group, 1 / 2 / 4 / 8)
    static const int rb = [] {
      const int e = getenv("ARC_H8_DEC_R") ? atoi(getenv("ARC_H8_DEC_R")) : 4;
      return e == 1 || e == 2 || e == 4 || e == 8 ? e : 4;  // (4: the laptop 27B, 1,180 -> 953 (1) -> 876 ms (4), 2026-11-14)
    }();
    const int hb = h8_block(w.K);
    // (tiled IQ3_S rows, ARC_IQ3S_TILE: -DIQ3T, never with H8D_PRIV, whose one-superblock copy cannot hold a pair)
    const bool pv = priv && !w.iq3t;
    cl_kernel k = kernel("decode", h8_grf() + "-DH8RDEC" + (d ? " -DH8_D" : "") + (pv ? " -DH8D_PRIV" : "") + (w.iq3t ? " -DIQ3T" : "") +
                                       " -DH8DR=" + std::to_string(rb) + " -DQT=" +
                                       std::to_string(w.type) + " -DHB=" + std::to_string(hb) + " -DSRC=" +
                                       std::to_string(w.gemm_qtype) + (packed ? " -DXW_PACKW_OUT" : "") +
                                       (packed > 1 ? " -DXW_WGLB=" + std::to_string(packed) : std::string()), "h8r_decode");
    int N = w.N, K = w.K;
    set_args(k, 0, w.raw, N, K, tabs_, wq, col);
    size_t gws[2] = {(size_t)N / 16 * 256, (size_t)((K / hb + rb - 1) / rb)}, lws[2] = {256, 1};
    run("h8_decode", k, 2, gws, lws);
  }
  // W8 / Q8_0 weights (wq, wh) into H8 in place, their scales into hh
  void h8_weights(const GpuWeight& w, cl_mem wq, cl_mem wh, cl_mem hh) {
    const int hb = h8_block(w.K);
    cl_kernel k = h8_kernel(hb, w.gemm_qtype, "h8_weights");
    int N = w.N, K = w.K;
    set_args(k, 0, wq, wh, hh, N, K);
    size_t gws[2] = {(size_t)N * (hb / 4), (size_t)(K / hb)}, lws[2] = {(size_t)hb / 4, 1};
    run("h8_weights", k, 2, gws, lws);
  }
  // GEMMs in a row on the same input (q / k / v, gate / up, qkv / z) rotate it once (QUEUE 33.3): the caller holds an
  // H8Share around them, nothing in its scope writes aq_ / asc_, and a later GEMM with the same rows, K and scale
  // form finds h8_aq_ / h8_asc_ as the first one left them. The same kernel on the same input: bit-identical.
  // ARC_H8_REUSE=0 rotates before each GEMM as before.
  int h8_share_ = 0, h8_last_M_ = -1, h8_last_K_ = -1, h8_last_a32_ = -1;
  cl_mem h8_last_aq_ = nullptr;  // (the input it was rotated from: a two-pass model swaps aq_)
  // (a Data Center GPU Max: off by default, see xehpc_on in zero/xe2/engine_xe2.h)
  bool h8_reuse_env() const {
    static const char* e = getenv("ARC_H8_REUSE");
    return e ? atoi(e) != 0 : !xehpc_on();
  }
  struct H8Share {
    Engine* e;
    explicit H8Share(Engine* en) : e(en) {
      e->h8_share_ = e->h8_reuse_env() ? 1 : 0;
      e->h8_last_K_ = -1;
    }
    ~H8Share() {
      e->h8_share_ = 0;
      e->h8_last_K_ = -1;
      e->h8_pre_live_ = false;  // (rmsnorm_h8's rotation is good for this scope's GEMMs only)
    }
    H8Share(const H8Share&) = delete;
    H8Share& operator=(const H8Share&) = delete;
  };
  // The FFN's down GEMM in H8 from f32 (QUEUE 33.3): when the up GEMM's epilogue wrote silu(gate) * up to big_
  // ([rows][F] f32), those are the values swiglu_quant would only copy to aq_ + H8F (fused: v = the GEMM's value, the
  // same row stride F), and an H8 GEMM reads neither aq_'s int8 rows nor asc_ / absum_ (no weight minimum): the
  // rotation reads big_ itself and swiglu_quant is left out. Bit-identical; ARC_H8_SWI=0 as before.
  // h8_takes: matmul1's prompt path takes the H8 GEMM for w (its first branch past the GEMVs)
  bool h8_takes(const GpuWeight& w, int Mp) const {
    return cur_T_ > 8 && !q2mm_ && h8_ok(w) && (!p140_ || h8_env() || p140_take(w, Mp)) && (w.wq || h8_igpu_rows_ok(Mp));
  }
  // (2026-11-19) iGPUs with 16-lane XMX (Lunar Lake's Arc 140V, Panther Lake's B390) take the default H8 from chunks
  // of ARC_H8_MIN rows (default 256), shorter chunks the plain XMX GEMM. a1ex4's 140V (issue #60), the 27B, --bench-pp
  // best of 3, H8 / without: 128 tokens 86 / 89-93 tok/s, 256 156 / 96, 512 252 / 176, 1K 339 / 185, 4K 384 / 170: the
  // lines cross at ~140 tokens (a 128-token prompt spends 59 % of its time decoding weights for H8). Accuracy: KL against
  // f32 on 480 tokens 0.00117 (max 0.029, top token 470 / 480) with H8, 0.00057 (473 / 480) without: both far inside
  // 0.015, and generation never runs H8. (The first threshold, 1,024, came from cold one-shot runs and made 256-1,023
  // tokens 1.4-1.8x slower: release .05.) Discrete cards and ARC_H8=1 as before; the 125H never runs H8.
  // (Weights resident in W8 always take H8 (h8_takes): h8_prepare rotates a resident copy in place the first time, so
  //  the plain GEMM would then read rotated weights with unrotated activations, wrong logits on a short chunk after a
  //  long one. Self audit 2026-11-20, A1; the 140T preset's resident form keeps its copy unrotated instead, p140_res_)
  bool h8_igpu_rows_ok(int Mp) const {
    if (!h8_auto_ || !zrt::integrated() || h8_env()) return true;
    static const int min_rows = getenv("ARC_H8_MIN") ? std::max(0, atoi(getenv("ARC_H8_MIN"))) : 256;
    return Mp >= min_rows;
  }
  bool h8_swi_ok(const GpuWeight& down, int Mp) const {
    static const bool on = !getenv("ARC_H8_SWI") || atoi(getenv("ARC_H8_SWI")) != 0;
    return on && h8_row() && !h8f_defs_.empty() && h8_takes(down, Mp);
  }
  cl_mem h8_fsrc_ = nullptr;  // (set: the next rotations read f32 rows of K from here, not aq_ + H8F; H8Src)
  struct H8Src {
    Engine* e;
    H8Src(Engine* en, cl_mem src) : e(en) { e->h8_fsrc_ = src; }
    ~H8Src() { e->h8_fsrc_ = nullptr; }
    H8Src(const H8Src&) = delete;
    H8Src& operator=(const H8Src&) = delete;
  };
  // ---- the norm and the rotation of its rows in one kernel (B60 prefill plan 13.5; ops.cl rmsnorm_h8r; ARC_H8_NORM,
  // on by default on discrete cards and in the 140T preset, see h8_norm_env): the norm of a layer's input runs inside the H8Share scope of the GEMMs that read
  // it and leaves h8_aq_ / h8_asc_ rotated (h8_pre_live_), so their h8_act has nothing left to do; the scope's end or
  // any other rotation ends it. The same values bit for bit as rmsnorm_quant + h8r_act from the f32 copy (H8F), which
  // it no longer writes: the copy's write and read and a launch saved per input (the 27B: 128 of a prompt's 256 h8_act)
  bool h8_pre_live_ = false;
  int h8_pre_M_ = -1, h8_pre_K_ = -1;
  // (on by default in the Arc 140T preset since 2026-11-17: a 140T's 27B 4K prompt 368.3 -> 382.1 tok/s, logits
  //  identical (PascalSoftwares, issue #5, release .07). On discrete cards since 2026-11-18: a Pro B70's 27B 4K prompt
  //  2,890 -> 2,972 tok/s, h8_act 88.4 -> 60.3 ms and the norm 69.7 -> 60.0, hashes equal (joshdev-de, issue #27,
  //  7368768); the B60's no gain of 11-14 was before the 128-register default. iGPUs off; =0 off, =1 on)
  bool h8_norm_env() const {
    static const char* e = getenv("ARC_H8_NORM");
    return e ? atoi(e) != 0 : p140_ || (!zrt::integrated() && !xehpc_on());  // (a Max: off, see xehpc_on)
  }
  // rmsnorm_quant for the input of `next` (the first GEMM of the open H8Share scope that reads it); rmsnorm_quant as
  // before where the fused form does not apply
  void rmsnorm_h8(cl_mem x, int row_off, cl_mem w, int K, int Mp, int T, int act32, const GpuWeight& next) {
    XwRole xr(this, next.role);  // (the input in the order of the GEMM that reads it)
    const int hb = h8_block(K);
    if (!h8_norm_env() || h8_q2() || !h8_share_ || !h8_row() || h8f_defs_.empty() || wide_rows(T) || !h8_takes(next, Mp) ||
        next.K != K || K % 256 || K % hb || K / hb > 16 || (size_t)K * 4 + 64 > 60 * 1024) {  // (the row in local memory)
      rmsnorm_quant(x, row_off, w, K, Mp, T, act32);
      return;
    }
    static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;
    // (gemm_xw8's packed order where it will read them, as h8_act writes it: since 2026-11-17, so the 140T preset, whose
    //  GEMMs read the packed order, can take the fused norm; before, ARC_H8_NORM fell back to rmsnorm_quant there)
    cl_kernel k = kernel("ops", h8_grf() + "-DH8NORM" + (d ? " -DH8_D" : "") + " -DHB=" +
                                    std::to_string(hb) + " -DH8N_K=" + std::to_string(K) +
                                    (xw_pack_a(Mp, K) ? " -DXW_PACKA_OUT" : "") + xw_wgla_defs(Mp, K), "rmsnorm_h8r");
    float eps = hp.eps;
    int wh = 0, KK = K, MM = Mp, TT = T, a32 = act32;
    set_args(k, 0, x, row_off, w, KK, MM, TT, eps, aq_, asc_, absum_, big_, wh, a32, h8_aq_, h8_asc_);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("rmsnorm_h8r", k, 2, gws, lws);
    h8_pre_live_ = true, h8_pre_M_ = Mp, h8_pre_K_ = K;
  }
  // ---- ARC_H8_OUT=1 (2026-11-22, opt-in; the A770 plan's H8 passes): the inputs of the DeltaNet and attention output
  // GEMMs (ssm_out, wo) rotated by the kernels that make them, as rmsnorm_h8 does for the norms: ops.cl gated_norm_h8r
  // and ops_attn.cl attn_gate_h8r keep the row in local memory and rotate it there (h8n_rotate), so the f32 copy
  // (H8F: K x 4 bytes a token, 100 MB a layer of the 27B at 4K) is neither written nor read back and h8_act's launch
  // goes (the 27B: 64 of a prompt's 128 h8_act). The same floats, transform and scale: h8_aq_ / h8_asc_ bit for bit.
  // Where it applies: H8R with the f32 copy, one GEMM taking it, a whole row in a work-group (the same limits as
  // rmsnorm_h8); the caller holds an H8Share around the kernel and the GEMM
  // (on by default on discrete 8-lane cards, Alchemist, since 2026-11-23: skipzx-lab's A770, #45, h8_act 101 -> 75 ms, the 27B
  //  4K +0.7 %; =0 off, =1 on elsewhere)
  bool h8_out_env() const {
    static const int e = getenv("ARC_H8_OUT") ? atoi(getenv("ARC_H8_OUT")) : -1;
    return e > 0 || (e < 0 && cl_.sg8 && !zrt::integrated());
  }
  bool h8_out_ok(const GpuWeight& next, int K, int Mp, int T) const {
    const int hb = h8_block(K);
    return h8_out_env() && !h8_q2() && h8_row() && !h8f_defs_.empty() && !wide_rows(T) && h8_takes(next, Mp) &&
           next.K == K && K % 256 == 0 && K % hb == 0 && K / hb <= 16 && (size_t)K * 4 + 64 <= 60 * 1024;
  }
  // (name: gated_norm_h8r or attn_gate_h8r; args: the kernel's own up to act32, set by the caller after this builds it)
  cl_kernel h8_out_kernel(const char* name, int K, int Mp, int role) {
    XwRole xr(this, role);  // (the input in the order of the GEMM that reads it, as rmsnorm_h8)
    static const bool d = getenv("ARC_H8_D") && atoi(getenv("ARC_H8_D")) != 0;
    return kernel("ops", h8_grf() + "-DH8NORM" + (d ? " -DH8_D" : "") + " -DHB=" + std::to_string(h8_block(K)) +
                             " -DH8N_K=" + std::to_string(K) + (xw_pack_a(Mp, K) ? " -DXW_PACKA_OUT" : "") +
                             xw_wgla_defs(Mp, K), name);
  }
  void h8_out_run(const char* name, cl_kernel k, int K, int Mp) {
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run(name, k, 2, gws, lws);
    h8_pre_live_ = true, h8_pre_M_ = Mp, h8_pre_K_ = K;
  }
  // this GEMM's input (aq_ / asc_, per-32 scales when act32) rotated into h8_aq_ / h8_asc_
  // (the guard against a GEMM reading a stale input, 2026-11-23: 326febb merged this call into a comment and every H8
  //  prompt GEMM but the fused gate / up read the previous one, wrong prompts on discrete cards in release .16, unseen
  //  by the start checks, whose inputs come from the host. h8_act counts its calls, whether it rotates or finds the input
  //  ready; an H8 GEMM launch without one since its call began throws: h8_input_check)
  uint64_t h8_act_calls_ = 0;
  // (and at start, after the buffers: one H8 prompt GEMM replayed end to end through the engine's own path (the norm and
  //  its quantization into aq_, h8_act, the GEMM) on random rows: input A, then B, then A again. A's two results must
  //  be equal bit for bit and B's clearly different, which a stale or misrouted input breaks; the GEMM kernels' own
  //  start checks feed them from the host and could not see it. A failure stops the load with an error (not H8 turned
  //  off: the replay may have rotated a resident weight in place for H8, which the DP4a GEMMs would then read wrong);
  //  ARC_H8=0 runs without H8. The first layer's first H8 matrix, chunks of up to 256 tokens; dense Qwen only.
  //  ARC_H8_E2E=0: skipped)
  void h8_e2e_check() {
    static const bool off = getenv("ARC_H8_E2E") && atoi(getenv("ARC_H8_E2E")) == 0;
    if (off || !h8_ || opt.dry || layers_.empty() || hp.moe || hp.hc || hp.gemma || hp.ds41 || hp.inkling || !hp.n_embd) return;
    Layer& L = layers_[0];
    const GpuWeight* w = L.qkv.N ? &L.qkv : L.wq.N ? &L.wq : nullptr;
    const int Mp = std::min(max_chunk, 256) / 32 * 32, E = hp.n_embd;
    if (!w || !L.attn_norm || Mp < 32 || w->K != E) return;
    struct Keep { int& r; int v; ~Keep() { r = v; } } keep{cur_T_, cur_T_};
    cur_T_ = Mp;
    if (!h8_takes(*w, Mp)) return;
    const int a_in = act32_for(false), N = w->N;
    cl_int err;
    cl_mem C = clCreateBuffer(cl_.ctx, CL_MEM_READ_WRITE, (size_t)Mp * N * 4, nullptr, &err);
    CL_CHECK(err);
    struct Rel { cl_mem m; ~Rel() { clReleaseMemObject(m); } } rel{C};
    uint32_t seed = 4711;
    auto rows = [&] {
      std::vector<float> x((size_t)Mp * E);
      for (auto& v : x) { seed = seed * 1664525u + 1013904223u; v = (float)((int)(seed >> 9) % 2001 - 1000) / 500.0f; }
      return x;
    };
    const std::vector<float> xa = rows(), xb = rows();
    auto pass = [&](const std::vector<float>& x) {
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, x.size() * 4, x.data(), 0, nullptr, nullptr));
      rmsnorm_quant(x_, 0, L.attn_norm, E, Mp, Mp, a_in);
      matmul("h8_e2e", *w, Mp, C, N, 0, 0, a_in);
      std::vector<float> c((size_t)Mp * N);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, C, CL_TRUE, 0, c.size() * 4, c.data(), 0, nullptr, nullptr));
      return c;
    };
    // (the state the replay leaves as it found it: the column-scale entries and the stage it made are dropped, so a
    //  later residency change (w8_part_auto) or the first real prompt starts as without the check; #80)
    std::set<const GpuWeight*> had;
    for (const auto& e : h8_col_) had.insert(e.first);
    struct Restore {
      Engine* e;
      std::set<const GpuWeight*>& had;
      ~Restore() {
        // (only the staged path's entries: a resident copy the replay rotated stays rotated in place, so its entry and
        //  its mark in h8_res_rot_ stay too; erasing them rotated it a second time at the first prompt (the laptop's 4B,
        //  W8 resident from load, H8 forced: f903415's first form))
        for (auto it = e->h8_col_.begin(); it != e->h8_col_.end();) {
          if (had.count(it->first) || e->h8_res_rot_.count(it->first)) { ++it; continue; }
          e->free_buf(it->second);
          it = e->h8_col_.erase(it);
        }
        for (int r = 0; r < kRoles; ++r) e->stage_owner_[r] = nullptr, e->stage_h8_[r] = false;
      }
    } restore{this, had};
    const std::vector<float> c1 = pass(xa), c2 = pass(xb), c3 = pass(xa);
    double d = 0, n = 0;
    bool finite = true;
    for (size_t i = 0; i < c1.size(); ++i) {
      finite = finite && std::isfinite(c1[i]) && std::isfinite(c2[i]);
      d += (double)(c1[i] - c2[i]) * (c1[i] - c2[i]);
      n += (double)c1[i] * c1[i];
    }
    const bool same = memcmp(c1.data(), c3.data(), c1.size() * 4) == 0, moved = n > 0 && d > 0.0025 * n;
    if (finite && same && moved) return;
    throw std::runtime_error(std::string("the H8 prompt GEMM replayed at start does not follow its input (") +
                             (!finite ? "not finite" : !same ? "the same input gave two results" : "another input gave the same result") +
                             "): please report it; ARC_H8=0 runs without H8");
  }
  void h8_input_check(uint64_t calls_at_start, const char* name) const {
    if (h8_act_calls_ == calls_at_start)
      throw std::logic_error(std::string("gemm ") + name + ": H8 GEMM launched without its input prepared (h8_act not called)");
  }
  void h8_act(int Mp, int K, bool act32) {
    ++h8_act_calls_;
    if (h8_q2_pass_) return;  // (ARC_H8_Q2's second pass: both planes were made by the first)
    // (rmsnorm_h8 left this input rotated: from the f32 values, so per-32 scales or not make no difference)
    if (h8_pre_live_ && !h8_fsrc_ && h8_row() && Mp == h8_pre_M_ && K == h8_pre_K_) return;
    h8_pre_live_ = false;
    const cl_mem src = h8_fsrc_ ? h8_fsrc_ : aq_;
    if (h8_share_) {
      if (h8_last_K_ == K && h8_last_M_ == Mp && h8_last_a32_ == (act32 ? 1 : 0) && h8_last_aq_ == src) return;
      h8_last_K_ = K, h8_last_M_ = Mp, h8_last_a32_ = act32 ? 1 : 0, h8_last_aq_ = src;
    }
    const int hb = h8_block(K);
    int a32 = act32 ? 1 : 0, M = Mp, KK = K;
    if (h8_row()) {
      // (gemm_xw8's packed order where it will read them, ARC_GEMM_XW_PACK: engine_xw.h xw_pack_a)
      cl_kernel k = h8r_kernel(hb, 0, "h8r_act", (h8_fsrc_ ? std::string(" -DH8F=0") : h8f_defs_) +
                                                     (xw_pack_a(Mp, K) ? " -DXW_PACKA_OUT" : "") + xw_wgla_defs(Mp, K) +
                                                     (h8_q2() ? " -DQ2" : ""));
      set_args(k, 0, src, asc_, a32, h8_aq_, h8_asc_, M, KK);
      if (h8_q2()) set_args(k, 7, h8_aq2_, h8_asc2_);  // (after the 7 above: aqh2, asch2)
      const size_t wg = 16 * (size_t)(K / hb);
      size_t gws[1] = {wg * Mp}, lws[1] = {wg};
      run("h8_act", k, 1, gws, lws);
      return;
    }
    if (h8_fsrc_) throw std::logic_error("h8_act: an f32 source without H8R");
    cl_kernel k = h8_kernel(hb, 0, "h8_act");
    set_args(k, 0, aq_, asc_, a32, h8_aq_, h8_asc_, M, KK);
    size_t gws[3] = {(size_t)hb / 4, (size_t)Mp, (size_t)(K / hb)}, lws[3] = {(size_t)hb / 4, 1, 1};
    run("h8_act", k, 3, gws, lws);
  }
  // the H8 form of a weight for this GEMM: resident ones converted once, staged ones after each decode (stage_owner_
  // stays the weight, so a layer's next chunks find it converted)
  // (packed: a staged weight in gemm_xw8's packed order, ARC_GEMM_XW_PACK; resident weights stay in the plain one)
  // (packed: 0 the plain order, 1 gemm_xw8's packed order, G > 1 that in ARC_GEMM_XW_WGL's blocks of G groups where
  //  the one-pass decode makes it (h8r_decode -DXW_WGLB), else 1. Returns the order made: 0, 1 or G)
  int h8_prepare(const GpuWeight& w, cl_mem& wq, cl_mem& hh, int packed = 0) {
    if (w.wq && h8_row() && !p140_res_) {  // (p140_res_: the resident copy stays unrotated for the short chunks)
      if (packed) throw std::logic_error("h8_prepare: a resident weight in the packed order");
      if (!h8_res_rot_.count(&w)) {  // (rotate this resident copy once; h8r_weights reuses column scales it finds)
        hh = h8r_weights(w, w.wq, w.wh);
        h8_res_rot_.insert(&w);
      } else {
        hh = h8_col_.at(&w);
      }
      wq = w.wq;
      return 0;
    }
    if (h8_row()) {
      wq = stage_q_[w.role];
      auto it = h8_col_.find(&w);
      // (the block order only from the one-pass decode: h8r_weights rotates in place, within each group's region)
      const bool one = it != h8_col_.end() && h8_dec_ok(w);
      const int order = packed > 1 && !one ? 1 : packed;
      if (stage_owner_[w.role] != &w || !stage_h8_[w.role] || stage_packed_[w.role] != order) {
        if (one) {
          h8r_decode(w, wq, it->second, order);  // (decode + rotation in one pass, bit-identical)
          hh = it->second;
        } else {
          decode_w8(w, wq, stage_h_[w.role]);
          hh = h8r_weights(w, wq, stage_h_[w.role], order != 0);
        }
        stage_owner_[w.role] = &w;
        stage_h8_[w.role] = true;
        stage_packed_[w.role] = order;
      } else {
        hh = h8_col_.at(&w);
      }
      return order;
    }
    if (w.wq) {
      auto it = h8_res_.find(&w);
      if (it == h8_res_.end()) {
        cl_mem h = buf((size_t)w.N * (w.K / 256) * 4);
        h8_weights(w, w.wq, w.wh, h);
        it = h8_res_.emplace(&w, h).first;
      }
      wq = w.wq;
      hh = it->second;
      return 0;
    }
    const size_t need = (size_t)w.N * (w.K / 256) * 4;
    if (stage_hh_bytes_[w.role] < need) {  // (a matrix made after alloc_buffers: the encoders', MTP)
      stage_hh_[w.role] = buf(stage_hh_bytes_[w.role] = need);
    }
    wq = stage_q_[w.role];
    hh = stage_hh_[w.role];
    if (stage_owner_[w.role] != &w || !stage_h8_[w.role]) {
      decode_w8(w, wq, stage_h_[w.role]);
      h8_weights(w, wq, stage_h_[w.role], hh);
      stage_owner_[w.role] = &w;
      stage_h8_[w.role] = true;
    }
    return 0;
  }
  // the split-K pieces of an H8 GEMM: 17 or 34 superblocks each for rows longer than 8192 (ARC_GEMM_SPLIT), one with the
  // fused SwiGLU (silu(gate) * (u1 + u2) is not silu(gate) * u1 + u2). Each piece is rounded on its own before it is
  // added to C, so the pieces decide the result's bits: gemm_xw (zero/xe2/engine_xw.h) takes them from here too.
  // Discrete cards: 34 superblocks since 2026-11-15 (2 pieces of the 27B's FFN down, K = 17,408, not 4): a Pro B70's
  // 4K prompt +3.1-3.2 % in three interleaved rounds (2,244 -> 2,315 tok/s, gemm_down 253 -> 214 ms), KL against the
  // plain path 0.00145 -> 0.00152 (max 0.039 -> 0.052), top token 97.55 -> 98.14 % (joshdev-de, issue #27). iGPUs: 17 as
  // before (the same bits). ARC_GEMM_SPLIT=N sets it, 0: one piece. (Only the variable is cached: iGPU or card is
  //  asked per call, so each GPU of a split model gets its own default; the value is clamped so the piece count
  //  cannot overflow.)
  // (superblocks a piece: 17 on iGPUs and in the Arc 140T preset, 34 elsewhere. The preset's discrete cards, the
  //  Alchemist A-series, took 17 since 2026-11-18: an A770's 27B 4K prompt 1,150 -> 1,223 tok/s, gemm_down 823 -> 623
  //  ms (TheUniKian, issue #36, release .08; 0, one piece: 1,155); before, the B-series' 34)
  int h8_splits(int K, int a) const {
    static const int env = getenv("ARC_GEMM_SPLIT") ? std::clamp(atoi(getenv("ARC_GEMM_SPLIT")), 0, 4096) : -1;
    const int sk_sb = env >= 0 ? env : (zrt::integrated() || p140_ ? 17 : 34);
    const int nsb = K / 256;
    return a != 2 && K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1;
  }
  // the GEMM (matmul1's prompt path) in H8: the same grid, split-K and epilogues as the W8 GEMM. With ARC_H8_Q2 a
  // second pass on the residual plane, added (a = 1): the planes and the scales swapped in, the same kernels, pieces
  // and weights; the fused SwiGLU (a = 2) is not linear, so it is never asked for then (engine_qwen.h layer_ffn)
  void h8_matmul(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int a, bool act32) {
    h8_matmul1(name, w, Mp, C, ldc, c_off, a, act32);
    if (!h8_q2()) return;
    if (a == 2) throw std::logic_error(std::string("gemm ") + name + ": ARC_H8_Q2 with the fused SwiGLU epilogue");
    struct Swap {
      Engine* e;
      explicit Swap(Engine* en) : e(en) { flip(); e->h8_q2_pass_ = true; }
      ~Swap() { flip(); e->h8_q2_pass_ = false; }
      void flip() { std::swap(e->h8_aq_, e->h8_aq2_); std::swap(e->h8_asc_, e->h8_asc2_); }
    } sw(this);
    h8_matmul1(name, w, Mp, C, ldc, c_off, 1, act32);
  }
  void h8_matmul1(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int a, bool act32) {
    const uint64_t act0 = h8_act_calls_;  // (h8_input_check below)
    XwRole xr(this, w.role);  // (the big tile's work-groups and input order by role, ARC_GEMM_XW_SLM_WG_GU: engine_xw.h)
    // (H8R: a split's whole dot product is one int32 sum, |sum| <= 127 * 127 * K: past 133,000 values per split it
    //  could overflow. The models' K is 17-22K today and split-K cuts it further; a refusal, not a wrong answer)
    if (h8_row()) {
      const int sp = h8_splits(w.K, a);
      if ((w.K + sp - 1) / sp > 133000)
        throw std::runtime_error(std::string("gemm ") + name + ": H8 with K = " + std::to_string(w.K) + " in " + std::to_string(sp) +
                                 " pieces: over 133,000 values a piece, an int32 sum could overflow (ARC_GEMM_SPLIT: more pieces)");
    }
    cl_mem wq = nullptr, hh = nullptr;
    // (gemm_xw8's packed operand order where it takes this GEMM, ARC_GEMM_XW_PACK: engine_xw.h)
    const bool pw = xw_pack_w(w, Mp);
    // (ARC_GEMM_XW_WGL: the weight in the big tile's block order where it takes the GEMM, pack bit 8; the input in
    //  it, bit 4, is the same for every GEMM of this M and K, so a GEMM the big tile does not take gets a copy in the
    //  packed order, xw_unwgl)
    const int wb = pw ? xw_wgl_b(w, Mp) : 0;
    const int order = h8_prepare(w, wq, hh, wb > 1 ? wb : pw ? 1 : 0);
    int pack = (xw_pack_a(Mp, w.K) ? 1 : 0) | (pw ? 2 : 0) | (order > 1 ? 8 : 0);
    if ((pack & 8) && xw_bg_on()) pack |= 16;  // (ARC_GEMM_XW_BG: B from global memory, engine_xw.h)
    h8_act(Mp, w.K, act32);
    h8_input_check(act0, name);
    XwUnwgl un(this, xw_wgl_a(Mp, w.K) > 1 && wb <= 1 ? Mp : 0, w.K);
    if (xw_wgl_a(Mp, w.K) > 1 && wb > 1) pack |= 4;
    // (the 140T preset: 32 tokens, 2 column groups, 128 registers on XMX)
    const bool pt = p140_take(w, Mp) && !h8_env();
    // (xw_: wider XMX tiles, the same results, by default in the 140T preset and with h8_auto_ (discrete cards), elsewhere ARC_GEMM_XW=1; zero/xe2/engine_xw.h. On 8-lane GPUs the 140T preset's
    //  chunks (pt) too: gemm_xw8 instead of gemm_x8. A shape it does not take: below)
    if (xw_ && (cl_.sg8 || !pt) && xw_matmul(name, wq, hh, Mp, w.N, w.K, C, ldc, c_off, a, pack)) return;
    if (pack) throw std::logic_error(std::string("gemm ") + name + ": packed operands for a shape gemm_xw8 did not take");
    const int cg = pt ? 2 : h8_cg_for(w.N);
    cl_kernel k = pt ? gemm_kernel(h8_qtype(), false, 1, 2, 32, 1) : gemm_kernel(h8_qtype(), false, -1, cg);
    int M = Mp, N = w.N, K = w.K;
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    const int tm = pt ? 32 : gemm_tm();
    if (Mp % tm || N % 16 || K % 256)
      throw std::runtime_error(std::string("gemm ") + name + ": shape not a multiple of " + std::to_string(tm) + " x 16 x 256");
    const int nsb = K / 256, splits = h8_splits(K, a), rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16 * (size_t)cg, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1;
      set_args(k, 0, wq, hh, h8_aq_, h8_asc_, absum_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
  }
