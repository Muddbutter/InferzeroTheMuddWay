// DSpark drafting for Inkling 2 (I2 step 6, docs/inkling2.md "DSpark drafting: design"). A part of class Engine,
// included in its body after zero/engine_inkling.h, like that file: not a header of its own.
//
// The drafter: unsloth's dspark-Inkling-2-Mini-*.gguf (arch dflash, the model's own 3-layer block drafter). Its graph
// follows ollama's mlxrunner/model/inkling2/draft.go and model.go (MIT, NOTICE), whose tensor names map one for one
// onto the file's (docs/inkling2.md's table):
// - the target's taps: the stream (ink_cur_) as listed layer visits start (dflash.target_layers, counting the 48
//   visits of the loop), and the head's input (the visit count itself), 4 rows of n_embd per token;
// - per committed token, per drafter layer i: the taps' mix by the softmaxed fuse row i, RMS-normed (dflare_norm), its
//   K (attn_k_ctx, the head norm attn_k_ctx_norm, rope) and V (attn_v_ctx) appended to that layer's cache; the mix by
//   the last fuse row (dspark_norm) of the latest token is the head's feature row;
// - a draft: a block of 8 rows (the anchor, embedded as the target embeds it, then 7 mask_embd) at positions pos ..
//   pos + 7, 3 layers (sliding, global, sliding), each row attending to the context's keys and all 8 of the block's
//   (not causal), then output_norm;
// - the Markov head, greedy over the 7 mask rows: z = [state, markov_w1[last], h_i, f], g = joint_proj z + b,
//   keep / update / latent = sigmoid / tanh / tanh, state = keep state + (1 - keep) update, the row's logits
//   lm_head (h_i + hidden_out latent + b_h), the draft its argmax.
// The engine's form (the design's two costs): the 7 rows' base logits lm_head h_i in one head pass, then per row only
// its top-K (ARC_DSP_TOPK, 64) candidates get the correction lm_head[c] (hidden_out latent + b_h), from the head's
// rows in the model file; ARC_DSP_TOPK=0 the exact chain, a head pass per draft (to measure what the top-K costs in
// acceptance). Drafts never touch the target's numbers: verification keeps the greedy text as without drafting.
// Verification is an 8-row chunk of ink_forward; Inkling's short-conv states (4 per visit) are kept as they were and
// their inputs saved, then ink_conv_commit moves them on by the accepted rows only.
// Drafter weights: as the file has them (Q8_0), the head's small matrices f32: no KL rule applies to drafts.

 public:
  bool dsp_on() const { return dsp_; }
  // (runz: the draft steps' acceptance, per step and over the run)
  uint64_t dsp_drafted_ = 0, dsp_accepted_ = 0, dsp_steps_ = 0;
  // (the counts to zero: after Engine::warm_up_once, whose drafts (a marker missing: the first start after an update or
  //  another ARC_* switch) were counted in the run's, ARC_DSP_CMP's 145 / 146 against 138 / 139 without them)
  void dsp_stats_reset() {
    dsp_drafted_ = dsp_accepted_ = dsp_steps_ = 0;
    dsp_checks_ = dsp_check_bad_ = dsp_cmp_n_ = dsp_cmp_same_ = 0;
  }
 private:
  struct DspLayer {
    bool local = false;
    GpuWeight q, k, v, o, kx, vx, gate, up, down;
    cl_mem attn_norm = nullptr, ffn_norm = nullptr, ctx_norm = nullptr, q_norm = nullptr, k_norm = nullptr,
           kx_norm = nullptr, kc = nullptr, vc = nullptr;
    int cap = 0;
  };
  std::unique_ptr<GgufFile> dsp_f_;
  bool dsp_ = false;
  std::vector<DspLayer> dsp_l_;
  std::vector<int> dsp_taps_;
  int dsp_H_ = 0, dsp_HKV_ = 0, dsp_F_ = 0, dsp_R_ = 0, dsp_B_ = 8, dsp_win_ = 0, dsp_nvis_ = 0;
  float dsp_eps_ = 1e-6f, dsp_base_ = 1e6f;
  std::vector<cl_mem> dsp_fuse_;      // (layers + 1 rows of the softmaxed fuse weights, one buffer each)
  cl_mem dsp_mask_ = nullptr, dsp_onorm_ = nullptr, dsp_fnorm_ = nullptr;
  cl_mem dsp_joint_ = nullptr, dsp_joint_b_ = nullptr, dsp_hout_ = nullptr, dsp_hout_b_ = nullptr;  // (f32)
  const GgufTensor* dsp_w1_ = nullptr;    // (markov_w1 in the drafter's mapping: R values a token)
  const GgufTensor* dsp_head_t_ = nullptr;  // (the target's head in its mapping: the candidates' rows)
  cl_mem dsp_tap_ = nullptr;              // [taps][max_chunk][E]
  std::vector<cl_mem> dsp_tapsub_;        // (each tap's region)
  cl_mem dsp_x_ = nullptr, dsp_xn_ = nullptr, dsp_q_ = nullptr, dsp_kb_ = nullptr, dsp_vb_ = nullptr, dsp_att_ = nullptr,
         dsp_t_ = nullptr, dsp_gu_ = nullptr, dsp_h_ = nullptr;  // (the block's rows)
  std::vector<cl_mem> dsp_hrow_;          // (rows of dsp_h_)
  cl_mem dsp_hmask_ = nullptr;            // (its rows 1 .. B - 1)
  cl_mem dsp_ctx_ = nullptr, dsp_kk_ = nullptr, dsp_vv_ = nullptr;  // (a commit's rows, 8 at a time)
  cl_mem dsp_feat_ = nullptr, dsp_state_ = nullptr, dsp_z_ = nullptr, dsp_g_ = nullptr, dsp_lat_ = nullptr,
         dsp_u_ = nullptr, dsp_w1row_ = nullptr;
  cl_mem dsp_save_arena_ = nullptr, dsp_tab_ = nullptr, dsp_logits_ = nullptr;  // (dsp_logits_: the 7 rows' head)
  int dsp_ntab_ = 0, dsp_maxc_ = 0;
  int dsp_cn_ = 0;  // (the context positions the drafter's caches hold: 0 .. dsp_cn_ - 1)
  // (the chain on the GPU: F = lm_head hidden_out [V][R] f16, fb = lm_head b_h [V], markov_w1 [V][R] f16; the top-K
  //  candidates and values of the rows, the drafts' ids)
  cl_mem dsp_Fh_ = nullptr, dsp_fb_ = nullptr, dsp_w1g_ = nullptr, dsp_cidx_ = nullptr, dsp_cval_ = nullptr, dsp_ids_ = nullptr;
  std::vector<float> dsp_lg_, dsp_hrow_host_, dsp_u_host_, dsp_w1_host_;
  std::vector<int> dsp_idx_;

  static int dsp_drafts_default() {
    static const int d = getenv("ARC_DSP_DRAFTS") ? std::clamp(atoi(getenv("ARC_DSP_DRAFTS")), 1, 7) : 3;
    return d;
  }
  int dsp_topk() const {
    static const int k = getenv("ARC_DSP_TOPK") ? std::max(0, atoi(getenv("ARC_DSP_TOPK"))) : 64;
    return k;
  }
  int ink_visits() const {
    const InkHp& k = ink_;
    return k.start + (k.start < k.end ? k.passes * (k.end - k.start) : 0) + (hp.n_layer - k.end);
  }

  void load_dspark(const std::string& path) {
    dsp_f_ = std::make_unique<GgufFile>(path);
    const GgufFile& d = *dsp_f_;
    const std::string a = d.get_str("general.architecture");
    if (a != "dflash" || !d.find("markov_w1.weight") || !d.find("dspark_fuse.weight") || !d.find("joint_proj.weight"))
      throw std::runtime_error("DSpark: " + path + " is not a DSpark drafter (arch dflash with markov_w1, dspark_fuse, joint_proj)");
    auto I = [&](const std::string& k, int64_t def = -1) { return (int)d.get_int("dflash." + k, def); };
    const int nl = I("block_count"), E = hp.n_embd;
    dsp_H_ = I("attention.head_count");
    dsp_HKV_ = I("attention.head_count_kv");
    dsp_F_ = I("feed_forward_length");
    dsp_B_ = I("block_size", 8);
    dsp_win_ = I("attention.sliding_window", 0);
    dsp_eps_ = (float)d.get_float("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    dsp_base_ = (float)d.get_float("dflash.rope.freq_base", 1e6);
    dsp_nvis_ = ink_visits();
    const auto* tl = d.get_ints("dflash.target_layers");
    const auto* pat = d.get_ints("dflash.attention.sliding_window_pattern");
    const GgufTensor& w1 = d.get("markov_w1.weight");
    dsp_R_ = (int)w1.dims[0];
    const GgufTensor& fz = d.get("dspark_fuse.weight");
    const GgufTensor& jp = d.get("joint_proj.weight");
    const GgufTensor& ho = d.get("dspark_hidden_out.weight");
    bool ok = nl >= 1 && I("embedding_length") == E && I("attention.key_length") == hp.head_dim && dsp_H_ > 0 &&
              dsp_HKV_ > 0 && dsp_H_ % dsp_HKV_ == 0 && dsp_F_ % 256 == 0 && dsp_B_ >= 2 && dsp_B_ <= 8 && tl &&
              !tl->empty() && tl->size() <= 8 && pat && (int)pat->size() >= nl && (int)w1.dims[1] == hp.n_vocab &&
              (int)fz.dims[0] == (int)tl->size() && (int)fz.dims[1] == nl + 1 && (int)jp.dims[0] == 2 * (dsp_R_ + E) &&
              (int)jp.dims[1] == 3 * dsp_R_ && (int)ho.dims[0] == dsp_R_ && (int)ho.dims[1] == E && E % 256 == 0;
    for (size_t i = 0; ok && tl && i < tl->size(); ++i)
      ok = (*tl)[i] >= 0 && (*tl)[i] <= dsp_nvis_ && (i == 0 || (*tl)[i] > (*tl)[i - 1]);
    if (ok) ok = tl->back() == dsp_nvis_;  // (the last tap: the head's input, as ollama requires)
    if (!ok)
      throw std::runtime_error("DSpark: " + path + " does not match this model (its width, vocabulary, taps over " +
                               std::to_string(dsp_nvis_) + " layer visits, or a shape)");
    for (int64_t t : *tl) dsp_taps_.push_back((int)t);
    {  // (the fuse rows softmaxed once, as ollama does at load)
      std::vector<float> f(fz.nelements());
      dequant_row(fz.type, fz.data, (int)f.size(), f.data());
      const int nt = (int)tl->size();
      for (int r = 0; r <= nl; ++r) {
        float* row = &f[(size_t)r * nt];
        const float mx = *std::max_element(row, row + nt);
        float z = 0.0f;
        for (int t = 0; t < nt; ++t) z += row[t] = std::exp(row[t] - mx);
        for (int t = 0; t < nt; ++t) row[t] /= z;
        dsp_fuse_.push_back(buf((size_t)nt * 4, row));
      }
    }
    dsp_mask_ = f32_from(d, "mask_embd.weight");
    dsp_onorm_ = f32_from(d, "output_norm.weight");
    dsp_fnorm_ = f32_from(d, "dspark_norm.weight");
    dsp_joint_ = f32_from(d, "joint_proj.weight");
    dsp_joint_b_ = f32_from(d, "joint_proj.bias");
    dsp_hout_ = f32_from(d, "dspark_hidden_out.weight");
    dsp_hout_b_ = f32_from(d, "dspark_hidden_out.bias");
    dsp_w1_ = &w1;
    dsp_head_t_ = f_.find("output.weight") ? &f_.get("output.weight") : &f_.get("token_embd.weight");
    dsp_l_.resize(nl);
    for (int il = 0; il < nl; ++il) {
      DspLayer& L = dsp_l_[il];
      const std::string p = "blk." + std::to_string(il) + ".";
      L.local = (*pat)[il] != 0 && dsp_win_ > 0;
      L.q = gemv_weight(d, p + "attn_q.weight");
      L.k = gemv_weight(d, p + "attn_k.weight");
      L.v = gemv_weight(d, p + "attn_v.weight");
      L.o = gemv_weight(d, p + "attn_output.weight");
      L.kx = gemv_weight(d, p + "attn_k_ctx.weight");
      L.vx = gemv_weight(d, p + "attn_v_ctx.weight");
      L.gate = gemv_weight(d, p + "ffn_gate.weight");
      L.up = gemv_weight(d, p + "ffn_up.weight");
      L.down = gemv_weight(d, p + "ffn_down.weight");
      L.attn_norm = f32_from(d, p + "attn_norm.weight");
      L.ffn_norm = f32_from(d, p + "ffn_norm.weight");
      L.ctx_norm = f32_from(d, p + "dflare_norm.weight");
      L.q_norm = f32_from(d, p + "attn_q_norm.weight");
      L.k_norm = f32_from(d, p + "attn_k_norm.weight");
      L.kx_norm = f32_from(d, p + "attn_k_ctx_norm.weight");
      const int QW = dsp_H_ * hp.head_dim, KG = dsp_HKV_ * hp.head_dim;
      if (L.q.N != QW || L.k.N != KG || L.v.N != KG || L.kx.N != KG || L.vx.N != KG || L.o.N != E || L.o.K != QW ||
          L.gate.N != dsp_F_ || L.up.N != dsp_F_ || L.down.N != E || L.down.K != dsp_F_)
        throw std::runtime_error("DSpark: " + path + ": unexpected layer shape in " + p);
    }
    has_mtp = dsp_ = true;
    clFinish(cl_.queue);
    LOG_INF("DSpark drafter: %d layers, block %d, Markov rank %d, taps %s of %d visits, %d drafts a step, top-%d "
            "candidates (%s)\n", nl, dsp_B_, dsp_R_,
            [&] { std::string s; for (int t : dsp_taps_) s += (s.empty() ? "" : ",") + std::to_string(t); return s; }().c_str(),
            dsp_nvis_, opt.mtp_drafts, dsp_topk(), path.c_str());
  }

  // the chunk's and the block's buffers, the drafter's caches, the conv inputs' save arena (alloc_inkling)
  void dsp_alloc() {
    const int E = hp.n_embd, D = hp.head_dim, QW = dsp_H_ * D, KG = dsp_HKV_ * D, B = dsp_B_, nt = (int)dsp_taps_.size();
    const size_t tapB = (size_t)max_chunk * E * 4;
    dsp_tap_ = buf(tapB * nt);
    for (int t = 0; t < nt; ++t) dsp_tapsub_.push_back(ink_sub(dsp_tap_, t * tapB, tapB));
    dsp_x_ = buf((size_t)B * E * 4);
    dsp_xn_ = buf((size_t)B * E * 4);
    dsp_q_ = buf((size_t)B * QW * 4);
    dsp_kb_ = buf((size_t)B * KG * 4);
    dsp_vb_ = buf((size_t)B * KG * 4);
    dsp_att_ = buf((size_t)B * QW * 4);
    dsp_t_ = buf((size_t)B * E * 4);
    dsp_gu_ = buf((size_t)B * 2 * dsp_F_ * 4);
    dsp_h_ = buf((size_t)B * E * 4);
    for (int r = 0; r < B; ++r) dsp_hrow_.push_back(ink_sub(dsp_h_, (size_t)r * E * 4, (size_t)E * 4));
    dsp_hmask_ = ink_sub(dsp_h_, (size_t)E * 4, (size_t)(B - 1) * E * 4);  // (the 7 mask rows, for the head pass)
    dsp_ctx_ = buf((size_t)8 * E * 4);
    dsp_kk_ = buf((size_t)8 * KG * 4);
    dsp_vv_ = buf((size_t)8 * KG * 4);
    dsp_feat_ = buf((size_t)E * 4);
    dsp_state_ = buf((size_t)dsp_R_ * 4);
    dsp_z_ = buf((size_t)2 * (dsp_R_ + E) * 4);
    dsp_g_ = buf((size_t)3 * dsp_R_ * 4);
    dsp_lat_ = buf((size_t)dsp_R_ * 4);
    dsp_u_ = buf((size_t)E * 4);
    dsp_w1row_ = buf((size_t)dsp_R_ * 4);
    for (DspLayer& L : dsp_l_) {
      // (a sliding layer's ring holds the window and a chunk more, as the target's sliding slots (swa_ring_cap): a
      //  snapshot's window is still there after a chunk past it, Engine::snapshot)
      L.cap = L.local ? std::min(ctx, (dsp_win_ + max_chunk + 255) / 256 * 256) : ctx;
      L.kc = buf((size_t)L.cap * KG * 2);
      L.vc = buf((size_t)L.cap * KG * 2);
    }
    // (the save arena: per slot and conv, B rows of its channels; the table for ink_conv_commit, in floats)
    const InkHp& k = ink_;
    const int tkg = hp.n_head_kv * D, ch[4] = {tkg, tkg, E, E};  // (the target's conv channels: K, V, attn out, ffn out)
    std::vector<int> tab;
    size_t off = 0;
    std::vector<size_t> soff((size_t)k.nslot * 4);
    for (int s = 0; s < k.nslot; ++s)
      for (int c = 0; c < 4; ++c) {
        soff[(size_t)s * 4 + c] = off;
        tab.push_back((int)(ink_conv_off_[(size_t)s * 4 + c] / 4));
        tab.push_back((int)(off / 4));
        tab.push_back(ch[c]);
        dsp_maxc_ = std::max(dsp_maxc_, ch[c]);
        off += ((size_t)B * ch[c] * 4 + 255) / 256 * 256;
      }
    dsp_save_arena_ = buf(off);
    for (int s = 0; s < k.nslot; ++s)
      for (int c = 0; c < 4; ++c)
        inks_[s].save[c] = ink_sub(dsp_save_arena_, soff[(size_t)s * 4 + c], (size_t)B * ch[c] * 4);
    dsp_ntab_ = (int)tab.size() / 3;
    dsp_tab_ = buf(tab.size() * 4, tab.data());
    dsp_lg_.resize((size_t)(B - 1) * n_logits());
    dsp_logits_ = buf((size_t)8 * n_logits() * 4);  // (7 rows of the block's head; 8 for the fold's passes)
    dsp_hrow_host_.resize(E);
    dsp_u_host_.resize(E);
    dsp_w1_host_.resize(dsp_R_);
    dsp_idx_.resize(std::max(1, dsp_topk()));
    if (dsp_topk() > 0 && dsp_topk() <= 64 && !opt.dry) dsp_fold();
  }
  // (the chain on the GPU, 2026-11-24) F = lm_head hidden_out, a row of R per token (f16), as ollama folds it at load,
  // through the head itself: hidden_out's R columns as 8 "token" rows a pass (R / 8 head passes, the head's softcap left
  // out: a copy of the head, so not lm_head_); fb = lm_head b_h; markov_w1 to f16. With them a candidate's correction
  // is R multiply-adds. ~2 V R bytes + 2 V R bytes (Inkling: ~206 MB, with --dspark only)
  void dsp_fold() {
    const int E = hp.n_embd, R = dsp_R_, V = n_logits(), saved = cur_T_;
    dsp_Fh_ = buf((size_t)V * R * 2);
    dsp_fb_ = buf((size_t)V * 4);
    dsp_w1g_ = buf((size_t)V * R * 2);
    dsp_cidx_ = buf((size_t)8 * 64 * 4);
    dsp_cval_ = buf((size_t)8 * 64 * 4);
    dsp_ids_ = buf(8 * 4);
    GpuWeight head = lm_head_;  // (not &lm_head_: head_defs adds no softcap)
    std::vector<float> wt((size_t)R * E);  // (hidden_out's columns as rows: [R][E]; the tensor is [E][R], R fastest)
    {
      const GgufTensor& t = dsp_f_->get("dspark_hidden_out.weight");
      std::vector<float> w(t.nelements());
      dequant_row(t.type, t.data, (int)w.size(), w.data());
      for (int e = 0; e < E; ++e)
        for (int r = 0; r < R; ++r) wt[(size_t)r * E + e] = w[(size_t)e * R + r];
    }
    cl_mem rows = buf((size_t)8 * E * 4);
    cl_kernel kf = ink_k("dsp_fold");
    for (int r0 = 0; r0 < R; r0 += 8) {
      const int m = std::min(8, R - r0);
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, rows, CL_TRUE, 0, (size_t)m * E * 4, &wt[(size_t)r0 * E], 0, nullptr, nullptr));
      cur_T_ = m;
      quant_rows(rows, E, 16, m, 1);
      matmul("dsp_fold_head", head, 16, dsp_logits_, V, 0, 0, true);
      int nl = V, rr = r0, mm = m, RR = R, VV = V;
      set_args(kf, 0, dsp_logits_, nl, rr, mm, dsp_Fh_, RR, VV);
      ink_run1("dsp_fold", kf, (size_t)V * m);
    }
    cur_T_ = 1;
    quant_rows(dsp_hout_b_, E, 16, 1, 1);
    matmul("dsp_fold_bias", head, 16, dsp_fb_, V, 0, 0, true);
    cur_T_ = saved;
    free_buf(rows);
    {  // markov_w1 to f16, in row blocks
      const size_t rb = dsp_w1_->nbytes / hp.n_vocab;
      std::vector<float> f((size_t)R);
      std::vector<uint16_t> h((size_t)4096 * R);
      for (int c0 = 0; c0 < hp.n_vocab; c0 += 4096) {
        const int n = std::min(4096, hp.n_vocab - c0);
        for (int c = 0; c < n; ++c) {
          dequant_row(dsp_w1_->type, dsp_w1_->data + (size_t)(c0 + c) * rb, R, f.data());
          for (int r = 0; r < R; ++r) h[(size_t)c * R + r] = f32_to_fp16(f[r]);
        }
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, dsp_w1g_, CL_TRUE, (size_t)c0 * R * 2, (size_t)n * R * 2, h.data(), 0, nullptr, nullptr));
      }
    }
    clFinish(cl_.queue);
  }

  // (ink_conv's save buffer: during a verification pass)
  cl_mem dsp_save(const InkSlot& S, int c) const { return dsp_ && spec_ ? S.save[c] : nullptr; }
  // the stream as visit v starts, a tap's rows
  void dsp_tap(int v, int T) {
    for (size_t j = 0; j < dsp_taps_.size(); ++j)
      if (dsp_taps_[j] == v)
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, ink_cur_, dsp_tapsub_[j], 0, 0, (size_t)T * hp.n_embd * 4, 0, nullptr, nullptr));
  }
  // after ink_forward: the last tap, the head's input (output_norm of x_); outside verification the chunk's rows join
  // the drafter's context (verification commits its accepted rows in dsp_verify_done)
  void dsp_after_forward(int T) {
    if (dsp_taps_.back() == dsp_nvis_) ink_rmsnorm(x_, hp.n_embd, out_norm_, dsp_tapsub_.back(), T);
    if (!spec_) dsp_commit(0, T, pos);
  }
  // rows r0 .. r0 + n - 1 of the taps (positions p0 ..) into the drafter's caches, 8 at a time, and the feature row of
  // the last one
  void dsp_commit(int r0, int n, int p0) {
    const int E = hp.n_embd, D = hp.head_dim, KG = dsp_HKV_ * D, nt = (int)dsp_taps_.size(), saved = cur_T_;
    int64_t ts = (int64_t)max_chunk * E;
    float eps = dsp_eps_, base = dsp_base_;
    cl_kernel km = ink_k("dsp_mix"), ks = ink_k("dsp_kv_store");
    for (int g = 0; g < n; g += 8) {
      const int m = std::min(8, n - g);
      cur_T_ = m;
      for (size_t il = 0; il < dsp_l_.size(); ++il) {
        DspLayer& L = dsp_l_[il];
        int NT = nt, R0 = r0 + g, EE = E;
        set_args(km, 0, dsp_tap_, NT, ts, R0, dsp_fuse_[il], L.ctx_norm, dsp_ctx_, EE, eps);
        ink_rung("dsp_mix", km, m, 256);
        quant_rows(dsp_ctx_, E, 16, m, 1);
        matmul("dsp_kx", L.kx, 16, dsp_kk_, KG, 0, 0, true);
        matmul("dsp_vx", L.vx, 16, dsp_vv_, KG, 0, 0, true);
        int nkv = dsp_HKV_, P0 = p0 + g, cap = L.cap;
        set_args(ks, 0, dsp_kk_, dsp_vv_, nkv, L.kx_norm, P0, eps, base, L.kc, L.vc, cap);
        ink_rung("dsp_kv_store", ks, (size_t)m * dsp_HKV_, D);
      }
    }
    cur_T_ = saved;
    int NT = nt, R0 = r0 + n - 1, EE = E;
    set_args(km, 0, dsp_tap_, NT, ts, R0, dsp_fuse_.back(), dsp_fnorm_, dsp_feat_, EE, eps);
    ink_rung("dsp_mix", km, 1, 256);
    dsp_cn_ = p0 + n;
  }
  // y[N] = W x[K], W f32 (ink_mm_f32_sg: one sub-group an output)
  void dsp_mv(const char* name, cl_mem W, cl_mem x, cl_mem y, int N, int K) {
    cl_kernel k = ink_k("ink_mm_f32_sg");
    int NN = N, KK = K, T = 1;
    set_args(k, 0, W, x, y, NN, KK, T);
    size_t g = ((size_t)N * 16 + 63) / 64 * 64, l = 64;
    run(name, k, 1, &g, &l);
  }

  // one draft step: the block, the base logits of its 7 mask rows, then the Markov chain (greedy)
  std::vector<int> dsp_draft(int tok, int max_drafts) {
    const int E = hp.n_embd, D = hp.head_dim, B = dsp_B_, QW = dsp_H_ * D, KG = dsp_HKV_ * D, F = dsp_F_, R = dsp_R_;
    const int saved = cur_T_, nl = n_logits(), nv = ink_.nv, nd = B - 1;
    max_drafts = std::clamp(max_drafts, 1, std::min(nd, std::max(1, ctx - pos - 1)));
    float eps = dsp_eps_, base = dsp_base_;
    {  // the anchor as the target embeds it (one row from the mapped file), then the masks
      const size_t rb = tok_embd_->nbytes / hp.n_vocab;
      dequant_row(tok_embd_->type, tok_embd_->data + (size_t)tok * rb, E, dsp_hrow_host_.data());
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, dsp_x_, CL_TRUE, 0, (size_t)E * 4, dsp_hrow_host_.data(), 0, nullptr, nullptr));
      ink_rmsnorm(dsp_x_, E, ink_tok_norm_, dsp_x_, 1);
      cl_kernel k = ink_k("dsp_block_init");
      int EE = E, BB = B;
      set_args(k, 0, dsp_x_, dsp_mask_, EE, BB);
      ink_run1("dsp_block_init", k, (size_t)E * (B - 1));
    }
    cur_T_ = B;
    cl_kernel kq = ink_k("dsp_qk"), ka = ink_k("dsp_attn");
    for (DspLayer& L : dsp_l_) {
      ink_rmsnorm(dsp_x_, E, L.attn_norm, dsp_xn_, B);
      quant_rows(dsp_xn_, E, 16, B, 1);
      matmul("dsp_q", L.q, 16, dsp_q_, QW, 0, 0, true);
      matmul("dsp_k", L.k, 16, dsp_kb_, KG, 0, 0, true);
      matmul("dsp_v", L.v, 16, dsp_vb_, KG, 0, 0, true);
      int nh = dsp_H_, nkv = dsp_HKV_, p0 = pos;
      set_args(kq, 0, dsp_q_, nh, L.q_norm, p0, eps, base);
      ink_rung("dsp_qk", kq, (size_t)B * dsp_H_, D);
      set_args(kq, 0, dsp_kb_, nkv, L.k_norm, p0, eps, base);
      ink_rung("dsp_qk", kq, (size_t)B * dsp_HKV_, D);
      int cap = L.cap, cn = dsp_cn_, clo = L.local ? std::max(0, dsp_cn_ - dsp_win_) : 0, HH = dsp_H_, HK = dsp_HKV_, BB = B;
      set_args(ka, 0, dsp_q_, dsp_kb_, dsp_vb_, L.kc, L.vc, cap, clo, cn, dsp_att_, HH, HK, BB);
      ink_rung("dsp_attn", ka, (size_t)B * dsp_H_, D);
      quant_rows(dsp_att_, QW, 16, B, 1);
      matmul("dsp_o", L.o, 16, dsp_t_, E, 0, 0, true);
      ink_add(dsp_x_, dsp_t_, dsp_x_, B * E);
      ink_rmsnorm(dsp_x_, E, L.ffn_norm, dsp_xn_, B);
      quant_rows(dsp_xn_, E, 16, B, 1);
      matmul("dsp_gate", L.gate, 16, dsp_gu_, 2 * F, 0, 0, true);
      matmul("dsp_up", L.up, 16, dsp_gu_, 2 * F, F, 0, true);
      {
        int FF = F, MM = 16, TT = B, a32 = 1, fused = 0, gelu = 0;
        set_args(k_swiglu_, 0, dsp_gu_, FF, MM, TT, aq_, asc_, absum_, a32, fused, gelu);
        run_rows_quant("dsp_swiglu", F, B, 16);
      }
      matmul("dsp_down", L.down, 16, dsp_t_, E, 0, 0, true);
      ink_add(dsp_x_, dsp_t_, dsp_x_, B * E);
    }
    ink_rmsnorm(dsp_x_, E, dsp_onorm_, dsp_h_, B);
    // the 7 mask rows' base logits, one head pass (capped by the head's epilogue: un-capped on the host)
    cur_T_ = nd;
    quant_rows(dsp_hmask_, E, 16, nd, 1);
    matmul("dsp_head", lm_head_, 16, dsp_logits_, nl, 0, 0, true);
    const float cap = hp.softcap;
    fill(dsp_state_, (size_t)R * 4);
    const int K = std::min(dsp_topk(), nv);
    if (dsp_Fh_ && K > 0 && K <= 64) {
      // (the chain on the GPU: the rows' top-K, then per row the gates and the candidates' argmax, which fetches the
      //  next row's markov_w1; one read of the drafts at the end)
      cl_kernel kt = ink_k("dsp_topk"), kc = ink_k("dsp_cat"), kg = ink_k("dsp_gate"), ks = ink_k("dsp_select"),
                kw = ink_k("dsp_w1_gather");
      int NL = nl, NV = nv, KK = K, RR = R, EE = E, tk = tok;
      float cp = cap;
      set_args(kt, 0, dsp_logits_, NL, NV, cp, KK, dsp_cidx_, dsp_cval_);
      ink_rung("dsp_topk", kt, max_drafts, 256);
      set_args(kw, 0, tk, dsp_w1g_, dsp_w1row_, RR);
      ink_run1("dsp_w1_gather", kw, R);
      for (int i = 0; i < max_drafts; ++i) {
        set_args(kc, 0, dsp_z_, dsp_state_, dsp_w1row_, dsp_hrow_[i + 1], dsp_feat_, RR, EE);
        ink_run1("dsp_cat", kc, (size_t)2 * (R + E));
        dsp_mv("dsp_joint", dsp_joint_, dsp_z_, dsp_g_, 3 * R, 2 * (R + E));
        set_args(kg, 0, dsp_g_, dsp_joint_b_, dsp_state_, dsp_lat_, RR);
        ink_run1("dsp_gate", kg, R);
        int row = i;
        set_args(ks, 0, dsp_cidx_, dsp_cval_, row, KK, dsp_Fh_, RR, dsp_fb_, dsp_lat_, dsp_ids_, dsp_w1g_, dsp_w1row_);
        ink_rung("dsp_select", ks, 1, 64);
      }
      std::vector<int> drafts((size_t)max_drafts);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, dsp_ids_, CL_TRUE, 0, (size_t)max_drafts * 4, drafts.data(), 0, nullptr, nullptr));
      static const bool cmp = getenv("ARC_DSP_CMP") && atoi(getenv("ARC_DSP_CMP")) != 0;
      if (!cmp) {
        cur_T_ = saved;
        dsp_dump(drafts, nullptr);
        dsp_oracle(drafts);
        collect_profile();
        return drafts;
      }
      dsp_cmp_gpu_ = drafts;  // (tests, ARC_DSP_CMP=1: the host's chain below as well, the drafts compared)
      fill(dsp_state_, (size_t)R * 4);
    }
    // (the host's chain: the exact one, ARC_DSP_TOPK=0, or more than 64 candidates)
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, dsp_logits_, CL_TRUE, 0, (size_t)nd * nl * 4, dsp_lg_.data(), 0, nullptr, nullptr));
    const size_t hrb = dsp_head_t_->nbytes / hp.n_vocab, w1b = dsp_w1_->nbytes / hp.n_vocab;
    cl_kernel kc = ink_k("dsp_cat"), kg = ink_k("dsp_gate"), kb = ink_k("dsp_addb");
    std::vector<int> drafts;
    int last = tok;
    for (int i = 0; i < max_drafts; ++i) {
      dequant_row(dsp_w1_->type, dsp_w1_->data + (size_t)last * w1b, R, dsp_w1_host_.data());
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, dsp_w1row_, CL_TRUE, 0, (size_t)R * 4, dsp_w1_host_.data(), 0, nullptr, nullptr));
      int RR = R, EE = E, exact = K == 0 ? 1 : 0;
      set_args(kc, 0, dsp_z_, dsp_state_, dsp_w1row_, dsp_hrow_[i + 1], dsp_feat_, RR, EE);
      ink_run1("dsp_cat", kc, (size_t)2 * (R + E));
      dsp_mv("dsp_joint", dsp_joint_, dsp_z_, dsp_g_, 3 * R, 2 * (R + E));
      set_args(kg, 0, dsp_g_, dsp_joint_b_, dsp_state_, dsp_lat_, RR);
      ink_run1("dsp_gate", kg, R);
      dsp_mv("dsp_hidden_out", dsp_hout_, dsp_lat_, dsp_u_, E, R);
      set_args(kb, 0, dsp_u_, dsp_hout_b_, dsp_hrow_[i + 1], exact, EE);
      ink_run1("dsp_addb", kb, E);
      int best = 0;
      if (K == 0) {  // (the exact chain: lm_head (h_i + hidden_out lat + b) over the whole vocabulary)
        cur_T_ = 1;
        quant_rows(dsp_u_, E, 16, 1, 1);
        matmul("dsp_head_exact", lm_head_, 16, dsp_logits_, nl, 0, 0, true);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, dsp_logits_, CL_TRUE, 0, (size_t)nv * 4, dsp_lg_.data() + (size_t)i * nl, 0,
                                     nullptr, nullptr));
        const float* l = &dsp_lg_[(size_t)i * nl];
        best = (int)(std::max_element(l, l + nv) - l);
      } else {
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, dsp_u_, CL_TRUE, 0, (size_t)E * 4, dsp_u_host_.data(), 0, nullptr, nullptr));
        const float* l = &dsp_lg_[(size_t)i * nl];
        int n = 0;  // (the K largest base logits, as DFlash2's selector finds them)
        for (int c = 0; c < nv; ++c) {
          const float v = l[c];
          if (n == K && v <= l[dsp_idx_[K - 1]]) continue;
          int j = n < K ? n++ : K - 1;
          while (j > 0 && l[dsp_idx_[j - 1]] < v) { dsp_idx_[j] = dsp_idx_[j - 1]; --j; }
          dsp_idx_[j] = c;
        }
        double bs = -1e300;
        for (int j = 0; j < n; ++j) {
          const int c = dsp_idx_[j];
          const float capped = l[c];
          const double raw = cap > 0.0f ? cap * std::atanh(std::clamp((double)capped / cap, -0.999999, 0.999999)) : capped;
          dequant_row(dsp_head_t_->type, dsp_head_t_->data + (size_t)c * hrb, E, dsp_hrow_host_.data());
          double corr = 0.0;
          for (int e = 0; e < E; ++e) corr += (double)dsp_hrow_host_[e] * dsp_u_host_[e];
          if (raw + corr > bs) bs = raw + corr, best = c;
        }
      }
      drafts.push_back(best);
      last = best;
    }
    cur_T_ = saved;
    if (!dsp_cmp_gpu_.empty()) {
      dsp_dump(dsp_cmp_gpu_, &drafts);
      // (the chain's drafts the same as the host's: row by row until the first that differs, as acceptance would)
      for (size_t i = 0; i < drafts.size() && i < dsp_cmp_gpu_.size(); ++i) {
        ++dsp_cmp_n_;
        if (drafts[i] != dsp_cmp_gpu_[i]) break;
        ++dsp_cmp_same_;
      }
      dsp_cmp_gpu_.clear();
    }
    dsp_oracle(drafts);
    collect_profile();
    return drafts;
  }
  // (tests, ARC_DSP_DUMP=FILE: each step's position and draft ids appended, the GPU chain's and, with ARC_DSP_CMP=1,
  //  the host's after a bar: runs compared line by line)
  void dsp_dump(const std::vector<int>& g, const std::vector<int>* h) {
    static const char* path = getenv("ARC_DSP_DUMP");
    if (!path) return;
    if (FILE* f = fopen(path, "ab")) {
      fprintf(f, "%d:", pos);
      for (int d : g) fprintf(f, " %d", d);
      if (h) {
        fprintf(f, " |");
        for (int d : *h) fprintf(f, " %d", d);
      }
      fprintf(f, "\n");
      fclose(f);
    }
  }
  // (tests, ARC_DSP_ORACLE=FILE: the drafts replaced by the plain run's tokens (FILE: the prompt's ids and the plain
  //  greedy text, position by position), the one at index steps % (drafts + 1) made wrong, so the accepted count cycles
  //  0, 1, ..., all: every rollback length, with the text still the plain one)
  void dsp_oracle(std::vector<int>& drafts) {
    static const char* path = getenv("ARC_DSP_ORACLE");
    if (!path) return;
    static std::vector<int> seq;
    if (seq.empty()) {
      std::ifstream fi(path);
      std::string s((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
      std::stringstream ss(s);
      std::string tkn;
      while (std::getline(ss, tkn, ',')) seq.push_back(std::stoi(tkn));
    }
    const int nd = (int)drafts.size(), bad = (int)(dsp_steps_ % (uint64_t)(nd + 1));
    for (int i = 0; i < nd; ++i) {
      const size_t p = (size_t)pos + 1 + i;
      const int t = p < seq.size() ? seq[p] : 0;
      drafts[i] = i == bad ? (t + 1) % ink_.nv : t;
    }
  }
  // (tests, ARC_DSP_CHECK=1: after each verification, the conv arena moved on by ink_conv_commit against the plain
  //  kernel's step (ink_conv_state over the same saved inputs, state by state) from a copy taken before, byte for byte)
  void dsp_check_commit(int rows) {
    const size_t bytes = ink_conv_off_.empty() ? 0 : ink_conv_off_.back() + (size_t)hp.n_embd * (ink_.nc - 1) * 4;
    std::vector<uint8_t> got(bytes), want(bytes);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, ink_conv_arena_, CL_TRUE, 0, bytes, got.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ink_conv_arena_, CL_TRUE, 0, bytes, dsp_check_before_.data(), 0, nullptr, nullptr));
    const int tkg = hp.n_head_kv * hp.head_dim, ch[4] = {tkg, tkg, hp.n_embd, hp.n_embd};
    cl_kernel ks = ink_k("ink_conv_state");
    for (auto& S : inks_)
      for (int c = 0; c < 4; ++c) {
        int C = ch[c], T = rows, NC = ink_.nc;
        set_args(ks, 0, S.save[c], S.conv[c], C, T, NC);
        ink_run1("ink_conv_state", ks, C);
      }
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, ink_conv_arena_, CL_TRUE, 0, bytes, want.data(), 0, nullptr, nullptr));
    ++dsp_checks_;
    if (got != want) {
      ++dsp_check_bad_;
      LOG_ERR("DSpark check: the conv states after %d rows DIFFER from ink_conv_state's (step %llu)\n", rows,
              (unsigned long long)dsp_steps_);
    }
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ink_conv_arena_, CL_TRUE, 0, bytes, got.data(), 0, nullptr, nullptr));
  }
  std::vector<uint8_t> dsp_check_before_;
  std::vector<int> dsp_cmp_gpu_;
 public:
  uint64_t dsp_checks_ = 0, dsp_check_bad_ = 0, dsp_cmp_n_ = 0, dsp_cmp_same_ = 0;
 private:

  // the target's pass over [tok, drafts...] (as verify): the conv states kept and their inputs saved (spec_)
  const std::vector<float>& dsp_verify(int tok, const std::vector<int>& drafts) {
    const int T = 1 + (int)drafts.size();
    // (spec_logits_host_ holds 1 + max_drafts() rows, as verify() checks; and the block's B rows bound the saves)
    if (drafts.empty() || (int)drafts.size() > max_drafts() || T > dsp_B_)
      throw std::runtime_error("DSpark verify: " + std::to_string(drafts.size()) + " drafts");
    if (pos + T > ctx) throw std::runtime_error("context overflow");
    std::vector<int> toks{tok};
    toks.insert(toks.end(), drafts.begin(), drafts.end());
    last_drafts_ = drafts;
    embed(toks.data(), T, 16);
    spec_ = true;
    struct Off { bool& s; ~Off() { s = false; } } off{spec_};
    for (auto& L : layers_) run_layer(L, T, 16);
    spec_ = false;
    pos += T;
    head_rows(0, T, out_norm_, nullptr);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * T * 4, spec_logits_host_.data(), 0,
                                 nullptr, nullptr));
    if (pos > written_hi_) written_hi_ = pos;
    collect_profile();
    return spec_logits_host_;
  }
  // the first `accepted` drafts were right: the conv states moved on by accepted + 1 rows, the position back, the
  // accepted rows into the drafter's context
  void dsp_verify_done(int accepted) {
    // (no verification pending: nothing to move on; the saves would be another pass's rows. As verify_done)
    if (last_drafts_.empty()) return;
    const int nd = (int)last_drafts_.size();
    accepted = std::clamp(accepted, 0, nd);
    const int rows = accepted + 1;
    static const bool check = getenv("ARC_DSP_CHECK") && atoi(getenv("ARC_DSP_CHECK")) != 0;
    if (check) {
      const size_t bytes = ink_conv_off_.back() + (size_t)hp.n_embd * (ink_.nc - 1) * 4;
      dsp_check_before_.resize(bytes);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, ink_conv_arena_, CL_TRUE, 0, bytes, dsp_check_before_.data(), 0, nullptr, nullptr));
    }
    {
      cl_kernel k = ink_k("ink_conv_commit");
      int n = dsp_ntab_, r = rows, NC = ink_.nc;
      set_args(k, 0, ink_conv_arena_, dsp_save_arena_, dsp_tab_, n, r, NC);
      size_t gws[2] = {((size_t)dsp_maxc_ + 63) / 64 * 64, (size_t)dsp_ntab_}, lws[2] = {64, 1};
      run("ink_conv_commit", k, 2, gws, lws);
    }
    if (check) dsp_check_commit(rows);
    pos -= nd - accepted;
    dsp_commit(0, rows, pos - rows);
    ++dsp_steps_;
    dsp_drafted_ += nd;
    dsp_accepted_ += accepted;
    last_drafts_.clear();
  }
