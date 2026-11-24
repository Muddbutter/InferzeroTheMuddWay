// Qwen3.8 Flash Next: hyper-connection mixers and the PLE n-gram embeddings.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

  // ---------------------------------------------------------------- Flash Next hyper-connections and PLE
  // The mixer of rows r0 .. r0 + T - 1 of the residual streams x_: per-stream norm, low-rank down (+ inject logits,
  // into hc_lo_), silu(lo / hc_n), up to the stream gates, then the gated mean over the streams = the block input
  // (f32 in hco_, quantized in aq_ / asc_ / absum_ with Mp rows)
  void hc_mix(cl_mem normw, const GpuWeight& down, const GpuWeight& up, int T, int Mp, int act32, int r0) {
    const int E = hp.n_embd, HC = hp.hc_n, W = hp.hce(), LD = down.N, LR = hp.hc_lr, KP = hc_lr_pad();
    float eps = hp.eps;
    // single tokens and up to 8 rows (a verification pass: each row computed exactly as a single token, -DHC_RT): two
    // fused kernels with f32 activations (norm + down + inject, then silu + up + gated mean) and the block input's
    // quantization; prompt chunks below: int8 GEMMs
    if (T <= 8 && W <= 10240 && KP <= 1024) {
      if (!hc_rinv_) hc_rinv_ = buf(64 * 4);
      int rr = r0, e = E, hc = HC, n = down.N, lr = LR, kp = KP;
      const std::string rt = T > 1 ? "-DHC_RT=" + std::to_string(T) : std::string();
      cl_kernel k1 = kernel("ops", rt + hc_weight_defs(down), "hc_down1");
      set_args(k1, 0, x_, rr, normw, e, hc, eps, down.raw, n, hc_xn_, hc_rinv_);
      size_t g1[2] = {(size_t)256 * ((n + 15) / 16), 8}, l1[2] = {256, 1};  // (8 K slices: HC_KS)
      run("hc_down1", k1, 2, g1, l1);
      // (the block input's quantization inside hc_up1, -DHC_UPQ, 2026-11-18: one launch fewer a mixer, bit-identical;
      //  ARC_HC_UPQ=0: the separate quant_rows as before)
      static const bool upq_env = !getenv("ARC_HC_UPQ") || atoi(getenv("ARC_HC_UPQ")) != 0;
      const bool upq = upq_env && E % 256 == 0;
      cl_kernel k2 = kernel("ops", rt + hc_weight_defs(up) + (upq ? " -DHC_UPQ " + h8f_defs_ : std::string()), "hc_up1");
      set_args(k2, 0, x_, rr, normw, hc_rinv_, hc_xn_, n, hc_lo_, lr, kp, e, hc, up.raw, hco_);
      if (upq) {
        if (!hc_cnt_) {
          const std::vector<int> z((size_t)E / 256, 0);
          hc_cnt_ = buf(z.size() * sizeof(int), z.data());
        }
        int mm = Mp, a32 = act32;
        set_args(k2, 13, mm, aq_, asc_, absum_, a32, hc_cnt_);  // (after hc_up1's 13 arguments)
      }
      size_t g2 = (size_t)256 * ((E + 15) / 16), l2 = 256;
      run("hc_up1", k2, 1, &g2, &l2);
      rd_bytes += raw_bytes(down) + raw_bytes(up);
      if (!upq) quant_rows(hco_, E, Mp, T, act32);
      return;
    }
    {
      cl_kernel k = kernel("ops", "", "hc_norm_quant");
      int rr = r0, e = E, hc = HC, mm = Mp, tt = T, a32 = act32;
      set_args(k, 0, x_, rr, normw, e, hc, mm, tt, eps, hc_xn_, aq_, asc_, absum_, a32);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("hc_norm", k, 2, gws, lws);
    }
    // (ARC_HC_F32=1: the prompt chunk's mixers with f32 activations as single tokens have them, hc_gemm_f32; Flash Next
    //  is sensitive there: the f32 reference with int8 mixer inputs drifts by KL 0.06 on its own)
    // (1: both projections, 2: the down projection only, 3: the up projection only)
    // (4, the default since 20 Oct: the up projection's input as two int8 passes, the values then their rounding
    //  residual, accumulated: KL 0.0647 -> 0.0483 against the f32 reference, prompts 5-7 % slower; 0: one int8 pass)
    // (BF16 mixer rows: always the f32 GEMMs, mode 1)
    // (the environment's mode read once; precise mode, which a loaded AR4 model turns on, asked every time)
    static const int env_mode = [] {
      const int m = getenv("ARC_HC_F32") ? atoi(getenv("ARC_HC_F32")) : 4;
      if (m < 0 || m > 4) zlog::eprintf("ARC_HC_F32=%d: not 0..4, the int8 mixers\n", m);
      return m < 0 || m > 4 ? 0 : m;
    }();
    const int f32_mode = hc_bf16() ? 1 : env_mode;
    static bool hc_logged = false;
    if (!hc_logged) {
      hc_logged = true;
      static const char* names[] = {"int8", "f32 down + up", "f32 down", "f32 up", "two-pass int8 up"};
      zlog::eprintf("HC mixers of prompt chunks: %s (ARC_HC_F32=%d)\n", f32_mode >= 0 && f32_mode <= 4 ? names[f32_mode] : "int8", f32_mode);
    }
    const bool f32_down = f32_mode == 1 || f32_mode == 2, f32_up = f32_mode == 1 || f32_mode == 3;
    auto gemm_f32 = [&](cl_mem in, int ld_in, int K, int lo_in, const GpuWeight& w, int N, cl_mem C, int ldc) {
      hc_gemm_f32(in, ld_in, K, lo_in, w, N, C, ldc, Mp);
    };
    if (f32_down)
      gemm_f32(hc_xn_, HC * E, HC * E, 0, down, LD, hc_lo_, LD);
    else
      matmul("hc_down", down, Mp, hc_lo_, LD, 0, 0, act32);
    if (f32_up) {
      gemm_f32(hc_lo_, LD, KP, 1, up, W, hc_gate_, W);
    } else {
      {
        cl_kernel k = kernel("ops", "", "hc_lo_quant");
        int lr = LR, ld = LD, kp = KP, hc = HC, mm = Mp, tt = T, a32 = act32;
        set_args(k, 0, hc_lo_, lr, ld, kp, hc, mm, tt, aq_, asc_, absum_, a32);
        size_t gws[2] = {(size_t)16 * KP / 256, (size_t)Mp}, lws[2] = {16, 1};
        run("hc_lo", k, 2, gws, lws);
      }
      matmul("hc_up", up, Mp, hc_gate_, W, 0, 0, act32);
      if (f32_mode == 4 && !q2mm_) {  // (two-pass matmuls: hc_lo_quant stored the residual already)
        cl_kernel k = kernel("ops", "", "hc_lo_quant_res");
        int lr = LR, ld = LD, kp = KP, hc = HC, mm = Mp, tt = T, a32 = act32;
        set_args(k, 0, hc_lo_, lr, ld, kp, hc, mm, tt, aq_, asc_, absum_, a32);
        size_t gws[2] = {(size_t)16 * KP / 256, (size_t)Mp}, lws[2] = {16, 1};
        run("hc_lo", k, 2, gws, lws);
        matmul("hc_up", up, Mp, hc_gate_, W, 0, 1, act32);
      }
    }
    {
      cl_kernel k = kernel("ops", "", "hc_gate_mean_quant");
      int e = E, hc = HC, mm = Mp, tt = T, a32 = act32;
      set_args(k, 0, hc_xn_, hc_gate_, e, hc, mm, tt, hco_, aq_, asc_, absum_, a32);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("hc_mix", k, 2, gws, lws);
    }
  }
  // C [Mp][ldc] = in [Mp][ld_in] (K columns; lo_in: silu(in / hc_n) of the first hc_lr, zeros past them) times the
  // exact rows w [N][K] (BF16 or f32), f32 sums: the mixers' f32 GEMMs, and the PLE block's in precise mode
  void hc_gemm_f32(cl_mem in, int ld_in, int K, int lo_in, const GpuWeight& w, int N, cl_mem C, int ldc, int Mp) {
    // (rows a work-group, HCF_ROWS: each output is one work-item's sum in the same order at any value, so more rows only
    //  read the weights fewer times (a 2,048-row chunk at 16: 128 times); ARC_HCF_ROWS=16 / 32. Not 64: its rows,
    //  64 x 256 floats, are the whole 64 KiB of local memory (self audit 2026-11-19))
    static const int hcf_rows = [] {
      const int v = getenv("ARC_HCF_ROWS") ? atoi(getenv("ARC_HCF_ROWS")) : 16;
      return v == 32 ? v : 16;
    }();
    // (ARC_HCF_NB=n: output columns a work-item, 1 / 2 / 4, ops_flashnext.cl hc_gemm_f32: each input read once from
    //  local memory for all n; the same sums in the same order, bit-identical at any n)
    static const int hcf_nb = [] {
      const int v = getenv("ARC_HCF_NB") ? atoi(getenv("ARC_HCF_NB")) : 1;
      return v == 2 || v == 4 ? v : 1;
    }();
    // (ocloc, BF16 rows, 16 rows a group: mtl spills at NB 2 (450 scratch messages) and 4 (1,515), at 256 registers
    //  at NB 4 (411); bmg-g31 none at NB 2, none at NB 4 with 256 registers, the loop 1,102 / 1,772 / 3,030
    //  instructions for 1 / 2 / 4 columns. So NB 4 is built with 256 registers; the laptop's iGPU is slower at 2 and
    //  4 (spills): opt-in until a B-series card measures it)
    cl_kernel k = kernel("ops", hc_weight_defs(w) + " -DHCF_ROWS=" + std::to_string(hcf_rows) +
                                    (hcf_nb > 1 ? " -DHCF_NB=" + std::to_string(hcf_nb) : std::string()) +
                                    (hcf_nb == 4 ? " -cl-intel-256-GRF-per-thread" : ""),
                         "hc_gemm_f32");
    int ldi = ld_in, kk = K, li = lo_in, lr = hp.hc_lr, hc = hp.hc_n, nn = N, lc = ldc, mm = Mp;
    set_args(k, 0, in, ldi, kk, li, lr, hc, w.raw, nn, C, lc, mm);
    const size_t nwi = (size_t)(N + hcf_nb - 1) / hcf_nb;  // (work-items: hcf_nb columns each)
    size_t gws[2] = {(size_t)256 * ((nwi + 255) / 256), (size_t)(Mp + hcf_rows - 1) / hcf_rows}, lws[2] = {256, 1};
    run("hc_gemm_f32", k, 2, gws, lws);
  }
  // x_ streams += 2 sigmoid(inject / hc_n) * block output (hco_), inject = columns hc_lr.. of hc_lo_ (row stride ld)
  void hc_combine(int ld, int T) {
    cl_kernel k = kernel("ops", "", "hc_combine");
    int l = ld, off = hp.hc_lr, e = hp.n_embd, hc = hp.hc_n;
    set_args(k, 0, x_, hco_, hc_lo_, l, off, e, hc);
    size_t gws[2] = {(size_t)hp.hce(), (size_t)T};
    run("hc_combine", k, 2, gws, nullptr);
  }
  // the token ids before the current position, from the PLE layer's state (restored with snapshots / sessions)
  void ple_sync() {
    int32_t v[2] = {0, 0};
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, CL_TRUE, ple_tok_off(), 8, v, 0, nullptr,
                                 nullptr));
    ple_prev_[0] = v[0] - 1;  // (stored + 1: a reset state of zeros reads as "no token")
    ple_prev_[1] = v[1] - 1;
  }
  // n-gram rows of the chunk's tokens (host hash, rows read from the table in the GGUF) into ple_emb_; advances the
  // token history (host and the PLE layer's state). llama.cpp qwen4exp: predecessors that are missing (sequence
  // start) or an EOS cut the window; everything at or before the cut reads as EOS. An image's rows (forward_embd)
  // are <|image_pad|> here, ple.image_token_id, as the model's own input ids have them.
  void ple_embed(const int* tokens, int T) {
    const auto h0 = hclk::now();
    struct Acc { double& t; hclk::time_point a; ~Acc() { t += hms(a); } } acc{hst_ple_ms_, h0};
    if (!batch_.empty()) {  // a batched pass: each sequence's rows after its slot's token ids, into its slot's state
      for (const BatchSeq& q : batch_) {
        SlotHost& h = slot_host_[q.slot];
        ple_rows(tokens + q.row, q.n, h.ple_prev[0], h.ple_prev[1], q.row);
      }
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_emb_, CL_TRUE, 0, (size_t)T * hp.n_embd * 4, ple_host_.data(), 0,
                                    nullptr, nullptr));
      for (const BatchSeq& q : batch_) {
        SlotHost& h = slot_host_[q.slot];
        const int* tk = tokens + q.row;
        h.ple_prev[1] = q.n >= 2 ? tk[q.n - 2] : h.ple_prev[0];
        h.ple_prev[0] = tk[q.n - 1];
        const int32_t v[2] = {h.ple_prev[0] + 1, h.ple_prev[1] + 1};
        point_to(q.slot);
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, CL_TRUE, ple_tok_off(), 8, v, 0,
                                      nullptr, nullptr));
      }
      return;
    }
    ple_rows(tokens, T, ple_prev_[0], ple_prev_[1], 0);
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_emb_, CL_TRUE, 0, (size_t)T * hp.n_embd * 4, ple_host_.data(), 0, nullptr, nullptr));
    ple_prev_[1] = T >= 2 ? tokens[T - 2] : ple_prev_[0];
    ple_prev_[0] = tokens[T - 1];
    const int32_t v[2] = {ple_prev_[0] + 1, ple_prev_[1] + 1};
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, CL_TRUE, ple_tok_off(), 8, v, 0, nullptr,
                                  nullptr));
  }
  // the n-gram rows of tokens[0 .. T) after the token ids p0 (just before) and p1 into ple_host_ rows r0 ..
  void ple_rows(const int* tokens, int T, int p0, int p1, int r0) {
    const int ng = hp.ple_ng, per = hp.ple_per, dim = hp.ple_dim, E = hp.n_embd;
    const int64_t eos = hp.ple_eos;
    const size_t prb = ple_table_->nbytes / ple_table_->dims[1];
#pragma omp parallel for if (T > 16)
    for (int t = 0; t < T; ++t) {
      int64_t ctxv[8];
      ctxv[0] = tokens[t];
      bool cut = false;
      for (int sft = 1; sft < ng; ++sft) {
        const int j = t - sft;
        const int64_t tk = cut ? -1 : j >= 0 ? tokens[j] : j == -1 ? p0 : j == -2 ? p1 : -1;
        cut = cut || tk < 0 || tk == eos;
        ctxv[sft] = cut ? eos : tk;
      }
      for (int n = 2; n <= ng; ++n) {
        uint64_t mixed = (uint64_t)ctxv[0] * (uint64_t)hp.ple_mult[0];
        for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctxv[j] * (uint64_t)hp.ple_mult[j];
        for (int g = 0; g < per; ++g) {
          const int h = (n - 2) * per + g;
          const uint64_t row = mixed % (uint64_t)hp.ple_voc[h] + (uint64_t)hp.ple_off[h];
          dequant_row(ple_table_->type, ple_table_->data + row * prb, dim,
                      &ple_host_[(size_t)(r0 + t) * E + (size_t)h * dim]);
        }
      }
    }
  }
  // the PLE block on the residual streams (before the layer's first mixer): gates from the key / query match, then
  // the dilated causal conv over the normed gated values (history in the layer's state)
  void ple_block(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, W = hp.hce(), a32 = act32_for(false);
    if (L.ple_key.type == GGML_BF16 || L.ple_key.type == GGML_F32) {
      // (precise mode: the file's rows on f32 inputs, engine_load.h load_exact; 83 tokens of markdown, the block's
      //  value 0.45 % and its key 0.15 % off the two-pass reference with Q8_0 rows)
      // (up to 8 rows: gemv_bf16, one read of the rows for all; hc_gemm_f32's 16-row tiles made a generated token
      //  ~5 % slower on the laptop)
      auto gemv = [&](const GpuWeight& w, int N, cl_mem out) {
        cl_kernel k = kernel("ops", std::string("-DRT=1 -DTT=1") + (w.type == GGML_F32 ? " -DW_F32" : ""), "gemv_bf16");
        int kk = E, nn = N, tt = T, ld = N;
        set_args(k, 0, ple_emb_, kk, w.raw, nn, tt, out, ld);
        size_t gws[2] = {(size_t)16 * N, (size_t)T}, lws[2] = {16, 1};
        run("ple_gemv", k, 2, gws, lws);
      };
      if (T <= 8) {
        gemv(L.ple_key, W, hc_gate_);
        gemv(L.ple_value, E, hco_);
      } else {
        hc_gemm_f32(ple_emb_, E, E, 0, L.ple_key, W, hc_gate_, W, Mp);
        hc_gemm_f32(ple_emb_, E, E, 0, L.ple_value, E, hco_, E, Mp);
      }
    } else {
      quant_rows(ple_emb_, E, Mp, T, a32);
      matmul("ple_key", L.ple_key, Mp, hc_gate_, W, 0, 0, a32);
      matmul("ple_value", L.ple_value, Mp, hco_, E, 0, 0, a32);
    }
    float eps = hp.eps;
    {
      cl_kernel k = kernel("ops", "", "ple_gate");
      int e = E, hc = hp.hc_n;
      set_args(k, 0, hc_gate_, hco_, x_, L.ple_nk, L.ple_nq, L.ple_nc, e, hc, eps, hc_xn_, big_);
      size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
      run("ple", k, 2, gws, lws);
    }
    // (debug, ARC_DBG_PLE=file: the last row's key, value, streams, gated values and their norms, then the streams
    //  after the block; tools/ref_forward REF_DBG_PLE writes the same)
    static const char* dbg_ple = getenv("ARC_DBG_PLE");
    auto dbg_rd = [&](FILE* f, cl_mem b, int ld, int n) {
      std::vector<float> v((size_t)n);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, b, CL_TRUE, (size_t)(T - 1) * ld * 4, (size_t)n * 4, v.data(), 0, nullptr, nullptr));
      fwrite(v.data(), 4, v.size(), f);
    };
    FILE* fdbg = dbg_ple ? fopen(dbg_ple, "ab") : nullptr;  // (appended: one record a pass)
    if (fdbg) {
      dbg_rd(fdbg, hc_gate_, W, W);
      dbg_rd(fdbg, hco_, E, E);
      dbg_rd(fdbg, x_, W, W);
      dbg_rd(fdbg, hc_xn_, W, W);
      dbg_rd(fdbg, big_, W, W);
    }
    // (the conv over each sequence's rows and its history; a batched pass: per sequence, in its slot's state)
    auto seq = [&](int r0, int Tq) {
      const size_t off = (size_t)r0 * W * 4;
      {
        cl_kernel k = kernel("ops", "", "ple_conv_add");
        int w = W, nt = hp.ple_taps, dil = hp.ple_ng, hist = ple_hist();
        set_args(k, 0, from(x_, off), from(hc_xn_, off), from(big_, off), ple_hist_, L.ple_cw, w, nt, dil, hist);
        size_t gws[2] = {(size_t)W, (size_t)Tq};
        run("ple", k, 2, gws, nullptr);
      }
      {
        cl_kernel k = kernel("ops", "", "ple_hist_update");
        int w = W, tt = Tq, hist = ple_hist();
        set_args(k, 0, ple_hist_, from(big_, off), w, tt, hist);
        size_t g1 = W;
        run("ple", k, 1, &g1, nullptr);
      }
    };
    if (batch_.empty()) seq(0, T);
    else
      for (const BatchSeq& q : batch_) {
        point_to(q.slot);
        seq(q.row, q.n);
      }
    if (fdbg) {
      dbg_rd(fdbg, x_, W, W);
      fclose(fdbg);
    }
    if (spec_) CL_CHECK(clEnqueueCopyBuffer(cl_.queue, big_, ple_sv_, 0, 0, (size_t)T * W * 4, 0, nullptr, nullptr));
  }
  // before a verification pass: the PLE / QSA state and the token ids before it (see hc_rollback)
  void hc_spec_save(int tok) {
    spec_prev_[0] = ple_prev_[0];
    spec_prev_[1] = ple_prev_[1];
    spec_tok_ = tok;
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, hc_bak_, ple_hist_off(), 0,
                                 conv_bytes() - ple_hist_off(), 0, nullptr, nullptr));
  }
  // after a verification pass with rejected drafts (pos still after the whole pass): the PLE history, the token ids and
  // every QSA layer's tail as after the pass's first `rows` tokens (the state before the pass, then the same kernels
  // over those rows of the pass)
  void hc_rollback(int rows) {
    const int p0 = pos - 1 - (int)last_drafts_.size();
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hc_bak_, layers_[hp.ple_layer].conv_state, 0, ple_hist_off(),
                                 conv_bytes() - ple_hist_off(), 0, nullptr, nullptr));
    {
      cl_kernel k = kernel("ops", "", "ple_hist_update");
      int w = hp.hce(), tt = rows, hist = ple_hist();
      set_args(k, 0, ple_hist_, ple_sv_, w, tt, hist);
      size_t g1 = w;
      run("ple", k, 1, &g1, nullptr);
    }
    qsa_tails_from_pass(p0, rows);  // (engine_qwen.h: the build options and arguments qsa_seq's tail kernel takes)
    auto tk = [&](int i) { return i == 0 ? spec_tok_ : last_drafts_[i - 1]; };
    ple_prev_[0] = tk(rows - 1);
    ple_prev_[1] = rows >= 2 ? tk(rows - 2) : spec_prev_[0];
    const int32_t v[2] = {ple_prev_[0] + 1, ple_prev_[1] + 1};
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, CL_TRUE, ple_tok_off(), 8, v, 0, nullptr,
                                  nullptr));
  }
  // the mean of the residual streams of rows 0 .. T into hco_ (the DFlash2 features of Flash Next)
  void hc_mean(int T) {
    cl_kernel k = kernel("ops", "", "hc_mean");
    int e = hp.n_embd, hc = hp.hc_n;
    set_args(k, 0, x_, e, hc, hco_);
    size_t gws[2] = {(size_t)hp.n_embd, (size_t)T};
    run("hc_mean", k, 2, gws, nullptr);
  }
