// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from (the
// class is the same text), not a header of its own: expert streaming (mixture-of-experts weights read from the SSD into an expert cache).

  // ---------------------------------------------------------------- inferzero expert streaming
  void open_experts() {
    xc_ = std::make_unique<XCache>();
    std::vector<std::string> pre(hp.n_layer + 1);
    for (int il = 0; il <= hp.n_layer; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      // (an MTP head from a file of its own is not in the model's expert file: its experts are pinned, pin_mtp_experts)
      if (il == hp.n_layer && mtp_ext_) continue;
      if (f_.find(p + "ffn_gate_exps.weight") || f_.find(p + "ffn_gate_up_exps.weight")) pre[il] = p;
    }
    // (the expert types the kernels take, checked before the expert file is written: a BF16 Flash Next wrote its
    //  257.7 GB expert file first, then stopped at its first float matrix (manoreken2, issue #4, 2026-11-17))
    if (hp.moe_std)
      for (const std::string& p : pre) {
        if (p.empty()) continue;
        for (const char* n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_up_exps.weight"}) {
          const GgufTensor* t = f_.find(p + n);
          // (Inkling 2: its BF16 experts written as Q8_0 records, ExpertStore::float_to_q8, docs/inkling2.md I1.7)
          if (t && !is_w8_type(t->type) && !(hp.inkling && ExpertStore::is_float_type(t->type)))
            throw std::runtime_error("unsupported expert tensor " + t->name + " (" + ggml_type_name(t->type) +
                                     "): the experts run in a quantized type only (Q8_0 is close to the unquantized model)");
        }
      }
    xc_->st.float_to_q8 = hp.inkling;
    xc_->st.open_or_build(f_, path_, pre, hp.n_expert);
    // (staging records per half: ARC_XSTAGE, default 64 = 8 verification tokens x 8 experts)
    // (a layer split's stage: the cache for its own layers' experts, and the MTP block's on the last stage)
    std::vector<char> keep;
    if (opt.pp_size > 1) {
      const int lb = std::max(0, opt.pp_begin), le = opt.pp_end < 0 ? hp.n_layer : std::min(opt.pp_end, hp.n_layer);
      keep.assign(hp.n_layer + 1, 0);
      for (int il = lb; il < le; ++il) keep[il] = 1;
      keep[hp.n_layer] = pp_last();
    }
    xc_->init(hp.n_layer + 1, hp.n_expert, getenv("ARC_XSTAGE") ? atoi(getenv("ARC_XSTAGE")) : 64, keep.empty() ? nullptr : &keep);
    if (tp_moe()) xc_->set_range((int)tpe_.first, (int)tpe_.second);  // (multi-GPU: this GPU's experts only)
    if (xc_->st.built)
      zlog::eprintf("inferzero: expert file written in %.1f s\n", xc_->st.build_ms / 1000);
    // (--expert-ram-gb when given, else ARC_XRAM_GB, else automatic: the option wins since 2026-11-22, as --npu and
    //  --precise do over their variables; the variable overrode it before, the repair list A3)
    const double ram_gb = opt.expert_ram_gb >= 0 ? opt.expert_ram_gb : getenv("ARC_XRAM_GB") ? atof(getenv("ARC_XRAM_GB")) : -1.0;
    uint64_t ram = 0;
    if (ram_gb > 0) {
      ram = (uint64_t)(ram_gb * 1073741824.0);
    } else if (ram_gb < 0 && !zrt::integrated() && tp_on()) {
      // (a split model: at most 32 GiB per GPU, and at most half the RAM free beyond 16 GiB shared by the GPUs. An
      //  unbounded tier (a quarter of the free RAM each, ~240 GiB) ran DeepSeek-V4.1-Flash Q8_K_XL over 4 Pro B60s with
      //  1 TB of RAM out of memory at its first prompt: the pinned host memory of the one context the shards share.
      //  manoreken2's sweep over the same 4 B60s (issue #4, 2026-11-16, release .10), ARC_XRAM_GB 0 / 8 / 16 / 32 per GPU:
      //  512-token prompt 14.4 / 14.5 / 15.9 / 23.9 tok/s, generation 3.55 / 6.69 / 8.51 / 9.25, every size without an
      //  error. ARC_XRAM_GB=0: none; =N: N GiB per GPU)
      // (and at most 128 GiB for all the GPUs together: the same machine with 48 and 64 GiB per GPU (192 / 256 GiB
      //  pinned in all) stopped after loading, 4 x 32 = 128 ran; release .16, 2026-11-17)
      const uint64_t keep = 16ull << 30, avail = plat::ram_available(),
                     cap = std::min<uint64_t>(32ull << 30, (128ull << 30) / std::max(1, opt.tp_size));
      ram = avail > keep ? std::min<uint64_t>(cap, (avail - keep) / 2 / std::max(1, opt.tp_size)) : 0;
    } else if (ram_gb < 0 && !zrt::integrated()) {
      // (the free RAM less 16 GiB, or where more: half the free RAM, leaving at least 4 GiB, since 2026-11-22. A
      //  32 GB machine got no tier: StealthDrifter's B580 (#80), the 35B Q3_K_M with 14.09 GB of experts and 6.31 GiB
      //  of them cached, 174 GB from the SSD in 150 s; --expert-ram-gb 7-8: 92 -> 97-98 % of the experts found, an
      //  uncached prompt 94-681 -> 1,358-1,532 tok/s. A machine with more free RAM keeps what it had)
      const uint64_t keep = 16ull << 30, avail = plat::ram_available(), low = 4ull << 30;
      const uint64_t r_big = avail > keep ? avail - keep : 0, r_half = avail > low ? std::min(avail / 2, avail - low) : 0;
      ram = std::max(r_big, r_half) / std::max(1, opt.tp_size);
    }
    ram = std::min(ram, xc_->st.file_bytes);
    // (Windows, a discrete card: pinned host memory it copies from counts against what Windows lets the GPU map of
    //  RAM (WDDM's non-local budget, ~half the RAM), on top of what the engine maps already; a tier past it fails the
    //  copies' residency. LukaszBrzyszkiewicz's B70 (#82, 96 GB RAM, Flash Next): a 52.1 GiB tier against a 47.9 GiB
    //  budget, out of memory at the first prompt's copy. The automatic size is capped at the budget less the current
    //  use and 4 GiB; a size given by hand is kept, with a warning. Linux has no such budget)
    std::string ram_note;
    if (!zrt::integrated() && ram > 0) {
      uint64_t nb = 0, nu = 0;
      if (zrt::video_budget(nb, nu, true)) {
        const uint64_t margin = 4ull << 30, room = (nb > nu + margin ? nb - nu - margin : 0) / std::max(1, opt.tp_size);
        char b[200];
        // (an automatic size: the engine's own, or a layer split's stage share (opt.expert_ram_auto); a size the user
        //  set is kept with the warning. A stage's share was taken for a hand-set one: no cap, and a warning naming
        //  options never given, issue #82's failure on a split; the audit of 2026-11-23)
        const bool auto_size = ram_gb < 0 || opt.expert_ram_auto;
        if (ram > room && auto_size) {
          ram = room;
          snprintf(b, sizeof b, " (capped by what Windows lets the GPU map of RAM: %.1f GiB, %.1f in use)", nb / 1073741824.0,
                   nu / 1073741824.0);
          ram_note = b;
        } else if (ram > room) {
          zlog::eprintf("inferzero: warning: --expert-ram-gb / ARC_XRAM_GB %.1f GiB is more than Windows lets the GPU map of "
                        "RAM now (%.1f GiB left of %.1f): its copies may fail (issue #82)\n", ram / 1073741824.0,
                        room / 1073741824.0, nb / 1073741824.0);
        }
      }
    }
    if (ram >= (1ull << 30)) {
      // (pinned host memory of the context: every GPU of a split model copies from it)
      xc_->st.ram_alloc = [zc = xc_->zc](size_t n) -> char* {
        ze_host_mem_alloc_desc_t hd{ZE_STYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
        void* p = nullptr;
        return zeMemAllocHost(zc, &hd, n, 4096, &p) == 0 ? (char*)p : nullptr;
      };
      xc_->st.ram_free = [zc = xc_->zc](char* p) { zrt::mem_free(zc, p); };
      xc_->st.set_ram(ram);
      zlog::eprintf("inferzero: experts read from the SSD are also kept in RAM, up to %.1f GiB%s\n", ram / 1073741824.0,
                    ram_note.c_str());
    }
  }
  void size_expert_cache() {
    const double GiB = 1073741824.0;
    const uint64_t all = xc_->all_bytes();
    uint64_t want;
    // (--expert-cache-gb, then --expert-cache-pct, then ARC_XCACHE_GB, then automatic: the options win since
    //  2026-11-22; ARC_XCACHE_GB overrode both before, the repair list A3)
    const char* env_gb = getenv("ARC_XCACHE_GB");
    if (opt.expert_cache_gb > 0) {
      want = (uint64_t)(opt.expert_cache_gb * GiB);
    } else if (opt.expert_cache_pct > 0) {
      want = (uint64_t)(all * std::min(opt.expert_cache_pct, 100.0) / 100);
    } else if (env_gb) {
      want = (uint64_t)(atof(env_gb) * GiB);
    } else {
      // the whole process (weights, buffers, transient area, cache) within the GPU share Windows gives the iGPU, less
      // driver_slack() for the driver: the share is the user's setting, the rest of the RAM what they leave to Windows
      // (the user's rule since 2026-10-22; before: also RAM - a 14 GB Windows reserve, which held a 20.4 GiB share to
      // 17.7). (With 1.13 GiB left a kernel launch once failed with out of device memory, 35B at 262K.)
      // (device_mem(): this engine's GPU (a layer split's stage, not the thread's default) and ARC_GPU_MEM_GB, as the fit
      //  checks; zrt::D().global_mem ignored the cap, the audit of 2026-11-23)
      const uint64_t share = device_mem(), slack = (uint64_t)driver_slack();
      uint64_t limit = share > slack ? share - slack : 0;
      // (ARC_TP_SIM: every shard on GPU 0, each with its part of the memory)
      if (tp_on() && getenv("ARC_TP_SIM") && atoi(getenv("ARC_TP_SIM")) != 0) limit /= opt.tp_size;
      // (+ kernels and host side, the server's state snapshots, and what the caller keeps for its encoders)
      // (a split model: also the sum buffers its first pass allocates, tp_part_ f32, the f16 rows and the add buffer,
      //  ~max_chunk x n_embd x 10 bytes; a dry load counted them (engine.h), a real one did not, and the cache took
      //  their room: DeepSeek-V4.1 over 4 B60s, 262K, the cache planned to 0.01 GiB of the limit, the engine's memory
      //  then 20.48 -> 20.53 GiB on the first prompt and a launch out of device memory, issue #4, 2026-11-14)
      const uint64_t tp_sums = tp_on() ? (uint64_t)max_chunk * (uint64_t)hp.n_embd * 10 : 0;
      const uint64_t used = gpu_bytes + xc_->stage_bytes + (uint64_t)(0.1 * GiB) + snapshot_bytes() * std::max(opt.slots, 1) + opt.stream_reserve + tp_sums;
      want = limit > used ? limit - used : 0;
      zlog::eprintf("inferzero: memory limit %.2f GiB (GPU share %.2f - %.1f for the driver), model + buffers %.2f GiB\n",
              limit / GiB, share / GiB, driver_slack() / GiB, used / GiB);
    }
    want = std::min(want, all);  // (a layer split's stage: all_bytes() counts its own layers' experts, open_experts)
    // at least one layer's worth of slots per pool (the experts of a few verification tokens must fit)
    want = std::max<uint64_t>(want, xc_->min_bytes());
    xc_->set_cache(want);
    gpu_bytes += xc_->cache_bytes;
    // (out of device memory shrinks this cache: XCache::shrink_this_gpu; a split model's shards are on GPUs of
    //  their own, each cache registered with its shard's GPU)
    // (a one-GPU engine's cache registered for its GPU too: a split model left after a fault keeps its entries, and
    //  GPU 0's hook found that dead shard's cache before active(); audit 2026-11-14)
    if (!tp_on()) XCache::active() = xc_.get();
    XCache::on_gpu(&zrt::D(), xc_.get());
    XCache::install_hook();
    // fill the cache before the first request: the experts the last run had cached (or the first of every layer)
    // (multi-GPU: one state per GPU of the split, for its experts)
    xc_->state_path = xc_->st.path.substr(0, xc_->st.path.size() - 8) +
                      (tp_moe() ? ".gpu" + std::to_string(opt.tp_rank) + "of" + std::to_string(opt.tp_size) : std::string()) +
                      (opt.pp_size > 1 ? ".layers" + std::to_string(std::max(0, opt.pp_begin)) + "-" + std::to_string(opt.pp_end) : std::string()) +
                      ".lru";
    if (!getenv("ARC_XWARM") || atoi(getenv("ARC_XWARM")) != 0) {
      xc_->warm();
      zlog::eprintf("inferzero: cache filled from the SSD: %.2f GB in %.1f s\n", xc_->warm_bytes / 1e9, xc_->warm_ms / 1000);
    }
    zrt::print_memory();
    zlog::eprintf("inferzero: expert cache %.2f GiB (%.0f %% of %.2f GiB experts, %d slots), staging %.0f MiB, file %s\n",
            xc_->cache_bytes / GiB, 100.0 * xc_->cache_bytes / all, all / GiB, xc_->slots(), xc_->stage_bytes / 1048576.0,
            xc_->st.path.c_str());
    xcache_placement_check();
  }
  // (2026-11-23, #45: a discrete card nearly full of buffers can get part of a large allocation placed in system memory
  //  by the driver without an error. hansong80's A770 under Windows, the 35B with 14.6 GiB of buffers on 16 GB:
  //  generation alternated run to run, 43.7 against 23.2 tok/s, "weights read" 106 against 56 GB/s, the same load
  //  lines. Each 1 GiB piece of the expert cache is read here once, 64 MB of it, timed: the card's own memory reads
  //  186 GB/s or more (the A380 and up), PCIe ~20-30, so a piece under 64 GB/s is in system memory, and the run says
  //  how much, and what keeps it on the card. Discrete cards only, ~1 ms a GiB; ARC_XPLACE=0 skips it, =2 prints each
  //  piece's rate)
  static int xplace_mode() {
    static const int mode = getenv("ARC_XPLACE") ? atoi(getenv("ARC_XPLACE")) : 1;
    return mode;
  }
  // (a rate that means system memory: on a discrete card under 64 GB/s (its own memory reads 186 or more); on an iGPU
  //  (ARC_XPLACE=2, tests: all its memory is RAM, ~60-120 GB/s) under half the fastest one seen)
  static bool placement_slow(double gbs, double top) {
    return zrt::integrated() ? gbs < 0.5 * top : gbs < 64;
  }
  // (one device allocation's read rate, GB/s: up to 64 MB of it from ptr, the best of two timed reads after a warm-up;
  //  0 for one too small to time, under 16 MB)
  double placement_rate(cl_kernel k, cl_mem o, void* ptr, size_t size) {
    const size_t bytes = std::min<size_t>(size, (size_t)64 << 20) / 2048 * 2048;
    if (bytes < ((size_t)16 << 20)) return 0;  // (a small one: the launch's own cost would hide the rate)
    const size_t groups = std::max<size_t>(64, (size_t)std::max<uint32_t>(zrt::hw_threads(), 1024) * 4 / 16);
    int n = (int)(bytes / 4);
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
      set_args(k, 0, ptr, o, n);
      size_t gws[1] = {groups * 256}, lws[1] = {256};
      clFinish(cl_.queue);
      const auto t0 = std::chrono::steady_clock::now();
      run("xcache_place", k, 1, gws, lws);
      clFinish(cl_.queue);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (r > 0) best = std::min(best, ms);  // (the first: warm-up)
    }
    return bytes / 1e6 / best;
  }
  // (2026-11-23, #93: lact-lab's two A770s under Windows, the 35B at 4K 193 tok/s with gdn_recur 2,375 ms against
  //  ~200 on skipzx-lab's: not only the expert cache in system memory, the model's own buffers too. The same read for
  //  the engine's allocations of 64 MB or more (weights, the KV cache, the DeltaNet state, the work buffers), after
  //  everything is allocated. Discrete cards; ARC_XPLACE=0 skips it, =2 prints each one (also on an iGPU: tests))
  void buffers_placement_check() {
    const int mode = xplace_mode();
    if (mode == 0 || (zrt::integrated() && mode != 2) || opt.dry) return;
    cl_kernel k = cl_.kernel(rd_prog(), "rd_block");
    struct Rel { cl_kernel k; ~Rel() { clReleaseKernel(k); } } rel{k};
    cl_mem o = buf(64);
    struct FreeO { Engine& e; cl_mem o; ~FreeO() { e.free_buf(o); } } free_o{*this, o};
    const double GiB = 1073741824.0;
    double slow = 0, all = 0, slow_min = 1e30, fast_min = 1e30;
    int nslow = 0, n = 0;
    // (the largest 256 of 16 MB or more: a dense model's weights are a buffer each, most of them tens of MB)
    std::vector<cl_mem> big;
    for (cl_mem m : mems_)
      if (m && m->owner && m->size >= ((size_t)16 << 20) && m != o) big.push_back(m);  // (not sub-buffers: their parent is read)
    std::sort(big.begin(), big.end(), [](cl_mem a, cl_mem b) { return a->size > b->size; });
    if (big.size() > 256) big.resize(256);
    std::vector<std::pair<cl_mem, double>> rates;
    double top = 0;
    for (cl_mem m : big) {
      const double gbs = placement_rate(k, o, m->ptr, m->size);
      if (gbs > 0) rates.emplace_back(m, gbs), top = std::max(top, gbs);
    }
    for (const auto& [m, gbs] : rates) {
      ++n, all += m->size;
      if (placement_slow(gbs, top)) slow += m->size, ++nslow, slow_min = std::min(slow_min, gbs);
      else fast_min = std::min(fast_min, gbs);
      if (mode == 2) zlog::eprintf("inferzero: ARC_XPLACE: buffer of %.3f GiB: %.0f GB/s\n", m->size / GiB, gbs);
    }
    if (slow > 0)
      zlog::eprintf("inferzero: warning: %d of the model's %d large buffers (%.2f of %.2f GiB) are in system memory (read at "
                    "%.0f GB/s, the card's own memory at %.0f): every pass reads them over PCIe and the run is slow. A "
                    "shorter --ctx (or a smaller --expert-cache-gb on a MoE model) leaves the driver room for them (#93)\n",
                    nslow, n, slow / GiB, all / GiB, slow_min, fast_min < 1e29 ? fast_min : 0.0);
  }
  void xcache_placement_check() {
    const int mode = xplace_mode();
    if (mode == 0 || (zrt::integrated() && mode != 2) || opt.dry || !xc_) return;  // (=2 on an iGPU too: tests)
    cl_kernel k = cl_.kernel(rd_prog(), "rd_block");
    struct Rel { cl_kernel k; ~Rel() { clReleaseKernel(k); } } rel{k};
    cl_mem o = buf(64);
    struct FreeO { Engine& e; cl_mem o; ~FreeO() { e.free_buf(o); } } free_o{*this, o};
    const double GiB = 1073741824.0;
    double slow_bytes = 0, all_bytes = 0, slow_min = 1e30, fast_min = 1e30, top = 0;
    struct Piece { size_t pool, c, bytes; double gbs; };
    std::vector<Piece> pieces;
    for (size_t pi = 0; pi < xc_->pools.size(); ++pi) {
      auto& P = xc_->pools[pi];
      const int per = (int)std::max<uint64_t>(1, (1ull << 30) / P.rec);
      for (size_t c = 0; c < P.chunks.size(); ++c) {
        const int kslots = std::min(per, P.n - (int)c * per);
        const size_t piece = (size_t)kslots * P.rec;
        const double gbs = placement_rate(k, o, P.chunks[c], piece);
        if (gbs > 0) pieces.push_back({pi, c, piece, gbs}), top = std::max(top, gbs);
      }
    }
    for (const Piece& q : pieces) {
      all_bytes += q.bytes;
      if (placement_slow(q.gbs, top)) slow_bytes += q.bytes, slow_min = std::min(slow_min, q.gbs);
      else fast_min = std::min(fast_min, q.gbs);
      if (mode == 2)
        zlog::eprintf("inferzero: ARC_XPLACE: pool %zu piece %zu (%.2f GiB): %.0f GB/s\n", q.pool, q.c, q.bytes / GiB, q.gbs);
    }
    if (slow_bytes > 0)
      zlog::eprintf("inferzero: warning: %.2f of the expert cache's %.2f GiB are in system memory (read at %.0f GB/s, the "
                    "card's own memory at %.0f): their experts are read over PCIe and generation slows. A smaller cache keeps "
                    "it on the card: --expert-cache-gb %.1f (#45)\n",
                    slow_bytes / GiB, all_bytes / GiB, slow_min, fast_min < 1e29 ? fast_min : 0.0,
                    std::max(0.5, (xc_->cache_bytes - slow_bytes) / GiB - 0.5));
  }
  // After the routing of layer L: read the chosen experts and make them resident (see zero/xcache.h).
  // (tl1: one token's tiles when the caller has read them already)
  // (host time of streamed MoE, printed by expert_stats: the routing read (waits for the GPU), ensure (expert reads
  //  and copies while the GPU waits), and Flash Next's PLE rows)
  double hst_sync_ms_ = 0, hst_ensure_ms_ = 0, hst_ple_ms_ = 0;
  // (the same for one-token steps alone, plain decoding, and how many layers waited: per token and per wait,
  //  2026-11-17, for Flash Next's generation on a B70, ~25 ms a step outside its kernels)
  double gen_sync_ms_ = 0, gen_ensure_ms_ = 0;
  uint64_t gen_waits_ = 0;
  int seg_n_ = 0;  // tokens of the layer-major segment being processed (0: none)
  std::vector<int> stream_ids_;  // (stream_layer's expert ids)
  int xall_prev_il_ = 1 << 30;   // (ARC_XALL: the last MoE layer of one token; a lower one starts the next token)
  using hclk = std::chrono::steady_clock;
  static double hms(hclk::time_point a) { return std::chrono::duration<double, std::milli>(hclk::now() - a).count(); }
  void stream_layer(Layer& L, int T, bool gemm, const int* tl1 = nullptr) {
    const int NU = hp.n_expert_used;
    const auto h0 = hclk::now();
    std::vector<int>& ids = stream_ids_;  // (per engine: a split model's shards run this on threads of their own)
    ids.resize((size_t)T * NU);
    if (T == 1) {
      std::vector<int> tl(1 + 3 * NU);
      if (tl1)
        std::copy(tl1, tl1 + tl.size(), tl.begin());
      else
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
      // (NU tiles; fewer on a split model's GPU, its experts' only)
      ids.resize((size_t)std::clamp(tl[0], 0, NU));
      for (size_t k = 0; k < ids.size(); ++k) ids[k] = tl[1 + 3 * k];
    } else {
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, mids_, CL_TRUE, 0, ids.size() * 4, ids.data(), 0, nullptr, nullptr));
    }
    const auto h1 = hclk::now();
    const double sync_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();
    hst_sync_ms_ += sync_ms;
    // (Inkling: whether this layer runs again in this forward, a later loop pass: its experts kept at the list's front)
    //  (cleared on every way out, an ensure() that throws included: the Researcher's review of 4546f5d)
    struct KeepFront { bool& f; ~KeepFront() { f = false; } } keep_guard{xc_->keep_front};
    xc_->keep_front = hp.inkling && std::find(ink_order_.begin() + std::min(ink_pos_, ink_order_.size()), ink_order_.end(), L.il) !=
                                        ink_order_.end();
    xc_->ensure(L.il, ids.data(), ids.size(), T > 8, gemm);
    const double ens_ms = hms(h1);
    hst_ensure_ms_ += ens_ms;
    if (T == 1) gen_sync_ms_ += sync_ms, gen_ensure_ms_ += ens_ms, ++gen_waits_;
    static const bool xcheck = getenv("ARC_XCHECK") != nullptr;  // (debug: every tile's expert address is in a pool)
    if (xcheck) {
      const int max_t = T * NU + hp.n_expert + 1;
      std::vector<int> tl(1 + 3 * (size_t)max_t);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
      const uint64_t* xa = xc_->xaddr + (size_t)L.il * hp.n_expert;
      auto valid = [&](uint64_t a) {
        for (auto& P : xc_->pools)
          for (size_t c = 0; c < P.chunks.size(); ++c) {
            const uint64_t b = (uint64_t)P.chunks[c];
            if (a >= b && a < b + (1ull << 30) + P.rec) return true;
          }
        return false;
      };
      if (tl[0] < 0 || tl[0] > max_t) zlog::eprintf("XCHECK layer %d T %d: tile count %d\n", L.il, T, tl[0]);
      for (int t = 0; t < std::min(tl[0], max_t); ++t) {
        const int e = tl[1 + 3 * t];
        if (e < 0 || e >= hp.n_expert || !valid(xa[e])) {
          zlog::eprintf("XCHECK layer %d T %d tile %d/%d: expert %d address %llx not in the cache\n", L.il, T, t, tl[0], e,
                  e >= 0 && e < hp.n_expert ? (unsigned long long)xa[e] : 0ull);
          fflush(stderr);
          _Exit(3);
        }
      }
      if (gemm)
        for (int e = 0; e < hp.n_expert; ++e)
          if (!valid(xa[e])) { zlog::eprintf("XCHECK layer %d gemm: expert %d address invalid\n", L.il, e); _Exit(3); }
    }
    // prompt chunks use nearly every expert (128 tokens: ~98 %): read the next layer's missing ones while the GPU
    // works on this one (not across the chunk's end: the next chunk's layer 0 has this layer's parity at times)
    static const int pf_min = getenv("ARC_XPF_MIN") ? atoi(getenv("ARC_XPF_MIN")) : 128;
    // (Inkling: the layer that runs next, which after a loop pass's last layer is the loop's first; -1 after the last
    //  layer. I1.7 step 2, engine_inkling.h ink_forward. Other models: the next index, as before)
    const int next = hp.inkling ? ink_next_il_ : L.il + 1;
    // (the tokens that go through the next layer before the layer after it: a layer-major segment's, else the chunk's)
    if (T >= pf_min && next >= 0 && (next < hp.n_layer || (next == hp.n_layer && mtp_block_)))
      xc_->prefetch(next, seg_n_ ? seg_n_ : T);
  }
 public:
  // expert cache statistics (empty without streaming)
  std::string expert_stats() const {  // (+ host times, see stream_layer)
    if (!xc_) return "";
    const XCache& x = *xc_;
    char b[512];
    // (the rate only when reads were timed: the cache's fill at load counts its bytes but not into read_ms, and a
    //  B390 printed "19.59 GB in 0 ms (19587399680000.00 GB/s)", issue #75)
    char rate[48] = "";
    if (x.read_ms >= 1.0) snprintf(rate, sizeof rate, " (%.2f GB/s)", x.st.bytes_read / 1e6 / x.read_ms);
    snprintf(b, sizeof b,
             "experts: generation %llu hits / %llu misses (%.1f %% hit) | prompt %llu cached, %llu read (%llu prefetched) | "
             "SSD %.2f GB in %.0f ms%s",
             (unsigned long long)x.hits, (unsigned long long)x.misses, 100.0 * x.hits / std::max<uint64_t>(1, x.hits + x.misses),
             (unsigned long long)x.p_hits, (unsigned long long)x.p_reads, (unsigned long long)x.pf_reads,
             x.st.bytes_read / 1e9, x.read_ms, rate);
    std::string r = b;
    if (x.st.ram_bytes()) {
      char m[128];
      snprintf(m, sizeof m, " | RAM tier %.2f GB, %llu records from it", x.st.ram_bytes() / 1e9, (unsigned long long)x.st.ram_hits.load());
      r += m;
    }
    char h[160];
    snprintf(h, sizeof h, "\nhost: routing reads %.0f ms (waiting for the GPU), ensure %.0f ms, PLE rows %.0f ms",
             hst_sync_ms_, hst_ensure_ms_, hst_ple_ms_);
    std::string out = r + h;
    if (const std::string xh = x.xhist_line(); !xh.empty()) out += "\n" + xh;  // (ARC_XHIST, zero/xcache.h)
    if (xcpu_n_) {  // (ARC_XCPU, zero/xe2/engine_xcpu.h)
      char c[260];
      snprintf(c, sizeof c, "\nCPU experts (ARC_XCPU): %llu computed (%llu from RAM), %.0f ms (%.3f ms an expert, %d threads); cache fills %llu (%llu dropped); host: input reads %.0f ms, fills %.0f ms",
               (unsigned long long)xcpu_n_, (unsigned long long)xcpu_ram_, xcpu_ms_, xcpu_ms_ / xcpu_n_, xcpu_threads(), (unsigned long long)x.fills_done,
               (unsigned long long)x.fills_dropped, xcpu_rd_ms_, xcpu_fill_ms_);
      out += c;
    }
    if (gen_waits_) {  // (one-token steps alone: plain decoding)
      char g[224];
      snprintf(g, sizeof g, "\nhost, one-token steps: %llu layer waits, routing reads %.0f ms (%.3f ms a wait), ensure %.0f ms (%.3f ms a wait)",
               (unsigned long long)gen_waits_, gen_sync_ms_, gen_sync_ms_ / gen_waits_, gen_ensure_ms_, gen_ensure_ms_ / gen_waits_);
      out += g;
    }
    return out;
  }
 private:
  // part offsets of an expert record (gate, up, down)
  int xoff(const Layer& L, int i) const {
    const auto& li = xc_->st.layers[L.il];
    return i == 0 ? 0 : i == 1 ? (int)li.part[0] : (int)(li.part[0] + li.part[1]);
  }
  // decode_w8 of experts [e0, e0 + G) of matrix i from their records (moe_gemm)
  void decode_w8_x(const Layer& L, int i, int e0, int G, int N, int K) {
    const uint32_t type = L.ex_type[i];
    const int nsb = K / 256;
    if (N % 16) throw std::runtime_error("decode_w8_x: N = " + std::to_string(N) + " is not a multiple of 16");
    const bool sl = !w8_requant(type) && nsb <= 16 && N % 16 == 0 &&
                    16 * nsb * raw_padded_sb_bytes(type) <= 32768 && type != GGML_IQ1_M && type != GGML_IQ1_S;
    cl_mem tab = xc_->xtab[L.il];
    int NN = G * N, xo = xoff(L, i);
    if (sl) {
      cl_kernel k = kernel("decode", "-DXCACHE -DQT=" + std::to_string(type) + " -DSL_NSB=" + std::to_string(nsb), "decode_w8_sl");
      set_args(k, 0, tab, NN, K, tabs_, moe_wq_, moe_wh_, tab, e0, N, xo);
      size_t g1 = (size_t)NN * nsb, l1 = (size_t)16 * nsb;
      run("decode_w8", k, 1, &g1, &l1);
      return;
    }
    size_t lx = 256;
    while (NN % lx) lx /= 2;
    cl_kernel k = kernel("decode", "-DXCACHE -DQT=" + std::to_string(type), "decode_w8_sb");
    set_args(k, 0, tab, NN, K, tabs_, moe_wq_, moe_wh_, tab, e0, N, xo);
    size_t gws[2] = {(size_t)NN, (size_t)nsb}, lws[2] = {lx, 1};
    run("decode_w8", k, 2, gws, lws);
  }

  void load_moe(Layer& L, const std::string& p) {
    const int E = hp.n_embd, NR = hp.router_rows();
    std::vector<float> v((size_t)NR * E, 0.0f);
    const GgufTensor& r = f_.get(p + "ffn_gate_inp.weight");
    // (DeepSeek-V4.1: no shared-expert gate, its row stays zero; moe.cl ROUTE_DS sets the gate to 1)
    const GgufTensor* sg = hp.ds41 ? f_.find(p + "ffn_gate_inp_shexp.weight") : &f_.get(p + "ffn_gate_inp_shexp.weight");
    if (r.nelements() != (uint64_t)hp.n_expert * E || (sg && sg->nelements() != (uint64_t)E))
      throw std::runtime_error("unexpected MoE router shape");
    dequant_row(r.type, r.data, hp.n_expert * E, v.data());
    if (sg) dequant_row(sg->type, sg->data, E, v.data() + (size_t)hp.n_expert * E);
    auto q8 = quantize_rows_q8_0(v.data(), NR, E);
    // (Flash Next's precise mode: the router and shared-expert gate rows as BF16, as the file has them (exact, half
    //  the bytes of f32), read with f32 inputs: near-tie expert choices stay as the f32 reference makes them)
    // (DeepSeek-V4.1 always: its router is BF16 in the file and computed in f32 by the reference)
    // (ARC_FN_RT32=1, a test of 2026-11-22: only this part of precise mode, the router and shared gate in f32 on BF16
    //  rows, the default mode otherwise: whether the int8 router's near-tie choices carry Flash Next's KL on markdown)
    static const bool rt32 = getenv("ARC_FN_RT32") && atoi(getenv("ARC_FN_RT32")) != 0;
    if ((hp.hc && (fn_precise() || rt32)) || hp.ds41) {  // (BF16 as in the file: v holds exactly those values)
      std::vector<uint16_t> bf(v.size());
      for (size_t i = 0; i < v.size(); ++i) {
        uint32_t u;
        memcpy(&u, &v[i], 4);
        bf[i] = (uint16_t)(u >> 16);
      }
      L.router_bf16 = buf(bf.size() * 2, bf.data());
    }
    const GgufTensor& tg = f_.get(p + "ffn_gate_shexp.weight");
    const GgufTensor& tu = f_.get(p + "ffn_up_shexp.weight");
    const int Fs = hp.n_ff_shexp, NRP = rgu_rows();
    if (tg.type == GGML_Q8_0 && tu.type == GGML_Q8_0 && E % 256 == 0 && tg.nelements() == (uint64_t)Fs * E &&
        tu.nelements() == (uint64_t)Fs * E && !(w8_resident && !hp.moe)) {
      // router + shared-expert gate + up rows in one Q8_0 buffer: single tokens run one GEMV over all of them
      // (Layer::rgu); the three matrices are sub-buffers of it (stored once; their offsets are 512-byte aligned)
      const size_t rowb = (size_t)E / 32 * 34, sbr = (size_t)E / 256 * raw_padded_sb_bytes(GGML_Q8_0);
      std::vector<uint8_t> cat((size_t)(NRP + 2 * Fs) * rowb, 0);
      memcpy(cat.data(), q8.data(), q8.size());
      memcpy(cat.data() + (size_t)NRP * rowb, tg.data, tg.nbytes);
      memcpy(cat.data() + (size_t)(NRP + Fs) * rowb, tu.data, tu.nbytes);
      L.rgu.type = GGML_Q8_0;
      L.rgu.N = NRP + 2 * Fs;
      L.rgu.K = E;
      L.rgu.role = kRoles - 1;
      L.rgu.raw = upload_padded(GGML_Q8_0, cat.data(), cat.size());
      has_rgu_ = true;
      auto view = [&](int row0, int N, int role) {
        GpuWeight w;
        w.type = GGML_Q8_0;
        w.gemm_qtype = (int)GGML_Q8_0;  // (prompt chunks: decoded just in time, as make_weight does for MoE models)
        w.N = N;
        w.K = E;
        w.role = role;  // (a slot each: kept decoded through a layer-major segment, weight_role)
        stage_need_q_[w.role] = std::max(stage_need_q_[w.role], w8_bytes_q(N, E));
        stage_need_h_[w.role] = std::max(stage_need_h_[w.role], w8_bytes_h(N, E));
        if (L.rgu.raw) {  // (+kRawSlack: the GEMV's block reads run past a row's end, up to BBP + 63 bytes for an odd
                          //  superblock count, as upload_padded allows)
          cl_buffer_region reg{(size_t)row0 * sbr, (size_t)N * sbr + kRawSlack};
          cl_int err;
          w.raw = clCreateSubBuffer(L.rgu.raw, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
          CL_CHECK(err);
          mems_.push_back(w.raw);
        }
        return w;
      };
      L.router = view(0, NR, weight_role(p + "ffn_gate_inp.weight"));
      L.gate_sh = view(NRP, Fs, weight_role(p + "ffn_gate_shexp.weight"));
      L.up_sh = view(NRP + Fs, Fs, weight_role(p + "ffn_up_shexp.weight"));
    } else {
      L.router = make_weight(GGML_Q8_0, q8.data(), q8.size(), NR, E, weight_role(p + "ffn_gate_inp.weight"));
      L.gate_sh = load_w_pad(p + "ffn_gate_shexp.weight", Fs);
      L.up_sh = load_w_pad(p + "ffn_up_shexp.weight", Fs);
    }
    L.down_sh = load_w_pad(p + "ffn_down_shexp.weight", E);
    if (L.gate_sh.N != Fs || L.down_sh.K != Fs) throw std::runtime_error("unexpected shared expert shape");
    const char* names[3] = {"gate", "up", "down"};
    const uint64_t per_expert = (uint64_t)hp.n_ff_exp * E;  // weights of one expert matrix
    if (hp.moe_std) {
      for (int i = 0; i < 3; ++i) {
        const GgufTensor& t = f_.get(p + "ffn_" + names[i] + "_exps.weight");
        if (t.type == GGML_AR4_128 && (!hp.hc || !fn_precise()))  // (iGPUs too; a split: each shard streams its experts)
          throw std::runtime_error("AR4_128 requires precise Flash Next (--precise)");
        const uint64_t K = t.dims.empty() ? 0 : t.dims[0];  // (the shape tested below before anything else of it)
        if (t.dims.size() != 3 || t.dims[2] != (uint64_t)hp.n_expert || t.nelements() != per_expert * hp.n_expert ||
            K != (i == 2 ? (uint64_t)hp.n_ff_exp : (uint64_t)E) ||
            // (every 32-value block type: rows padded with zero blocks as for Gemma 4 26B's (pad_small_block_rows, the
            //  expert file and pin_mtp_experts alike). Flash Next's MTP head files have Q4_0 or Q8_0 experts; Q8_0 was
            //  refused here, "unexpected expert tensor blk.48.ffn_down_exps.weight" (gee456, issue #31, 2026-11-17))
            (K % 256 && !is_per32_type(t.type)))
          throw std::runtime_error("unexpected expert tensor " + t.name);
        if (!is_w8_type(t.type))
          throw std::runtime_error("unsupported expert tensor " + t.name + " (" + ggml_type_name(t.type) + ")");
        if (K % 256 && !stream_) throw std::runtime_error("this model runs with expert streaming only");
        if (!stream_) L.ex_q[i] = upload_padded(t.type, t.data, t.nbytes);
        else stream_bytes_ += t.dims[1] * t.dims[2] * (padded_k((int)K) / 256) * raw_padded_sb_bytes(t.type);
        L.ex_type[i] = t.type;
      }
      if (L.ex_type[0] != L.ex_type[1]) throw std::runtime_error("MoE: gate and up experts of different types");
      return;
    }
    for (int i = 0; i < 3; ++i) {
      const GgufTensor& q = f_.get(p + "ffn_" + names[i] + "_exps.weight");
      const GgufTensor& s = f_.get(p + "ffn_" + names[i] + "_exps.scale");
      if (q.type != GGML_I32 || s.type != GGML_BF16 || q.nbytes != per_expert / 4 * hp.n_expert ||
          s.nbytes != per_expert / 64 * hp.n_expert)
        throw std::runtime_error("unexpected expert tensor " + q.name);
      L.ex_q[i] = buf(q.nbytes, q.data);
      L.ex_s[i] = buf(s.nbytes, s.data);
    }
  }

  // The MTP head's experts (a file of its own, mtp_ext_; QUEUE F34) in one device buffer, each in the expert file's
  // record form (rows padded to whole superblocks, gpu_raw_sb, gate | up | down, 4 KB records: ExpertStore::build),
  // its table row fixed in the expert cache (XCache::pin). Q4_0 head: 512 x ~2.9 MB = ~1.5 GiB, counted in gpu_bytes
  // before the cache is sized, so the cache gives the room.
  // (the record layout of layer il's experts, as ExpertStore::open_or_build lays out a layer)
  ExpertStore::LayerInfo pin_layout(int il, ExpertStore::Src m[3]) const {
    const std::string p = "blk." + std::to_string(il) + ".";
    ExpertStore::LayerInfo li;
    uint64_t sum = 0;
    for (int i = 0; i < 3; ++i) {
      m[i] = ExpertStore::src(f_, p, i);
      li.type[i] = m[i].t->type;
      li.part[i] = m[i].rows * (uint64_t)(padded_k((int)m[i].t->dims[0]) / 256) * raw_padded_sb_bytes(m[i].t->type);
      sum += li.part[i];
    }
    li.rec = (sum + ExpertStore::kAlign - 1) / ExpertStore::kAlign * ExpertStore::kAlign;
    return li;
  }
  size_t pin_bytes(int il) const {  // (a dry load counts them too: --ctx auto)
    ExpertStore::Src m[3];
    return (size_t)hp.n_expert * pin_layout(il, m).rec + 65536;
  }
  void pin_mtp_experts(Layer& L) {
    ExpertStore::Src m[3];
    const ExpertStore::LayerInfo li = pin_layout(L.il, m);
    if ((int)xc_->st.layers.size() <= L.il) xc_->st.layers.resize(L.il + 1);
    xc_->st.layers[L.il] = li;  // (xoff reads the parts' offsets from here; lrec stays 0: the layer is in no pool)
    const int NE = hp.n_expert;
    const size_t bytes = (size_t)NE * li.rec + 65536;  // (+64 KB: the kernels' block reads past a record's end)
    char* base = (char*)zrt::device_alloc(bytes);
    mtp_xbuf_ = base;
    mtp_xbytes_ = bytes;
    gpu_bytes += bytes;
    const int G = 32;  // (experts per batch, built in parallel into pinned host memory, then copied)
    char* host = (char*)zrt::host_alloc((size_t)G * li.rec);
    struct Free { void* p; ~Free() { zrt::mem_free(zrt::C(), p); } } free_host{host};
    for (int e0 = 0; e0 < NE; e0 += G) {
      const int ne = std::min(G, NE - e0);
      memset(host, 0, (size_t)ne * li.rec);
#pragma omp parallel for
      for (int j = 0; j < ne; ++j) {
        uint8_t* dst = (uint8_t*)host + (size_t)j * li.rec;
        for (int i = 0; i < 3; ++i) {
          const GgufTensor& t = *m[i].t;
          const uint32_t ty = li.type[i];
          const size_t sbb = gguf_sb_bytes(ty), bp = raw_padded_sb_bytes(ty), nsb = li.part[i] / bp;
          const int K = (int)t.dims[0], rows = (int)m[i].rows;
          const size_t row1 = (size_t)(e0 + j) * m[i].per + m[i].row0;
          const uint8_t* src;
          std::vector<uint8_t> padded;
          if (K % 256) {
            int blk = 0, bb = 0;
            ggml_block_info(ty, blk, bb);
            padded.resize(nsb * sbb);
            pad_small_block_rows(ty, t.data + row1 * (K / blk) * bb, rows, K, padded.data());
            src = padded.data();
          } else {
            src = t.data + row1 * (K / 256) * sbb;
          }
          for (size_t s = 0; s < nsb; ++s) gpu_raw_sb(ty, src + s * sbb, dst + s * bp);
          dst += li.part[i];
        }
      }
      ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::imm(), base + (size_t)e0 * li.rec, host, (size_t)ne * li.rec, nullptr, 0,
                                             nullptr));
      zrt::sync();
    }
    xc_->pin(L.il, base, li.rec);
    zlog::eprintf("inferzero: MTP head experts on the GPU: %.2f GiB (%s)\n", bytes / 1073741824.0,
            ggml_type_name(li.type[0]));
  }

  // Gemma 4 26B-A4B experts of layer L (see layer_gemma_ffn): router rows padded to router_rows() (zeros past the
  // experts), the norms and scales, the experts (streamed: the expert file; else gate / up taken apart from
  // ffn_gate_up_exps, the down rows padded to whole superblocks)
  void load_moe_gemma(Layer& L, const std::string& p) {
    const int E = hp.n_embd, NR = hp.router_rows(), NE = hp.n_expert, F = hp.n_ff_exp;
    const GgufTensor& r = f_.get(p + "ffn_gate_inp.weight");
    if (r.nelements() != (uint64_t)NE * E) throw std::runtime_error("unexpected Gemma 4 router shape");
    // (f32 rows, read by gemv_f32 with f32 inputs; router_rows() of them, zeros past the experts)
    std::vector<float> v((size_t)NR * E, 0.0f);
    dequant_row(r.type, r.data, NE * E, v.data());
    L.router_f32 = buf(v.size() * 4, v.data());
    L.router.N = NR;
    L.router.K = E;
    {
      const GgufTensor& t = f_.get(p + "ffn_gate_inp.scale");
      if (t.nelements() != (uint64_t)E) throw std::runtime_error("unexpected tensor shape: " + p + "ffn_gate_inp.scale");
      std::vector<float> w(E);
      dequant_row(t.type, t.data, E, w.data());
      for (float& x : w) x /= std::sqrt((float)E);  // (llama.cpp gemma4: rms_norm(x) / sqrt(n_embd) * scale)
      L.rnorm = buf((size_t)E * 4, w.data());
    }
    {
      const GgufTensor& t = f_.get(p + "ffn_down_exps.scale");
      if (t.nelements() != (uint64_t)NE) throw std::runtime_error("unexpected tensor shape: " + p + "ffn_down_exps.scale");
      std::vector<float> w(NE);
      dequant_row(t.type, t.data, NE, w.data());
      L.escale = buf((size_t)NE * 4, w.data());
    }
    L.post_ffw1 = load_f32(p + "post_ffw_norm_1.weight");
    L.ffn_norm2 = load_f32(p + "pre_ffw_norm_2.weight");
    L.post_ffw2 = load_f32(p + "post_ffw_norm_2.weight");
    for (int i = 0; i < 3; ++i) {
      const ExpertStore::Src m = ExpertStore::src(f_, p, i);
      const GgufTensor& t = *m.t;
      const int K = (int)t.dims[0];
      if (t.dims.size() != 3 || t.dims[2] != (uint64_t)NE || m.rows != (uint64_t)(i == 2 ? E : F) || K != (i == 2 ? F : E) ||
          !is_w8_type(t.type))
        throw std::runtime_error("unexpected Gemma 4 expert tensor " + t.name);
      L.ex_type[i] = t.type;
      const size_t rows = (size_t)NE * m.rows, prow = (size_t)padded_k(K) / 256 * gguf_sb_bytes(t.type);
      if (stream_) {
        stream_bytes_ += rows * (padded_k(K) / 256) * raw_padded_sb_bytes(t.type);
        continue;
      }
      if (opt.dry) {
        gpu_bytes += rows * (padded_k(K) / 256) * raw_padded_sb_bytes(t.type) + kRawSlack;
        continue;
      }
      int blk = 0, bytes = 0;
      ggml_block_info(t.type, blk, bytes);
      const size_t rb = (size_t)K / blk * bytes;  // (GGUF row bytes)
      std::vector<uint8_t> all(rows * prow);
      for (int e = 0; e < NE; ++e) {
        const uint8_t* src = t.data + ((size_t)e * m.per + m.row0) * rb;
        uint8_t* dst = all.data() + (size_t)e * m.rows * prow;
        if (K % 256) pad_small_block_rows(t.type, src, (int)m.rows, K, dst);
        else memcpy(dst, src, m.rows * rb);
      }
      L.ex_q[i] = upload_padded(t.type, all.data(), all.size());
    }
    if (L.ex_type[0] != L.ex_type[1]) throw std::runtime_error("MoE: gate and up experts of different types");
  }

  void load_attn(Layer& L, const std::string& p) {
    {
      // (Qwen3 dense: q, k, v as one stack, its output rows [q | k | v] as layer_qwen3_attn lays them out; L.wk / L.wv
      //  stay empty. Qwen3-ASR, one token at 2048 of context: q + k + v 0.166 -> 0.139 ms a layer, gate + up (load_weights)
      //  0.437 -> 0.414; generation 23.8 -> 25.1 tok/s, the 2:34 recording's 507 tokens 19.97 -> 19.56 s (mean of 3); logits
      //  bit-identical)
      // (also float ones of the gated attention (a BF16 GGUF: Xiaomi-OCR-0): rows [q + gate | k | v] as layer_attn reads
      //  them, one GEMV for the three)
      const bool stack = (hp.qwen3 || (float_type(f_.get(p + "attn_q.weight").type) && !getenv("ARC_NO_QKV_FLT"))) &&
                         f_.get(p + "attn_q.weight").type == f_.get(p + "attn_k.weight").type &&
                         f_.get(p + "attn_q.weight").type == f_.get(p + "attn_v.weight").type;
      {
        if (stack) {
          L.wq = load_w_stack({p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"});
        } else {
          L.wq = load_w(p + "attn_q.weight");
          L.wk = load_w(p + "attn_k.weight");
          L.wv = load_w(p + "attn_v.weight");
        }
        L.wo = load_w(p + "attn_output.weight");
        expect(L.wo, hp.n_embd, -1, p + "attn_output.weight");
        expect(L.wq, -1, hp.n_embd, p + "attn_q.weight");
        if (!stack) {
          expect(L.wk, -1, hp.n_embd, p + "attn_k.weight");
          expect(L.wv, -1, hp.n_embd, p + "attn_v.weight");
        }
        L.q_norm = load_f32(p + "attn_q_norm.weight");
        L.k_norm = load_f32(p + "attn_k_norm.weight");
        if (hp.qwen3) {  // (float attention: f32 K / f16 V rows [kv head][ctx][hd], no int8 copy; see q2_init)
          const size_t kvb = (size_t)hp.n_head_kv * ctx * hp.head_dim;
          L.nkv = hp.n_head_kv;
          L.hd = hp.head_dim;
          L.cap = ctx;
          L.i8 = false;
          L.kq = buf(64);
          L.vq = buf(64);
          L.ksc = buf(64);
          L.vsc = buf(64);
          L.kc = buf(kvb * kc_bytes());
          L.vc = buf(kvb * 2);
          fill(L.kc, kvb * kc_bytes());
          fill(L.vc, kvb * 2);
          return;
        }
        const size_t kvb = (size_t)hp.n_head_kv * ctx * hp.head_dim * 2;
        // int8 attention: K/V int8 in 16-key blocks + per-row scales (Qwen: SB = 1, sb_opt); f32 attention: f16 K/V rows
        const size_t kvi = opt.attn_int8 ? kvb : 64, scb = opt.attn_int8 ? (size_t)hp.n_head_kv * ctx * 4 : 64;
        L.kc = buf(opt.attn_int8 ? 64 : kvb);
        L.vc = buf(opt.attn_int8 ? 64 : kvb);
        L.kq = buf(kvi / 2);
        L.vq = buf(kvi / 2);
        L.ksc = buf(scb);
        L.vsc = buf(scb);
        for (cl_mem m : {L.kc, L.vc}) fill(m, opt.attn_int8 ? 64 : kvb);
        for (cl_mem m : {L.kq, L.vq}) fill(m, kvi / 2);
        for (cl_mem m : {L.ksc, L.vsc}) fill(m, scb);
      }
    }
    if ((hp.hc || hp.qsa_own) && L.il < (int)hp.qsa_ratio.size() && hp.qsa_ratio[L.il] > 0) {
      L.qr = hp.qsa_ratio[L.il];
      L.ix = load_q8_concat(p + "indexer.k_proj.weight", p + "indexer.q_proj.weight");
      L.ix_kn = load_f32(p + "indexer.k_norm.weight");
      L.ix_qn = load_f32(p + "indexer.q_norm.weight");
      if (hp.qsa_own) {
        if (!f_.find(p + "indexer.k_norm.pre_weight")) throw std::runtime_error("QSA indexer: no indexer.k_norm.pre_weight");
        L.ix_kp = load_f32(p + "indexer.k_norm.pre_weight");
      }
      if (L.ix.N != (1 + hp.qsa_heads) * hp.qsa_dim) throw std::runtime_error("unexpected QSA indexer shape");
      L.pk = buf((size_t)qsa_blocks(ctx) * hp.qsa_dim * 4 + 64);
      L.ixsv = buf((size_t)(1 + kMaxDrafts) * L.ix.N * 4);
      L.qsa_i = qsa_n_++;
      if (!opt.dry) {  // its tail raw keys inside a recurrent state (see conv_bytes, qsa_host)
        cl_buffer_region r{qsa_tail_off(L.qsa_i), (size_t)(L.qr - 1) * hp.qsa_dim * 4};
        cl_int err;
        L.qtail = clCreateSubBuffer(layers_[qsa_host()].conv_state, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
        CL_CHECK(err);
        mems_.push_back(L.qtail);
      }
    }
  }

