// The expert GEMMs of prompt chunks on XMX straight from the raw expert rows (docs/arc-plan-2026-11-17.md, item X1):
// moe_rg (zero/kernels/decode_moe.cl), the int8 GEMM that decodes each expert row in registers instead of a decode_w8
// pass, with its 64-value chunks' dot products as int8 DPAS (-DRG_XMX) for the types with one scale per 32 values.
// A part of class Engine (zero/engine.h), included inside the class body like zero/xe2/engine_xe2.h: not a header of
// its own.
//
// Why: a Pro B70's Flash Next spent 45 % of its 4K prompt in the per-tile gate / up kernel (DP4a), and a B60's 35B
// prompt had its expert GEMMs at ~21 TOPS against the dense GEMMs' ~50 (issues #4, #27).
//
// ARC_MOE_RG_XMX=1 (opt-in until a card measures it): where decode + GEMM ran (unpadded MoE models such as the 35B:
// chunks of 1,152+ tokens), moe_rg on XMX instead, from rg_min (320) tokens a chunk, when all three of a layer's expert
// types have one scale per 32 values; and every moe_rg that already ran (Flash Next's down projection, Gemma 4 26B,
// split models) takes the XMX form for those types. Flash Next's gate / up stay per tile (moe_rg there is not the
// per-tile kernel's results: KL 0.0181). Where XMX is on (16 lanes, its start-time checks passed). The results are
// moe_rg's bit for bit, and moe_rg's are decode + GEMM's: checked once per expert type at its first use, on the chunk
// itself (the DP4a kernel writes the layer's output, the XMX one a copy, compared byte for byte; any difference, or an
// XMX kernel that does not build, keeps DP4a for that type, with a line; the copy is freed after each check).
// ARC_MOE_RG_XEMU=1: DPAS emulated with DP4a, on any GPU (tests); =2 on an 8-lane GPU a wrong one, which must fail.

  static bool rgx_emu() {
    static const bool e = getenv("ARC_MOE_RG_XEMU") && atoi(getenv("ARC_MOE_RG_XEMU")) != 0;
    return e;
  }
  bool moe_rgx_on() const {
    static const bool on = getenv("ARC_MOE_RG_XMX") && atoi(getenv("ARC_MOE_RG_XMX")) != 0;
    return on && hp.moe_std && !hp.ds41 && (rgx_emu() || (xmx_ && cl_.xmx));
  }
  // (8-lane GPUs, Alchemist and the 140T, since 2026-11-24: moe_rg's 8-lane XMX form, -DRG_SG8 (decode_moe.cl), the same
  //  rule and the same first-use check; for an A770's 35B, whose decode_w8 pass was about as long as its expert GEMMs
  //  (#45). ARC_MOE_RG_XEMU=1 on an 8-lane GPU emulates that form's DPAS)
  // moe_rg's RG_SC32 types: one scale per 32 values (the DPAS sums are its c0 / c1)
  static bool rgx_type(uint32_t t) {
    return t == GGML_IQ4_XS || t == GGML_IQ3_S || t == GGML_IQ3_XXS || t == GGML_IQ2_XXS || is_per32_type(t);
  }
  std::map<uint32_t, int> rgx_state_;  // (per form, rgx_key: 0 not checked yet, 1 the XMX form passed, -1 it differed)
  // (the form: the expert type x 2, + 1 for Inkling 2's up pass, built with -DMOE_SITU (I1.7 step 4), so that variant
  //  gets its own first-use check; keyed by the type alone, the gate pass's check stood for it. The Researcher's review
  //  of 117c938. Other models: one form a type, as before)
  uint32_t rgx_key(const Layer& L, int i) const { return L.ex_type[i] * 2u + (hp.inkling && i == 1 ? 1u : 0u); }
  cl_mem rgx_scratch_ = nullptr;
  size_t rgx_scratch_bytes_ = 0;
  // the layer takes moe_rg for this matrix on XMX (or will check it now)
  bool rgx_part(const Layer& L, int i) const {
    if (!moe_rgx_on() || !rgx_type(L.ex_type[i])) return false;
    const auto it = rgx_state_.find(rgx_key(L, i));
    return it == rgx_state_.end() || it->second >= 0;
  }
  // a whole layer through moe_rg instead of decode + GEMM: all three matrices on XMX (a type on DP4a moe_rg would be
  // slower there than decode + GEMM at 1,280+ tokens: 565 against 595 tok/s at 1,536; audit 2026-11-17); a type whose
  // XMX form failed its check sends the layer back to decode + GEMM
  bool rgx_layer(const Layer& L) const {
    return moe_rgx_on() && moe_rg_layer_ok(L) && rgx_part(L, 0) && rgx_part(L, 1) && rgx_part(L, 2);
  }
  // the XMX kernel, built without ending the process if the driver cannot (cl_.soft_fail): DP4a kept for the type
  cl_kernel rgx_kernel(uint32_t key, const std::string& opts) {
    const uint32_t type = key / 2;
    struct Soft { bool& f; bool was; ~Soft() { f = was; } } soft{cl_.soft_fail, cl_.soft_fail};
    cl_.soft_fail = true;
    try {
      return kernel("decode", opts, "moe_rg");
    } catch (const std::exception& e) {
      rgx_state_[key] = -1;
      zlog::eprintf("inferzero: moe_rg on XMX for %s experts%s does not build (%s): DP4a kept (please report)\n",
                    ggml_type_name(type), key & 1 ? " (situ)" : "", e.what());
      return nullptr;
    }
  }
  std::string rgx_defs(const Layer& L, int i) const {
    if (!rgx_part(L, i)) return std::string();
    return std::string(" -DRG_XMX") + (cl_.sg8 ? " -DRG_SG8" : "") + (cl_.sg8 && moe_dpasw_on() ? " -DRG_DPASW" : "") +
           (rgx_emu() ? " -DRG_XEMU" : "") +
           (cl_.sg8 && rgx_emu() && atoi(getenv("ARC_MOE_RG_XEMU")) == 2 ? " -DRG_XBAD" : "");
  }
  // The first use of a type on XMX: run the DP4a kernel as always into C, the XMX kernel into a copy of C taken before
  // either ran, then compare the two buffers whole (the same tiles, rows and columns written by both). run(k) sets the
  // arguments with the C it is given and launches.
  template <class Run>
  void rgx_first_use(uint32_t key, cl_mem C, size_t cbytes, cl_kernel k_dp4a, cl_kernel k_xmx, const Run& run) {
    const uint32_t type = key / 2;  // (rgx_key: the type x 2, + 1 for Inkling's situ up pass)
    // (the copy is outside the memory plan: on a nearly full card its allocation may fail, which ends the process
    //  unless zrt::oom_throws is set; so it throws here and a failure keeps DP4a for this type, as a failed build
    //  does; freed on every way out, an exception from a run or a read included. A Gemini review, 2026-11-17)
    free_buf(rgx_scratch_);
    rgx_scratch_bytes_ = 0;
    cl_mem sc = nullptr;
    {
      struct Throws { bool was = zrt::oom_throws; Throws() { zrt::oom_throws = true; } ~Throws() { zrt::oom_throws = was; } } t;
      try {
        sc = clCreateBuffer(cl_.ctx, CL_MEM_READ_WRITE, cbytes, nullptr, nullptr);
      } catch (const std::exception&) {
        sc = nullptr;
      }
    }
    if (!sc) {
      rgx_state_[key] = -1;
      zlog::eprintf("inferzero: moe_rg on XMX for %s experts: no room for its check's copy (%.0f MiB): DP4a kept\n",
                    ggml_type_name(type), cbytes / 1048576.0);
      run(k_dp4a, C);
      return;
    }
    mems_.push_back(sc);
    gpu_bytes += cbytes;
    rgx_scratch_ = sc;
    rgx_scratch_bytes_ = cbytes;
    struct Free {
      Engine& e;
      ~Free() {
        e.free_buf(e.rgx_scratch_);
        e.rgx_scratch_bytes_ = 0;
      }
    } free_after{*this};
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, C, rgx_scratch_, 0, 0, cbytes, 0, nullptr, nullptr));
    run(k_dp4a, C);
    run(k_xmx, rgx_scratch_);
    std::vector<uint8_t> a(cbytes), b(cbytes);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, C, CL_TRUE, 0, cbytes, a.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, rgx_scratch_, CL_TRUE, 0, cbytes, b.data(), 0, nullptr, nullptr));
    const bool same = a == b;
    rgx_state_[key] = same ? 1 : -1;
    // (the copy only lives for a check: up to slots x max(n_embd, Fp) floats, 128 MiB for the 35B at 2,048-token chunks,
    //  outside the memory plan; freed at once by free_after, audit 2026-11-17)
    static std::once_flag said;
    const char* form = !cl_.sg8 ? "" : moe_dpasw_on() ? ", 8 lanes, the split DPAS: ARC_MOE_DPASW=1" : ", 8 lanes";
    std::call_once(said, [form] { zlog::eprintf("inferzero: expert GEMMs of prompt chunks on XMX from the raw rows (ARC_MOE_RG_XMX=1%s)\n", form); });
    zlog::eprintf("inferzero: moe_rg on XMX for %s experts%s: %s\n", ggml_type_name(type), key & 1 ? " (situ)" : "",
                  same ? "the same results as DP4a bit for bit" : "DIFFERS from DP4a: DP4a kept for this type (please report)");
  }
