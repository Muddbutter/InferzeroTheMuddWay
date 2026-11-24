// One model over several GPUs by layers (--split layer; QUEUE "a layer split beside the tensor split", L1 + L2;
// docs/tiered-placement.md P2). Each GPU is a stage: an Engine over a range of the layers (EngineOptions pp_begin /
// pp_end), with those layers' weights, KV caches and DeltaNet states only; the first stage embeds, the last has the
// norm, the head and the MTP block. A token's rows go through the stages in turn: stage r's rows are copied to stage
// r + 1 (through host memory: the first form, L2's GPU-to-GPU copies later). The stages run the same kernels on the
// same rows in the same order as one GPU does, so the logits are the same bit for bit (ARC_TP_SIM=1: every stage on GPU
// 0, the check).
// Where the stages are: the last one (the head, the logits) on GPU 0, the others on GPUs 1, 2, ... in layer order, so
// the server, which drives one Engine on its threads' default device (GPU 0), drives the last stage (serve()); its
// calls run on the earlier stages first through the mirror (zero/mgpu/tp_mirror.h, pp_mirror_start / finish below).
// --tensor-split a,b,...: GPU g's share of the layers (GPU 0's is the last layers').
// What it is for: a model that a card cannot hold alone without paging, cards of different kinds, and links too slow
// for the tensor split's sums per layer. What it is not: faster for one request than one card that holds the model,
// since a token visits every stage in turn (the tensor split is the fast form on well-linked matching cards).
// Not yet (QUEUE): MTP over the stages, saved prompts and sessions in the server, the prompts pipelined over the
// stages (L3), two requests in flight (L4).
#pragma once
#include <memory>
#include <string>
#include <vector>

class PpEngine {
 public:
  // the GPU of stage r of n: the last stage on GPU 0, the others on 1, 2, ...
  static int gpu_of(int r, int n) { return (r + 1) % n; }
  // the layers' cut over n stages: stage r has the layers [cut[r], cut[r + 1]); shares by GPU (empty: equal)
  static std::vector<int> cut(const std::string& path, int n, const std::vector<double>& shares) {
    if (n < 2 || n > 64) throw std::runtime_error("--split layer: 2 to 64 GPUs");
    int NL = 0;
    {
      GgufFile g(path);
      const std::string a = g.get_str("general.architecture");
      NL = (int)g.get_int(a + ".block_count", 0) - (int)g.get_int(a + ".nextn_predict_layers", 0);  // (not the MTP layer)
    }
    if (NL < n) throw std::runtime_error("--split layer: " + std::to_string(NL) + " layers for " + std::to_string(n) + " GPUs");
    std::vector<double> s = shares;
    if (s.empty()) s.assign(n, 1.0);
    double sum = 0;
    for (double x : s) sum += std::max(0.0, x);
    // (a typo is said, not quietly made equal shares; no share > 0: no cut. The two said apart: 3 shares for 2 GPUs
    //  read "at least one above 0", MaineTim, issue #44)
    if ((int)s.size() != n)
      throw std::runtime_error("--tensor-split: " + std::to_string(s.size()) + " shares for " + std::to_string(n) +
                               " GPUs: give one a GPU");
    if (!(sum > 0)) throw std::runtime_error("--tensor-split: at least one share above 0");
    std::vector<int> c(n + 1, 0);
    double acc = 0;
    for (int r = 0; r < n; ++r) {  // (each stage at least one layer)
      acc += std::max(0.0, s[gpu_of(r, n)]);
      c[r + 1] = r + 1 == n ? NL : std::clamp((int)(NL * acc / sum + 0.5), c[r] + 1, NL - (n - 1 - r));
    }
    return c;
  }
  // a stage's options: its layers, the MTP block only on the last, the NPU off (its table is by the global layer
  // number, sized by layers_: a stage not from layer 0 would read past it)
  static EngineOptions stage_options(const EngineOptions& o, const std::vector<int>& c, int r) {
    EngineOptions so = o;
    so.pp_begin = c[r];
    so.pp_end = c[r + 1];
    so.pp_size = (int)c.size() - 1;
    if (r + 2 < (int)c.size()) so.mtp = false;
    // (said once: the stages do not drive MTP or a drafter yet, so a layer split decodes plainly; #92)
    if (o.mtp || !o.dflash.empty()) {
      static std::once_flag said;
      std::call_once(said, [] {
        zlog::eprintf("inferzero: MTP / drafting is not used on a split model yet (--gpus 2 or more): plain decoding\n");
      });
    }
    so.npu = 0;
    return so;
  }
  static bool sim() { return getenv("ARC_TP_SIM") && atoi(getenv("ARC_TP_SIM")) != 0; }
  // the GPUs opened (the runtime started first): before stage engines or their memory probes
  static void open(int n) {
    ClRt rt;
    if (!rt.init()) throw std::runtime_error("cannot start Level Zero");
    std::string err;
    if (ClRt::open_devices(n, sim(), err, true) != n) throw std::runtime_error("--gpus " + std::to_string(n) + ": " + err);
  }

  // host_gb: stage r's weights in host memory (EngineOptions::host_weights_gb; the server's --ctx auto probes; empty or
  // < 0: the options')
  PpEngine(const std::string& path, EngineOptions o, int n, const std::vector<double>& shares = {},
           const std::vector<double>& host_gb = {}) {
    open(n);
    const std::vector<int> c = cut(path, n, shares);
    zlog::eprintf("inferzero: %d GPUs by layers%s:", n, sim() ? " (ARC_TP_SIM: all on GPU 0)" : "");
    for (int r = 0; r < n; ++r) zlog::eprintf(" GPU %d layers %d-%d", gpu_of(r, n), c[r], c[r + 1] - 1);
    zlog::eprintf("\n");
    zrt::Dev* const saved = zrt::cur;
    struct Back { zrt::Dev* s; ~Back() { zrt::cur = s; } } back{saved};
    // (a MoE model's RAM tier, one budget for all the stages, measured once before any of them pins memory: the free
    //  RAM beyond 16 GiB halved, as a tensor split's, at most 32 GiB a stage and 128 GiB in all, shared by each stage's
    //  share of the layers. Each stage took the one-GPU rule before, the free RAM less 16 GiB, and the stages are
    //  built one after another, so the first could pin nearly all of it (QUEUE row: a layer split's memory plan, B.2).
    //  ARC_XRAM_GB or --expert-ram still give each stage that much; a dense model has no tier)
    std::vector<double> ram_gb(n, -1.0);
    if (o.expert_ram_gb < 0 && !getenv("ARC_XRAM_GB") && !zrt::integrated()) {
      const double GiB = 1073741824.0, keep = 16 * GiB, avail = (double)plat::ram_available();
      double total = avail > keep ? std::min((avail - keep) / 2, 128 * GiB) : 0.0;
      // (Windows: what it lets the GPUs map of RAM (the non-local budget, less what is in use and 4 GiB), measured once
      //  here for all the stages, not by each against the whole: together they could pass it; the audit of 2026-11-23)
      uint64_t nb = 0, nu = 0;
      if (zrt::video_budget(nb, nu, true)) {
        const double room = nb > nu + (4ull << 30) ? (double)(nb - nu - (4ull << 30)) : 0.0;
        total = std::min(total, room);
      }
      const int nl = std::max(1, c[n] - c[0]);
      for (int r = 0; r < n; ++r)
        ram_gb[r] = std::min(32 * GiB, total * (double)(c[r + 1] - c[r]) / nl) / GiB;
    }
    try {
      for (int r = 0; r < n; ++r) {
        zrt::cur = zrt::devs[gpu_of(r, n)];
        EngineOptions so = stage_options(o, c, r);
        if (r < (int)host_gb.size() && host_gb[r] >= 0 && so.host_weights_gb < 0) so.host_weights_gb = host_gb[r];
        if (ram_gb[r] >= 0) so.expert_ram_gb = ram_gb[r], so.expert_ram_auto = true;  // (0: no tier; automatic, not hand-set)
        stages_.push_back(std::make_unique<Engine>(path, so));
      }
    } catch (...) {
      // (a stage that failed to load: the ones built freed each on its own GPU, as ~PpEngine does, before Back puts
      //  GPU 0 back. Freed after it, through GPU 0's context, a GPU of another driver's memory; audit 2026-11-23 M6)
      for (int r = (int)stages_.size() - 1; r >= 0; --r) {
        zrt::cur = zrt::devs[gpu_of(r, n)];
        stages_[r].reset();
      }
      stages_.clear();
      throw;
    }
    rows_.resize(n);
  }
  ~PpEngine() {  // (each stage freed on its own GPU)
    zrt::Dev* const saved = zrt::cur;
    for (int r = stages() - 1; r >= 0; --r) {
      zrt::cur = zrt::devs[gpu_of(r, stages())];
      stages_[r].reset();
    }
    zrt::cur = saved;
  }
  int stages() const { return (int)stages_.size(); }
  Engine& stage(int r) { return *stages_[r]; }
  Engine& last() { return *stages_.back(); }
  // the tokens through every stage; the last stage's logits (as Engine::forward). runz's path; the server's goes
  // through the mirror
  const std::vector<float>& forward(const int* tokens, int n) {
    zrt::Dev* const saved = zrt::cur;
    struct Back { zrt::Dev* s; ~Back() { zrt::cur = s; } } back{saved};
    const int S = stages();
    // (a stage that throws, a context overflow or a GPU fault, leaves the earlier stages a call ahead: all of them
    //  are reset, so the next call does not mix positions)
    try {
      for (int r = 0; r < S; ++r) {
        zrt::cur = zrt::devs[gpu_of(r, S)];
        Engine& e = *stages_[r];
        e.pp_in_ = r ? rows_[r - 1].data() : nullptr;
        e.pp_out_ = r + 1 < S ? &rows_[r] : nullptr;
        const std::vector<float>& l = e.forward(tokens, n);
        if (r + 1 == S) return l;
      }
    } catch (...) {
      try { reset(); } catch (const std::exception& ex_) { LOG_WRN("inferzero: resetting the layer split after a failed pass failed: %s\n", ex_.what()); } catch (...) { LOG_WRN("inferzero: resetting the layer split after a failed pass failed\n"); }
      throw;
    }
    static const std::vector<float> none;
    return none;
  }
  void reset() {
    zrt::Dev* const saved = zrt::cur;
    for (int r = 0; r < stages(); ++r) {
      zrt::cur = zrt::devs[gpu_of(r, stages())];
      stages_[r]->reset();
    }
    zrt::cur = saved;
  }
  size_t gpu_bytes(int r) const { return stages_[r]->gpu_bytes; }
  // serverz --split layer: the last stage becomes the server's engine (on GPU 0), its state-changing calls run on
  // the earlier stages first (zero/mgpu/tp_mirror.h); call before the first request
  void serve() { stages_.back()->pp_front_ = this; }
  // a stage that failed after an earlier one had run the call: the stages are out of step until the model is
  // loaded again (/health says so, as a broken tensor split)
  bool broken() const { return broken_; }

 private:
  friend void pp_mirror_start(Engine& last, const std::function<void(Engine&)>& f);
  friend void pp_mirror_finish(Engine& last, bool threw);
  std::vector<std::unique_ptr<Engine>> stages_;
  std::vector<std::vector<float>> rows_;  // (stage r's output rows, the next stage's input)
  bool broken_ = false;
  std::function<bool()> cancel_saved_;  // (the last stage's cancel, off during a mirrored call)
};

// The mirror's layer-split half (TpMirror::start / finish, zero/mgpu/engine_tp.h): the call on stages 0 .. S-2 in
// turn on this thread, each on its GPU with the rows of the one before, then the last stage's own call reads the
// rows of the stage before it. A cancel stops between the server's calls only (a call is at most a chunk of the
// prompt): one stopped inside a call would leave the earlier stages ahead. A stage failing before any other ran the
// call leaves all in step (a context overflow, a bad slot: the first stage says it); after one did, the split is
// broken until it is loaded again.
inline void pp_mirror_start(Engine& last, const std::function<void(Engine&)>& f) {
  PpEngine& p = *last.pp_front_;
  if (p.broken_) throw std::runtime_error("a GPU's stage failed earlier: restart (or load the model again)");
  const int S = p.stages();
  zrt::Dev* const saved = zrt::cur;
  struct Back { zrt::Dev* s; ~Back() { zrt::cur = s; } } back{saved};
  for (int r = 0; r + 1 < S; ++r) {
    zrt::cur = zrt::devs[PpEngine::gpu_of(r, S)];
    Engine& s = *p.stages_[r];
    s.rope_shift = last.rope_shift;  // (the fields the server sets on its engine before a call)
    s.profile = last.profile;
    s.cancel = nullptr;
    s.pp_in_ = r ? p.rows_[r - 1].data() : nullptr;
    s.pp_out_ = &p.rows_[r];
    try {
      f(s);
    } catch (const std::exception& e) {
      if (r > 0) {
        p.broken_ = true;
        zlog::eprintf("multi-GPU: GPU %d (layers from %d): %s\n", PpEngine::gpu_of(r, S), s.opt.pp_begin, e.what());
      }
      throw;
    } catch (...) {
      if (r > 0) p.broken_ = true;
      throw;
    }
  }
  last.pp_in_ = p.rows_[S - 2].data();
  last.pp_out_ = nullptr;
  p.cancel_saved_ = std::move(last.cancel);
  last.cancel = nullptr;
}
inline void pp_mirror_finish(Engine& last, bool threw) {
  PpEngine& p = *last.pp_front_;
  last.cancel = std::move(p.cancel_saved_);
  p.cancel_saved_ = nullptr;
  if (threw) {
    p.broken_ = true;  // (the earlier stages ran the whole call)
    zlog::eprintf("multi-GPU: the last stage (GPU 0) failed after the others ran the call: restart the server\n");
  }
}
