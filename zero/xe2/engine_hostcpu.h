// Part of the Engine class body (included from zero/engine.h): the one-token GEMV of the weights kept in host memory,
// done by the CPU (QUEUE "cards too small for the whole model", 2026-11-19, opt-in: ARC_HOST_CPU=1).
//
// A card too small for the whole model (an Arc B580 12 GB and the 27B: 0.5-2.2 GiB of its weights in RAM,
// plan_tight) has its GPU read those matrices over PCIe every token: d-dorok's B580 (PCIe 3.0) 7.1 tok/s with 0.49 GiB
// in RAM, where llama.cpp, which runs the layers it cannot fit on the CPU, makes 11.4 (issue #37). Here the CPU
// computes those matrices from RAM, and only the small input and result vectors cross the bus: a host wait per
// matrix, no weight read over PCIe.
//
// The CPU's form of such a matrix, made at load: its rows decoded (src/quants.h dequant_row) and requantized to int8
// with one f32 scale per 32 values (as Q8_0), the blocks the activations have too (quant_store16 with act32: int8 and
// one scale per 32). The dot product of a block in int16 / int32 (AVX2: maddubs on |a| and w signed by a), its two
// scales once per block, the sum over blocks in f32. One kernel for every weight type, read at ~1.1 bytes a weight
// from RAM (a 3-bit matrix's 0.43 over PCIe). Not bit-identical to the GPU's sums (int8 weights here, the GPU's own
// decode there, another order): KL-checked. RAM: ~1.1 bytes a weight of those matrices more, while the switch is on.

struct HostMat {
  std::vector<int8_t> q;   // (N x K int8)
  std::vector<float> d;    // (N x K / 32 scales)
  int N = 0, K = 0;
};
std::unordered_map<cl_mem, HostMat> host_mat_;  // (zero/engine_load.h upload_padded: a matrix put in host memory)
// ARC_HOST_CPU_NATIVE=1 (2026-11-23, opt-in; the A770 at 92K, #45: the CPU's rows ~54 ms of a ~100 ms step): an IQ3_S
// matrix kept as the GGUF's own blocks (110 bytes a superblock of 256: 0.43 bytes a weight) and decoded in the dot
// loop, in place of the int8 copy (1.125 bytes a weight, read from RAM every token, and that much RAM more). The dot
// is exact in integers inside a block of 32 (the grid's values 1..15 with their signs, times the int8 activations),
// then (d x the block's scale) x the activation's scale x that sum in f32: no requantization of the weights, so its
// sums differ from the int8 copy's (KL-checked, not bit for bit). Other types keep the int8 copy
struct HostRaw {
  std::vector<uint8_t> b;  // (the superblocks in order, rows of K / 256 each)
  uint32_t type = 0;
  size_t nsb = 0;
};
std::unordered_map<cl_mem, HostRaw> host_raw_;
static bool host_native_on() {
  static const bool e = getenv("ARC_HOST_CPU_NATIVE") && atoi(getenv("ARC_HOST_CPU_NATIVE")) != 0;
  return e;
}
uint64_t hcpu_native_n_ = 0;
std::vector<int8_t> hcpu_q_;
std::vector<float> hcpu_y_, hcpu_sc_;
uint64_t hcpu_n_ = 0;
double hcpu_ms_ = 0, hcpu_last_ms_ = 0;
// (the input copy's shape and the zrt::gpu_writes count after the last host matrix: host_cpu_gemv's one read per input)
uint64_t hcpu_seen_ = ~0ull, hcpu_reused_ = 0;
int hcpu_inK_ = -1, hcpu_inM_ = -1;

// (on by default on discrete cards since 2026-11-19: d-dorok's B580 (PCIe 3.0, i9-9900KF), the 27B, two rounds each:
//  0.49 GiB in RAM (ARC_HOST_KEEP_GB=0.75) 7.11 -> 13.70 tok/s (a step 140.6 -> 72.6 ms), the default 0.85 GiB 4.54 ->
//  11.88, 1.02 GiB at a 4K context 3.90 -> 10.27; the prompt the same or faster (836 -> 1,022 tok/s at 512); issue #37.
//  llama.cpp SYCL on that card: 11.4. Off on iGPUs, whose GPU reads RAM at full speed (the laptop's 27B with 0.5 GiB
//  forced there 5.03 -> 4.22). ARC_HOST_CPU=0 / 1 decides)
bool host_cpu_on() const {
  static const int e = getenv("ARC_HOST_CPU") ? (atoi(getenv("ARC_HOST_CPU")) != 0 ? 1 : 0) : -1;
  return e < 0 ? !zrt::integrated() : e == 1;
}
// (upload_padded: a matrix it put in host memory; its rows are `data` (nbytes of GGUF `type`), valid during the call)
void host_src_note(cl_mem m, uint32_t type, const uint8_t* data, size_t nbytes, size_t nsb) {
  if (!host_cpu_on() || !m || opt.dry) return;
  // (every value in order, N x K: the rows are cut by the matmul's N and K later; nsb superblocks of 256 values,
  //  nbytes / nsb bytes each, as upload_padded counts them (gguf_sb_bytes))
  const size_t vals = nsb * 256;
  if (!vals || nbytes % nsb) return;
  if (host_native_on() && type == GGML_IQ3_S && nbytes / nsb == 110) {
    HostRaw r;
    r.b.assign(data, data + nbytes);
    r.type = type;
    r.nsb = nsb;
    if (host_raw_.empty())
      LOG_INF("ARC_HOST_CPU_NATIVE=1: the CPU's IQ3_S matrices in host memory as their own blocks, decoded in its dot "
              "loop (no int8 copy)\n");
    host_raw_[m] = std::move(r);
    return;
  }
  HostMat h;
  h.q.resize(vals);
  h.d.resize(vals / 32);
  const int64_t nch = (int64_t)(vals / 4096 + (vals % 4096 ? 1 : 0));
  const size_t bpv = nbytes / nsb;  // (bytes a superblock of 256)
#pragma omp parallel
  {
    std::vector<float> f(4096 + 256);
#pragma omp for schedule(static)
    for (int64_t c = 0; c < nch; ++c) {
      const size_t v0 = (size_t)c * 4096, n = std::min((size_t)4096, vals - v0);
      // (4096 is a whole number of superblocks of 256 and of blocks of 32)
      dequant_row(type, data + v0 / 256 * bpv, (int)n, f.data());
      for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(f[b * 32 + i]));
        const float d = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        h.d[(v0 >> 5) + b] = d;
        for (int i = 0; i < 32; ++i) h.q[v0 + b * 32 + i] = (int8_t)std::lrintf(f[b * 32 + i] * inv);
      }
    }
  }
  host_mat_[m] = std::move(h);
}
// (one row's dot with the int8 activations: sum over blocks of wd * ad * sum(w * a))
static float host_dot(const int8_t* w, const float* wd, const int8_t* a, const float* ad, int nb) {
#ifdef __AVX2__
  __m256 acc = _mm256_setzero_ps();
  const __m256i ones = _mm256_set1_epi16(1);
  for (int b = 0; b < nb; ++b) {
    const __m256i wv = _mm256_loadu_si256((const __m256i*)(w + 32 * b));
    const __m256i av = _mm256_loadu_si256((const __m256i*)(a + 32 * b));
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(av, av), _mm256_sign_epi8(wv, av));
    const __m256 p = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(wd[b] * ad[b]), p, acc);
  }
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
#elif defined(__SSSE3__)
  // (Ivy Bridge / Sandy Bridge, older AMD: the same arithmetic in 128-bit halves, build.sh -march=native, issue #50)
  __m128 acc = _mm_setzero_ps();
  const __m128i ones = _mm_set1_epi16(1);
  for (int b = 0; b < nb; ++b) {
    __m128i t = _mm_setzero_si128();
    for (int h = 0; h < 2; ++h) {
      const __m128i wv = _mm_loadu_si128((const __m128i*)(w + 32 * b + 16 * h));
      const __m128i av = _mm_loadu_si128((const __m128i*)(a + 32 * b + 16 * h));
      t = _mm_add_epi32(t, _mm_madd_epi16(_mm_maddubs_epi16(_mm_sign_epi8(av, av), _mm_sign_epi8(wv, av)), ones));
    }
    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(wd[b] * ad[b]), _mm_cvtepi32_ps(t)));
  }
  const __m128 s2 = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
#else
  float s = 0.0f;
  for (int b = 0; b < nb; ++b) {
    int t = 0;
    for (int i = 0; i < 32; ++i) t += (int)w[32 * b + i] * (int)a[32 * b + i];
    s += wd[b] * ad[b] * (float)t;
  }
  return s;
#endif
}
// (one IQ3_S row's dot with the int8 activations, ARC_HOST_CPU_NATIVE: ns superblocks of 110 bytes (d, qs 64, qh 8,
//  signs 32, scales 4; src/quants.h decode_sb's layout). Block ib of 32 values: grid entry k (values 4k..4k+3) is
//  iq3s_grid[qs[8 ib + k] | ((qh[ib] << (8 - k)) & 256)], value 8l + j negative where bit j of signs[4 ib + l] is set,
//  the block's scale 1 + 2 x its nibble)
static float host_dot_iq3s(const uint8_t* row, int ns, const int8_t* a, const float* ad) {
#ifdef __AVX2__
  __m256 acc = _mm256_setzero_ps();
  const __m256i ones = _mm256_set1_epi16(1), one8 = _mm256_set1_epi8(1);
  // (byte i of the 32 takes sign byte i / 8, bit i % 8)
  const __m256i sh = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                      3, 3, 3, 3);
  const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull);
  for (int s = 0; s < ns; ++s) {
    const uint8_t* p = row + 110 * (size_t)s;
    const float d = fp16_to_f32(rd16(p));
    const uint8_t *qs = p + 2, *qh = p + 66, *sg = p + 74, *sc = p + 106;
    for (int ib = 0; ib < 8; ++ib) {
      const uint32_t h = qh[ib];
      const uint8_t* q = qs + 8 * ib;
      const __m256i g = _mm256_setr_epi32(
          (int)iq3s_grid[q[0] | ((h << 8) & 256)], (int)iq3s_grid[q[1] | ((h << 7) & 256)],
          (int)iq3s_grid[q[2] | ((h << 6) & 256)], (int)iq3s_grid[q[3] | ((h << 5) & 256)],
          (int)iq3s_grid[q[4] | ((h << 4) & 256)], (int)iq3s_grid[q[5] | ((h << 3) & 256)],
          (int)iq3s_grid[q[6] | ((h << 2) & 256)], (int)iq3s_grid[q[7] | ((h << 1) & 256)]);
      const __m256i sb = _mm256_shuffle_epi8(_mm256_set1_epi32((int)rd32(sg + 4 * ib)), sh);
      const __m256i neg = _mm256_cmpeq_epi8(_mm256_and_si256(sb, bit), bit);  // (0xFF: negative)
      const __m256i av = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(a + 32 * (8 * s + ib))),
                                          _mm256_or_si256(neg, one8));
      const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(g, av), ones);
      const int bs = 1 + 2 * ((sc[ib / 2] >> (4 * (ib % 2))) & 0xF);
      acc = _mm256_fmadd_ps(_mm256_set1_ps(d * (float)bs * ad[8 * s + ib]), _mm256_cvtepi32_ps(p32), acc);
    }
  }
  const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
  return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
#else
  float r = 0.0f;
  for (int s = 0; s < ns; ++s) {
    const uint8_t* p = row + 110 * (size_t)s;
    const float d = fp16_to_f32(rd16(p));
    const uint8_t *qs = p + 2, *qh = p + 66, *sg = p + 74, *sc = p + 106;
    for (int ib = 0; ib < 8; ++ib) {
      int t = 0;
      for (int k = 0; k < 8; ++k) {
        const uint8_t* g = (const uint8_t*)(iq3s_grid + (qs[8 * ib + k] | ((qh[ib] << (8 - k)) & 256)));
        for (int j = 0; j < 4; ++j) {
          const int v = 4 * k + j, w = (sg[4 * ib + v / 8] >> (v % 8)) & 1 ? -(int)g[j] : (int)g[j];
          t += w * (int)a[32 * (8 * s + ib) + v];
        }
      }
      const int bs = 1 + 2 * ((sc[ib / 2] >> (4 * (ib % 2))) & 0xF);
      r += d * (float)bs * ad[8 * s + ib] * (float)t;
    }
  }
  return r;
#endif
}
// (matmul1, one token: true when the CPU did it; C[c_off + n] (+)= sum_k W[n][k] * a[k])
bool host_cpu_gemv(const GpuWeight& w, int M, cl_mem C, int c_off, int acc) {
  // (with --prof / --prof-gen too since 2026-11-23: a profile without it timed the GPU reading these matrices over
  //  PCIe, a path the plain run does not take (skipzx-lab's A770 at 92K, #45: the GEMVs 2-4x slower in the profile).
  //  Its time is the profile's row "<matrix> (CPU)", matmul1)
  if (!host_cpu_on() || acc > 1 || q2mm_) return false;
  const auto it = host_mat_.find(w.raw);
  const auto itr = it == host_mat_.end() ? host_raw_.find(w.raw) : host_raw_.end();
  if (it == host_mat_.end() && itr == host_raw_.end()) return false;
  const HostMat* hm = it != host_mat_.end() ? &it->second : nullptr;
  const HostRaw* hr = hm ? nullptr : &itr->second;
  const int N = w.N, K = w.K, nb = K / 32;
  if (N <= 0 || K % 32) return false;
  if (hm && hm->q.size() != (size_t)N * K) return false;
  if (hr && (K % 256 || hr->nsb != (size_t)N * (K / 256))) return false;
  hcpu_y_.resize(N);
  // (the activations: int8 values, then one scale per 32 values at stride M (quant_store16, act32); the blocking
  //  read last, so the queue has run up to here. One read per input since 2026-11-23: where nothing that can write a
  //  GPU buffer was enqueued since the last read of this shape (zrt::gpu_writes), the copy is the same input. The
  //  matrices of one input in host memory one after another (gate and up; qkv, z and ab) read it once: skipzx-lab's
  //  A770 at 92K, #45, ~1.4 ms a matrix of fixed cost, the read and the queue's drain part of it)
  const bool same_in = hcpu_seen_ == zrt::gpu_writes.load(std::memory_order_relaxed) && hcpu_inK_ == K && hcpu_inM_ == M;
  if (!same_in) {
    hcpu_q_.resize(K);
    hcpu_sc_.resize((size_t)nb * M);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, aq_, CL_FALSE, 0, (size_t)K, hcpu_q_.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, asc_, CL_TRUE, 0, hcpu_sc_.size() * 4, hcpu_sc_.data(), 0, nullptr, nullptr));
    hcpu_inK_ = K, hcpu_inM_ = M;
  } else {
    ++hcpu_reused_;
  }
  if (acc) CL_CHECK(clEnqueueReadBuffer(cl_.queue, C, CL_TRUE, (size_t)c_off * 4, (size_t)N * 4, hcpu_y_.data(), 0, nullptr, nullptr));
  // (timed from here: the blocking reads above also waited for the GPU's earlier work, which the profile counts
  //  under its own kernels)
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> ad(nb);
  for (int b = 0; b < nb; ++b) ad[b] = hcpu_sc_[(size_t)b * M];
  float* y = hcpu_y_.data();
  const int8_t* a = hcpu_q_.data();
  // (the threads of this loop: the CPU's physical cores since 2026-11-23, not every logical processor: skipzx-lab's
  //  Xeon W-2155 (10 cores, 20 threads), the 27B at 92K, #45: 9.96 / 9.66 tok/s with OpenMP's 20, 11.43 with 10.
  //  ARC_HOST_CPU_THREADS=N sets it; 0 or a count that cannot be read: OpenMP's default)
  static const int nthr = getenv("ARC_HOST_CPU_THREADS") ? std::max(0, atoi(getenv("ARC_HOST_CPU_THREADS")))
                                                         : plat::physical_cores();
  const int nsr = K / 256;
  auto row = [&](int n) {
    const float s = hm ? host_dot(hm->q.data() + (size_t)n * K, hm->d.data() + (size_t)n * nb, a, ad.data(), nb)
                       : host_dot_iq3s(hr->b.data() + (size_t)n * nsr * 110, nsr, a, ad.data());
    y[n] = acc ? y[n] + s : s;
  };
  if (hr) ++hcpu_native_n_;
  if (nthr > 0) {
#pragma omp parallel for schedule(static) num_threads(nthr)
    for (int n = 0; n < N; ++n) row(n);
  } else {
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) row(n);
  }
  CL_CHECK(clEnqueueWriteBuffer(cl_.queue, C, CL_TRUE, (size_t)c_off * 4, (size_t)N * 4, y, 0, nullptr, nullptr));
  // (after our own write, which counts: the next host matrix of this input finds the count unchanged. C is never the
  //  input buffer aq_ / asc_, so this write leaves the copy valid)
  hcpu_seen_ = zrt::gpu_writes.load(std::memory_order_relaxed);
  ++hcpu_n_;
  hcpu_last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  hcpu_ms_ += hcpu_last_ms_;
  return true;
}
