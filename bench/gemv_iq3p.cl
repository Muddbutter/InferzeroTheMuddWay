
// ---- PROTOTYPE (gemv_bench, GB_REPACK=1): IQ3_S GEMV over a planar repack of each row. Per pair of superblocks
// (the 16 lanes' 32-blocks; lane l: superblock l / 8, 32-block ib = l % 8) the row holds, plane by plane over the
// row's T pairs: qs (128 B a pair: lane l's 8 bytes as words l and 16 + l), signs (64 B: lane l's 4 bytes as word l),
// qh (16 B: lane l's byte at l), meta (16 B: bytes 0-3 / 4-7 the two superblocks' scales, 8-9 / 10-11 their d).
// Row size T * 224 = nsb * 112 (BBP), as before. Each lane gets its bytes by block reads straight into registers: no
// local-memory window, no barriers; the decode is decode32's IQ3_S expression on the same bytes, the dot products and
// the epilogue gemv's: the same results bit for bit.
#if QT == IQ3_S && defined(GRID_OFF) && !defined(Q2F) && !defined(GU2)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_iq3p(__global const uchar* raw, int N, int K, __global const uint* tabs, __global const char* aq,
                        __global const float* asc, __global const ushort* absum, int M, __global float* C, int c_off,
                        int accumulate) {
  __local uint lgrid[GRID_N];
  for (int i = get_local_id(0); i < GRID_N; i += get_local_size(0)) lgrid[i] = tabs[GRID_OFF + i];
  const uint* grid = lgrid;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), sbl = lane / 8, ib = lane % 8;
  const int r0 = (get_global_id(0) / 16) * GROWS;
  const int nsb = K / 256, T = (nsb + 1) / 2;
  float acc[GROWS];
#pragma unroll
  for (int r = 0; r < GROWS; ++r) acc[r] = 0.0f;
  for (int sp = 0; sp < nsb; sp += 2) {
    const int tp = sp / 2;
    const int b = sp * 8 + lane;  // this lane's 32-block within the row
    const bool has = sp + lane / 8 < nsb;
    const uint8 a8 = has ? vload8(0, (__global const uint*)(aq + 32 * b)) : (uint8)0;
    const float sa = has ? asc[(size_t)b * M] : 0.0f;
#pragma unroll
    for (int r = 0; r < GROWS; ++r) {
      const int n = min(r0 + r, N - 1);
      __global const uchar* row = raw + (size_t)n * nsb * BBP;
#ifdef IQ3P_TILE
      // (the planes inside the pair's own 224 bytes, where the pair was: superblock pairs keep their place in the row)
      __global const uchar* pr = row + tp * 224;
      const uint2 q = intel_sub_group_block_read2((__global const uint*)pr);
      const uint s = intel_sub_group_block_read((__global const uint*)(pr + 128));
      const uint h = intel_sub_group_block_read_uc(pr + 192);
      const uint mb = intel_sub_group_block_read_uc(pr + 208);
#else
      const uint2 q = intel_sub_group_block_read2((__global const uint*)(row + tp * 128));
      const uint s = intel_sub_group_block_read((__global const uint*)(row + T * 128 + tp * 64));
      const uint h = intel_sub_group_block_read_uc(row + T * 192 + tp * 16);
      const uint mb = intel_sub_group_block_read_uc(row + T * 208 + tp * 16);
#endif
      const uint scb = sub_group_shuffle(mb, sbl * 4 + ib / 2);
      const uint dl = sub_group_shuffle(mb, 8 + 2 * sbl), dh = sub_group_shuffle(mb, 9 + 2 * sbl);
      const float d = f16(dl | (dh << 8));
      const int sc0 = 1 + 2 * ((scb >> (4 * (ib % 2))) & 0xF), sc1 = sc0;
      uint w[8];
      const uint qs[8] = {q.x & 0xFF, (q.x >> 8) & 0xFF, (q.x >> 16) & 0xFF, q.x >> 24,
                          q.y & 0xFF, (q.y >> 8) & 0xFF, (q.y >> 16) & 0xFF, q.y >> 24};
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint g1 = qs[2 * l] | ((h << (8 - 2 * l)) & 256);
        const uint g2 = qs[2 * l + 1] | ((h << (7 - 2 * l)) & 256);
        const uint sg = (s >> (8 * l)) & 0xFF;
        w[2 * l] = apply_signs4(grid4w(grid, g1), sg & 0xF);
        w[2 * l + 1] = apply_signs4(grid4w(grid, g2), sg >> 4);
      }
      int lo = dot_4x8packed_ss_int(a8.s0, w[0]);
      lo += dot_4x8packed_ss_int(a8.s1, w[1]);
      lo += dot_4x8packed_ss_int(a8.s2, w[2]);
      lo += dot_4x8packed_ss_int(a8.s3, w[3]);
      int hi = dot_4x8packed_ss_int(a8.s4, w[4]);
      hi += dot_4x8packed_ss_int(a8.s5, w[5]);
      hi += dot_4x8packed_ss_int(a8.s6, w[6]);
      hi += dot_4x8packed_ss_int(a8.s7, w[7]);
      acc[r] += sa * d * (float)(sc0 * lo + sc1 * hi);
    }
  }
#pragma unroll
  for (int r = 0; r < GROWS; ++r) {
    const float v = HEAD_OUT(sub_group_reduce_add(acc[r]));
    const int n = r0 + r;
    if (lane == 0 && n < N) C[c_off + n] = accumulate ? C[c_off + n] + v : v;
  }
}
#endif
