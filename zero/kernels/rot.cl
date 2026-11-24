// The H8 rotation in registers, shared by h8.cl (h8r_colmax, h8r_weights, h8r_act), decode.cl (h8r_decode, -DH8RDEC)
// and ops.cl (rmsnorm_h8r, -DH8NORM): the engine puts this file in front of those sources (engine_setup.h kernel()).
//
// What it computes: the Walsh-Hadamard transform of one block of HB values (Sylvester's +-1 matrix of order HB,
// unnormalized; the 1 / sqrt(HB) goes into the int8 scales), the same on the weights and on the activations, so
// (H x) . (H w) = HB * x . w. One sub-group of 16 lanes holds one block, ROT_NV values per lane.
//
// Why it splits: entry (i, j) of the matrix is (-1)^popcount(i & j), a product over the bits of the index, so the
// transform is one butterfly stage per index bit and the stages can run in any order. A block's index has its low
// bits in a lane's registers and 4 bits in the lane number (in a layout of the caller's: h8r_load contiguous per lane,
// h8r_act 4 values of each 64), so the register stages move nothing between lanes and the 4 lane stages are one shuffle
// each, without local memory or barriers. The callers' layouts differ; the transform does not, since each value keeps
// its own index.
#ifdef HB
#define ROT_NV (HB / 16)  // values per lane
#if ROT_NV < 1 || (ROT_NV & (ROT_NV - 1))
#error "rot.cl: HB must be a power of two of at least 16"
#endif

// -DH8_D (ARC_H8_D=1): a fixed +-1 diagonal first, the sign of value i of the block taken from a hash of i, so a
// block's mean does not all land on its first coefficient. Without it the values pass as they are.
inline float rot_sign(float v, uint i) {
#ifdef H8_D
  uint x = i * 0x9E3779B1u;
  x ^= x >> 15;
  x *= 0x85EBCA77u;
  return as_float(as_uint(v) ^ (x & 0x80000000u));
#else
  return v;
#endif
}

// the stages over the register bits: value j without bit h pairs with j + h. (Two loops of constant bounds, so the
// compiler unrolls them whole and v[] stays in registers; a loop over runs of 2h, whose inner bounds depend on the
// outer one, was left partly rolled in h8r_decode and put v[] in memory on Xe-LPG: ocloc mtl, 2026-11-17.)
inline void rot_regs(float* v) {
#pragma unroll
  for (uint h = 1; h < ROT_NV; h <<= 1)
#pragma unroll
    for (uint j = 0; j < ROT_NV; ++j)
      if ((j & h) == 0) {
        const float a = v[j], c = v[j + h];
        v[j] = a + c;
        v[j + h] = a - c;
      }
}

// the stages over the lane bits: lane l pairs with lane l ^ h; the lane without bit h keeps the sum, the one with it
// the partner's value minus its own. (A select of the two: flipping the own value's sign bit and adding, the same
// result, cost h8r_colmax and h8r_act a 1-2 KB spill on Xe-LPG.)
inline void rot_lanes(float* v) {
  const uint l = get_sub_group_local_id();
#pragma unroll
  for (uint h = 1; h < 16; h <<= 1)
#pragma unroll
    for (uint j = 0; j < ROT_NV; ++j) {
      const float o = intel_sub_group_shuffle_xor(v[j], h);
      v[j] = (l & h) ? (o - v[j]) : (v[j] + o);
    }
}

inline void rot_block(float* v) {
  rot_regs(v);
  rot_lanes(v);
}

// the largest |value| of the lane's ROT_NV
inline float rot_amax(const float* v) {
  float a = 0.0f;
#pragma unroll
  for (int j = 0; j < ROT_NV; ++j) a = fmax(a, fabs(v[j]));
  return a;
}

// four values v[0..3] to int8 at inv (127 / the block's or column's largest value), packed low byte first
inline uint rot_pack(const float* v, float inv) {
  uint w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int q = clamp((int)rint(v[i] * inv), -127, 127);
    w |= ((uint)(q & 0xFF)) << (8 * i);
  }
  return w;
}

// the float scale stored for an int8 value at amax: amax / 127, and the transform's 1 / sqrt(HB)
inline float rot_scale(float amax) { return amax / (127.0f * sqrt((float)HB)); }
#endif  // HB
