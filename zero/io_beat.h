// A process-wide count of the expert reads from the SSD that finished (ExpertStore::read_ssd). A split model's shards
// waiting at a sum (zero/mgpu/engine_tp.h) take a change of it as a sign of life: a shard reading its experts from
// the SSD is slow, not hung. Before, a cold first prompt of DeepSeek-V4.1 over 4 B60s, every shard reading its
// experts at once, passed the 120-s limit while GPU 1 was still reading (issue #4, 2026-11-15).
#pragma once
#include <atomic>

inline std::atomic<unsigned long long> g_expert_reads_done{0};
