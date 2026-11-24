// AR4_128 (AutoRound INT4 experts, pull request #22): imports the pinned Flash Next AutoRound checkpoint's routed
// experts into an inferzero expert file, which tools/import_autoround_pack.cpp's hybrid GGUF names. The C++ form of
// the importer tools/import_autoround_experts.py was (no Python needed): the same file byte for byte, the same checks,
// the same progress lines on stdout.
// usage: import_autoround_experts SOURCE_DIRECTORY OUTPUT.experts --experimental-format [--wait-for-downloads]
//   SOURCE_DIRECTORY: the checkpoint's snapshot (config.json, model.safetensors.index.json, the expert shards) with
//   model-info.json, the Hugging Face API's file list (each file's size and SHA-256), scripts/autoround.ps1 writes it
//
// No model code is run. Every source shard is SHA-256 checked against model-info.json before it is read; every code
// and scale written is decoded again from the stored layout and compared with the source; the file is written under
// OUTPUT.experts.partial and renamed when complete; an existing output is never overwritten. Then OUTPUT.import.json
// (the shards' and the layers' SHA-256).
// Speed: the shards are hashed by two threads ahead of their use (Windows: CNG's SHA-256, which uses the CPU's SHA
// instructions; elsewhere the plain one of src/pbkdf2.h), the experts converted on every core, 32 at a time, while
// the 32 before them are hashed and written.
// (IMPORT_AR_LAYERS / IMPORT_AR_EXPERTS at compile time: fewer layers / experts per layer, for the dev test's tiny
//  checkpoint only; the release build is 48 x 512)
#include "../src/log.h"
#include "../src/json.h"
#include "../src/platform.h"
#ifdef _WIN32
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")  // (MinGW g++: -lbcrypt)
#else
#include "../src/pbkdf2.h"
#endif
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kAr4 = 0x41523447;
const char* const kRevision = "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897";
const char* const kModel = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound";
#ifndef IMPORT_AR_LAYERS
#define IMPORT_AR_LAYERS 48
#endif
#ifndef IMPORT_AR_EXPERTS
#define IMPORT_AR_EXPERTS 512
#endif
constexpr int kLayers = IMPORT_AR_LAYERS, kExperts = IMPORT_AR_EXPERTS;
constexpr uint64_t kPart[3] = {640 * 10 * 136, 640 * 10 * 136, 2560 * 3 * 136};  // (gate, up, down records)
constexpr uint64_t kRecord = kPart[0] + kPart[1] + kPart[2];
static_assert(kRecord % 4096 == 0);
const char* const kProj[3] = {"gate_proj", "up_proj", "down_proj"};

std::mutex g_out;  // (stdout: the hashers' waiting lines and the main thread's)
void say(const std::string& line) {
  std::lock_guard<std::mutex> lock(g_out);
  fwrite(line.data(), 1, line.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

// ---- Python's json.dumps forms: strings ASCII-only (\uXXXX), floats as repr
std::string py_str(const std::string& s) {
  std::string o = "\"";
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    uint32_t cp = c;
    size_t len = 1;
    if (c >= 0xF0 && i + 3 < s.size()) cp = (c & 7u) << 18 | (s[i + 1] & 63u) << 12 | (s[i + 2] & 63u) << 6 | (s[i + 3] & 63u), len = 4;
    else if (c >= 0xE0 && i + 2 < s.size()) cp = (c & 15u) << 12 | (s[i + 1] & 63u) << 6 | (s[i + 2] & 63u), len = 3;
    else if (c >= 0xC0 && i + 1 < s.size()) cp = (c & 31u) << 6 | (s[i + 1] & 63u), len = 2;
    i += len;
    char b[16];
    if (cp == '"') o += "\\\"";
    else if (cp == '\\') o += "\\\\";
    else if (cp == '\n') o += "\\n";
    else if (cp == '\r') o += "\\r";
    else if (cp == '\t') o += "\\t";
    else if (cp == '\b') o += "\\b";
    else if (cp == '\f') o += "\\f";
    else if (cp < 0x20 || (cp >= 0x7F && cp < 0x10000)) snprintf(b, sizeof b, "\\u%04x", cp), o += b;
    else if (cp >= 0x10000) {
      snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xD800 + ((cp - 0x10000) >> 10), 0xDC00 + ((cp - 0x10000) & 0x3FF));
      o += b;
    } else o.push_back((char)cp);
  }
  return o + "\"";
}
std::string py_float(double v) {
  char b[64];
  const double a = v < 0 ? -v : v;
  const auto fmt = a != 0 && (a < 1e-4 || a >= 1e16) ? std::chars_format::scientific : std::chars_format::fixed;
  const auto r = std::to_chars(b, b + sizeof b, v, fmt);
  std::string s(b, r.ptr);
  if (s.find_first_of(".en") == std::string::npos) s += ".0";
  return s;
}
// str(pathlib.Path(arg)): the separators of the system, "." parts and repeated separators dropped
fs::path u8p(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
std::string py_path(const std::string& arg) {
  fs::path p = u8p(arg);
  p.make_preferred();
  std::string s = p.lexically_normal().string();
  while (s.size() > 1 && (s.back() == '/' || s.back() == '\\') && s[s.size() - 2] != ':') s.pop_back();
  return s.empty() ? "." : s;
}

// ---- SHA-256 (hex digest)
class Sha {
 public:
  Sha(const Sha&) = delete;
  Sha& operator=(const Sha&) = delete;
#ifdef _WIN32
  Sha() {
    static BCRYPT_ALG_HANDLE alg = [] {
      BCRYPT_ALG_HANDLE a = nullptr;
      if (BCryptOpenAlgorithmProvider(&a, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) a = nullptr;
      return a;
    }();
    if (!alg || BCryptCreateHash(alg, &h_, nullptr, 0, nullptr, 0, 0) < 0) throw std::runtime_error("SHA-256 (CNG) unavailable");
  }
  ~Sha() { if (h_) BCryptDestroyHash(h_); }
  void add(const uint8_t* p, size_t n) {
    while (n) {
      const ULONG take = (ULONG)std::min<size_t>(n, (size_t)1 << 30);
      if (BCryptHashData(h_, (PUCHAR)p, take, 0) < 0) throw std::runtime_error("SHA-256 (CNG) failed");
      p += take, n -= take;
    }
  }
  std::string hex() {
    uint8_t d[32];
    if (BCryptFinishHash(h_, d, 32, 0) < 0) throw std::runtime_error("SHA-256 (CNG) failed");
    return to_hex(d);
  }

 private:
  BCRYPT_HASH_HANDLE h_ = nullptr;
#else
  Sha() = default;
  void add(const uint8_t* p, size_t n) { s_.add(p, n); }
  std::string hex() {
    uint8_t d[32];
    s_.finish(d);
    return to_hex(d);
  }

 private:
  pbkdf2::Sha256 s_;
#endif
  static std::string to_hex(const uint8_t* d) {
    static const char* x = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 32; ++i) s += x[d[i] >> 4], s += x[d[i] & 15];
    return s;
  }
};

std::string read_text(const fs::path& p) {
  FILE* f = plat::fopen_path(p, "rb");
  if (!f) throw std::runtime_error("cannot read " + p.string());
  std::string s;
  char b[1 << 16];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
  fclose(f);
  return s;
}
bool is_num(const Json& j, double v) { return j.is_number() && j.as_number() == v; }
bool is_basename(const std::string& f) {
  return !f.empty() && f != "." && f != ".." && f.find_first_of("/\\:") == std::string::npos;
}

// ---- the shards' SHA-256 checks, two threads ahead of the main thread's use (in the order the import reads them)
struct ShardCheck {
  std::string name;
  fs::path path;
  Json size, sha;  // (model-info.json's; null when the file is not listed)
  bool listed = false;
  bool done = false;
  std::string error, digest;
};

class Verifier {
 public:
  Verifier(std::vector<ShardCheck> items, bool wait) : items_(std::move(items)), wait_(wait) {
    for (int t = 0; t < kThreads; ++t) threads_.emplace_back([this] { run(); });
  }
  ~Verifier() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  // shard i checked: its digest (an error throws)
  const ShardCheck& need(size_t i) {
    std::unique_lock<std::mutex> lock(m_);
    want_ = std::max(want_, i + 1);
    cv_.notify_all();
    cv_.wait(lock, [&] { return items_[i].done; });
    if (!items_[i].error.empty()) throw std::runtime_error(items_[i].error);
    return items_[i];
  }

 private:
  static constexpr int kThreads = 2;
  std::vector<ShardCheck> items_;
  bool wait_;
  std::vector<std::thread> threads_;
  std::mutex m_;
  std::condition_variable cv_;
  size_t next_ = 0, want_ = 0;
  bool stop_ = false;

  bool stopping() {
    std::lock_guard<std::mutex> lock(m_);
    return stop_;
  }
  void run() {
    while (true) {
      size_t i;
      {
        std::unique_lock<std::mutex> lock(m_);
        // (at most kThreads shards past the one in use: the next ones stay in the file cache until read)
        cv_.wait(lock, [&] { return stop_ || (next_ < items_.size() && next_ < want_ + kThreads - 1); });
        if (stop_) return;
        i = next_++;
      }
      std::string error, digest;
      try {
        digest = check(items_[i]);
      } catch (const std::exception& e) {
        error = e.what();
      }
      std::lock_guard<std::mutex> lock(m_);
      items_[i].digest = digest;
      items_[i].error = error.empty() && digest.empty() ? "stopped" : error;
      items_[i].done = true;
      cv_.notify_all();
    }
  }
  std::string check(const ShardCheck& s) {
    auto last_report = std::chrono::steady_clock::now() - std::chrono::hours(1);
    std::error_code ec;
    while (!fs::exists(s.path, ec)) {
      if (!wait_) throw std::runtime_error("source file not found: " + s.path.string());
      if (std::chrono::steady_clock::now() - last_report > std::chrono::seconds(30)) {
        say("{\"waiting_for\": " + py_str(s.name) + "}");
        last_report = std::chrono::steady_clock::now();
      }
      for (int k = 0; k < 20; ++k) {
        if (stopping()) return {};
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    if (!s.listed) throw std::runtime_error("source file not listed in model-info.json: " + s.name);
    const uint64_t on_disk = fs::file_size(s.path, ec);
    if (ec || !is_num(s.size, (double)on_disk)) throw std::runtime_error("source file size mismatch: " + s.name);
    plat::SeqFile f;
    if (!f.open(s.path)) throw std::runtime_error("cannot read " + s.path.string());
    Sha sha;
    std::vector<uint8_t> buf((size_t)8 << 20);
    size_t n;
    while ((n = f.read(buf.data(), buf.size())) > 0) {
      if (stopping()) return {};
      sha.add(buf.data(), n);
    }
    const std::string digest = sha.hex();
    if (!s.sha.is_string() || digest != s.sha.as_string()) throw std::runtime_error("source SHA-256 mismatch: " + s.name);
    return digest;
  }
};

// ---- an opened shard: its mapping and its safetensors header
struct Shard {
  plat::FileMap map;
  uint64_t size = 0, base = 0;  // (base: 8 + the header's length, where the data starts)
  Json header;
  std::unordered_map<std::string, size_t> at;  // (tensor name -> header entry)
};

uint32_t load32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
uint16_t load16(const uint8_t* p) {
  uint16_t v;
  memcpy(&v, p, 2);
  return v;
}

// one expert matrix: its source tensors (qweight (k/8, n) u32, qzeros (k/128, n/8) u32, scales (k/128, n) f16)
struct Mat {
  const uint8_t *q = nullptr, *z = nullptr, *s = nullptr;
  int k = 0, n = 0;
  std::string prefix;
  uint8_t* out = nullptr;  // (n rows of kp / 256 superblocks of 136 bytes)
};

// The AR4_128 rows: per 256 codes a superblock of 136 bytes, its two f16 scales (groups of 128; past k: 0), 8 blocks
// of 16 bytes (each the low nibbles codes 0-15 of a 32, the high ones 16-31; past k: code 8), 4 zero bytes. Then
// every code and scale bit read back from those bytes and compared with the source. "" or the error.
std::string convert(const Mat& m) {
  const int k = m.k, n = m.n, kp = (k + 255) / 256 * 256, nsb = kp / 256, ng = k / 128;
  for (int i = 0; i < ng * (n / 8); ++i)
    if (load32(m.z + 4 * (size_t)i) != 0x77777777u) return "nonconstant symmetric zero point: " + m.prefix;
  for (int i = 0; i < ng * n; ++i)
    if ((load16(m.s + 2 * (size_t)i) & 0x7C00) == 0x7C00) return "nonfinite source scales: " + m.prefix;
  auto word = [&](int r, int c) { return load32(m.q + 4 * ((size_t)r * n + c)); };  // (codes 8r..8r+7 of column c)
  for (int c = 0; c < n; ++c) {
    uint8_t* row = m.out + (size_t)c * nsb * 136;
    for (int sb = 0; sb < nsb; ++sb) {
      uint8_t* b = row + (size_t)sb * 136;
      for (int t = 0; t < 2; ++t) {
        const int g = 2 * sb + t;
        const uint16_t v = g < ng ? load16(m.s + 2 * ((size_t)g * n + c)) : 0;
        memcpy(b + 2 * t, &v, 2);
      }
      for (int blk = 0; blk < 8; ++blk) {
        const int kk = sb * 256 + blk * 32;
        uint8_t* d = b + 4 + blk * 16;
        if (kk >= k) {  // (k is a multiple of 128: a block is all source codes or all padding)
          memset(d, 0x88, 16);
          continue;
        }
        const uint32_t lo[2] = {word(kk / 8, c), word(kk / 8 + 1, c)}, hi[2] = {word(kk / 8 + 2, c), word(kk / 8 + 3, c)};
        for (int i = 0; i < 16; ++i)
          d[i] = (uint8_t)(((lo[i >> 3] >> (4 * (i & 7))) & 15) | (((hi[i >> 3] >> (4 * (i & 7))) & 15) << 4));
      }
      memset(b + 132, 0, 4);
    }
  }
  // (the check, decoded independently: code kk of row c is nibble (kk % 32 >= 16) of byte 4 + (kk % 256) / 32 * 16
  //  + kk % 16 of superblock kk / 256; it implies equal decoded F32 weights)
  for (int c = 0; c < n; ++c) {
    const uint8_t* row = m.out + (size_t)c * nsb * 136;
    for (int kk = 0; kk < k; ++kk) {
      const uint8_t byte = row[(size_t)(kk / 256) * 136 + 4 + (kk % 256) / 32 * 16 + kk % 16];
      const uint32_t stored = (kk % 32) < 16 ? byte & 15u : byte >> 4;
      if (stored != ((word(kk / 8, c) >> (4 * (kk % 8))) & 15u)) return "imported codes differ: " + m.prefix;
    }
    for (int g = 0; g < ng; ++g)
      if (load16(row + (size_t)(g / 2) * 136 + 2 * (g % 2)) != load16(m.s + 2 * ((size_t)g * n + c)))
        return "imported scale bits differ: " + m.prefix;
  }
  return "";
}

struct Args {
  fs::path source, output;
  std::string output_arg;
  bool wait = false, experimental = false;
};

int run(const Args& args, fs::path& created) {
  std::error_code ec;
  if (fs::exists(args.output, ec)) throw std::runtime_error("refusing to overwrite an existing expert file");
  const Json info = Json::parse(read_text(args.source / "model-info.json"));
  const Json config = Json::parse(read_text(args.source / "config.json"));
  const Json index = Json::parse(read_text(args.source / "model.safetensors.index.json"));
  std::unordered_map<std::string, std::string> weight_map;
  for (const auto& [name, file] : index["weight_map"].entries()) weight_map[name] = file.as_string();
  std::unordered_map<std::string, const Json*> files;
  for (const auto& item : info["siblings"].items()) files[item["rfilename"].as_string()] = &item;
  const Json& quant = config["quantization_config"];
  if (info["id"].as_string() != kModel || info["sha"].as_string() != kRevision)
    throw std::runtime_error("unexpected checkpoint identity/revision");
  if (!is_num(quant["bits"], 4) || !is_num(quant["group_size"], 128) || !quant["sym"].is_bool() || !quant["sym"].as_bool())
    throw std::runtime_error("expected symmetric INT4/group-128");
  if (quant["packing_format"].as_string() != "auto_round:auto_gptq" || !quant["packing_format"].is_string())
    throw std::runtime_error("unsupported source packing convention");
  const Json& text = config["text_config"];
  if (!is_num(text["hidden_size"], 2560) || !is_num(text["moe_intermediate_size"], 640) ||
      !is_num(text["num_hidden_layers"], 48) || !is_num(text["num_experts"], 512) ||
      !is_num(text["num_experts_per_tok"], 10) || !is_num(text["max_position_embeddings"], 262144))
    throw std::runtime_error("unexpected source architecture or context capacity");

  // the shards in the order the import first reads them (the checks run ahead in that order)
  auto name_of = [](int layer, int expert, int p, const char* what) {
    return "model.language_model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(expert) + "." +
           kProj[p] + "." + what;
  };
  std::vector<ShardCheck> order;
  std::unordered_map<std::string, size_t> order_at;
  for (int layer = 0; layer < kLayers; ++layer)
    for (int e = 0; e < kExperts; ++e)
      for (int p = 0; p < 3; ++p)
        for (const char* what : {"qweight", "qzeros", "scales"}) {
          const auto it = weight_map.find(name_of(layer, e, p, what));
          if (it == weight_map.end() || !is_basename(it->second) || order_at.count(it->second)) continue;
          ShardCheck s;
          s.name = it->second;
          s.path = args.source / u8p(it->second);
          const auto f = files.find(it->second);
          if (f != files.end()) s.listed = true, s.size = (*f->second)["size"], s.sha = (*f->second)["lfs"]["sha256"];
          order_at[s.name] = order.size();
          order.push_back(std::move(s));
        }
  Verifier verifier(std::move(order), args.wait);
  std::unordered_map<std::string, std::unique_ptr<Shard>> opened;
  std::vector<std::pair<std::string, std::string>> verified;  // (name, digest), in the order opened

  // a tensor's data, its shape and dtype checked (lines: the source_verified lines to print)
  auto tensor = [&](const std::string& name, int64_t rows, int64_t cols, bool f16, std::vector<std::string>& lines) {
    const auto w = weight_map.find(name);
    if (w == weight_map.end()) throw std::runtime_error("tensor missing from model.safetensors.index.json: " + name);
    const std::string& filename = w->second;
    if (!is_basename(filename)) throw std::runtime_error("source filename is not a basename");
    auto it = opened.find(filename);
    if (it == opened.end()) {
      const ShardCheck& check = verifier.need(order_at.at(filename));
      auto shard = std::make_unique<Shard>();
      if (!shard->map.open_fs(check.path) || !shard->map.size(shard->size) || shard->size < 8 || shard->map.map() != 0)
        throw std::runtime_error("cannot map " + check.path.string());
      uint64_t header_bytes;
      memcpy(&header_bytes, shard->map.data(), 8);
      if (header_bytes > 16 * 1024 * 1024) throw std::runtime_error("oversized safetensors header");
      if (8 + header_bytes > shard->size) throw std::runtime_error("truncated safetensors header: " + filename);
      shard->header = Json::parse(std::string((const char*)shard->map.data() + 8, (size_t)header_bytes));
      const auto& entries = shard->header.entries();
      for (size_t i = 0; i < entries.size(); ++i) shard->at[entries[i].first] = i;
      shard->base = 8 + header_bytes;
      verified.emplace_back(filename, check.digest);
      lines.push_back("{\"source_verified\": " + py_str(filename) + ", \"sha256\": " + py_str(check.digest) + "}");
      it = opened.emplace(filename, std::move(shard)).first;
    }
    const Shard& sh = *it->second;
    const auto at = sh.at.find(name);
    if (at == sh.at.end()) throw std::runtime_error("tensor missing from its shard's header: " + name);
    const Json& meta = sh.header.entries()[at->second].second;
    const Json& shape = meta["shape"];
    if (shape.size() != 2 || !shape.is_array() || !is_num(shape[0], (double)rows) || !is_num(shape[1], (double)cols) ||
        meta["dtype"].as_string() != (f16 ? "F16" : "I32"))
      throw std::runtime_error("unexpected tensor shape/dtype: " + name);
    const Json& offs = meta["data_offsets"];
    if (!offs.is_array() || offs.size() != 2 || !offs[0].is_number() || !offs[1].is_number())
      throw std::runtime_error("invalid tensor bounds: " + name);
    const double start = offs[0].as_number(), stop = offs[1].as_number();
    const double size = (double)(rows * cols * (f16 ? 2 : 4));
    if (start < 0 || stop - start != size || (double)sh.base + stop > (double)sh.size)
      throw std::runtime_error("invalid tensor bounds: " + name);
    return sh.map.data() + sh.base + (uint64_t)start;
  };

  // the header: magic, version 1, the revision's first 16 hex digits, 49, 512, the length; a row per layer
  const uint64_t file_bytes = 4096 + (uint64_t)kLayers * kExperts * kRecord;
  std::vector<uint8_t> header(4096, 0);
  auto put32 = [&](size_t at, uint32_t v) { memcpy(&header[at], &v, 4); };
  auto put64 = [&](size_t at, uint64_t v) { memcpy(&header[at], &v, 8); };
  put32(0, 0x58435241);
  put32(4, 1);
  put64(8, std::stoull(std::string(kRevision, 16), nullptr, 16));
  put64(16, 49);
  put64(24, 512);
  put64(32, file_bytes);
  for (int layer = 0; layer < kLayers; ++layer) {
    const size_t at = 40 + (size_t)layer * 56;
    for (int i = 0; i < 3; ++i) put32(at + 4 * i, kAr4);
    for (int i = 0; i < 3; ++i) put64(at + 16 + 8 * i, kPart[i]);
    put64(at + 40, kRecord);
    put64(at + 48, 4096 + (uint64_t)layer * kExperts * kRecord);
  }
  if (args.output.has_parent_path()) fs::create_directories(args.output.parent_path());
  fs::path temporary = args.output;
  temporary += ".partial";
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
  FILE* out = plat::fopen_path(temporary, "wb");
  if (!out) throw std::runtime_error("cannot write " + temporary.string());
  created = temporary;
  std::unique_ptr<FILE, int (*)(FILE*)> out_guard(out, fclose);
  uint64_t written = 0;
  auto write = [&](const uint8_t* p, size_t n) {
    if (n && fwrite(p, 1, n, out) != n) throw std::runtime_error("write failed");
    written += n;
  };
  write(header.data(), header.size());

  // the experts, kChunk at a time: converted on every core into one buffer while the chunk before is hashed and
  // written from the other
  constexpr int kChunk = std::min(32, kExperts);
  static_assert(kExperts % kChunk == 0);
  std::vector<uint8_t> buf[2] = {std::vector<uint8_t>(kChunk * kRecord), std::vector<uint8_t>(kChunk * kRecord)};
  const int workers = std::max(1, (int)std::thread::hardware_concurrency() - 1);
  std::vector<std::string> layer_hashes;
  std::unique_ptr<Sha> digest;
  auto flush_chunk = [&](int layer, int e0, const std::vector<uint8_t>& b) {
    if (e0 == 0) digest = std::make_unique<Sha>();
    std::string error, hash_error;  // (the layer's hash on a thread of its own while the bytes are written)
    std::thread hashing([&] {
      try {
        digest->add(b.data(), b.size());
      } catch (const std::exception& e) {
        hash_error = e.what();
      }
    });
    try {
      write(b.data(), b.size());
    } catch (const std::exception& e) {
      error = e.what();
    }
    hashing.join();
    if (!error.empty() || !hash_error.empty()) throw std::runtime_error(error.empty() ? hash_error : error);
    if (e0 + kChunk == kExperts) {
      if (fflush(out) != 0) throw std::runtime_error("write failed");
      layer_hashes.push_back(digest->hex());
      say("{\"layer_complete\": " + std::to_string(layer) + ", \"bytes_written\": " + std::to_string(written) +
          ", \"sha256\": " + py_str(layer_hashes.back()) + ", \"elapsed_s\": " + py_float(elapsed()) + "}");
    }
  };
  int prev_layer = -1, prev_e0 = 0;
  for (int c = 0; c < kLayers * kExperts / kChunk; ++c) {
    const int layer = c * kChunk / kExperts, e0 = c * kChunk % kExperts;
    std::vector<uint8_t>& b = buf[c % 2];
    // this chunk's tensors, in the import's order (the first error stops there, as the script stopped)
    std::vector<Mat> mats;
    std::vector<std::string> lines;
    std::string meta_error;
    try {
      for (int e = e0; e < e0 + kChunk; ++e)
        for (int p = 0; p < 3; ++p) {
          Mat m;
          m.k = p == 2 ? 640 : 2560;
          m.n = p == 2 ? 2560 : 640;
          m.prefix = name_of(layer, e, p, "");
          m.out = b.data() + (size_t)(e - e0) * kRecord + (p > 0 ? kPart[0] : 0) + (p > 1 ? kPart[1] : 0);
          if (weight_map.count(m.prefix + "g_idx")) throw std::runtime_error("activation-order indices need a separate importer");
          m.q = tensor(m.prefix + "qweight", m.k / 8, m.n, false, lines);
          m.z = tensor(m.prefix + "qzeros", m.k / 128, m.n / 8, false, lines);
          m.s = tensor(m.prefix + "scales", m.k / 128, m.n, true, lines);
          mats.push_back(std::move(m));
        }
    } catch (const std::exception& e) {
      meta_error = e.what();
    }
    std::vector<std::string> errors(mats.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < workers; ++t)
      pool.emplace_back([&] {
        for (size_t i; (i = next++) < mats.size();) errors[i] = convert(mats[i]);
      });
    std::string flush_error;
    try {
      if (prev_layer >= 0) flush_chunk(prev_layer, prev_e0, buf[(c + 1) % 2]);
    } catch (const std::exception& e) {
      flush_error = e.what();
    }
    for (auto& t : pool) t.join();
    if (!flush_error.empty()) throw std::runtime_error(flush_error);
    for (const auto& l : lines) say(l);
    for (const auto& e : errors)
      if (!e.empty()) throw std::runtime_error(e);
    if (!meta_error.empty()) throw std::runtime_error(meta_error);
    prev_layer = layer, prev_e0 = e0;
  }
  flush_chunk(prev_layer, prev_e0, buf[(kLayers * kExperts / kChunk - 1) % 2]);
  if (fflush(out) != 0) throw std::runtime_error("write failed");
#ifdef _WIN32
  if (!FlushFileBuffers((HANDLE)_get_osfhandle(_fileno(out)))) throw std::runtime_error("write failed (flushing to disk)");
#else
  if (fsync(fileno(out)) != 0) throw std::runtime_error("write failed (flushing to disk)");
#endif
  if (fclose(out_guard.release()) != 0) throw std::runtime_error("write failed (closing the file)");
  if (fs::file_size(temporary) != file_bytes) throw std::runtime_error("incorrect expert-file length");

  // the summary: OUTPUT.import.json (json.dumps indent=2) and the last line
  std::vector<std::pair<std::string, std::string>> fields = {
      {"source_model", py_str(kModel)},
      {"revision", py_str(kRevision)},
      {"format", "\"AR4_128\""},
      {"gpu_superblock_bytes", "136"},
      {"record_bytes", std::to_string(kRecord)},
      {"file_bytes", std::to_string(file_bytes)},
      {"validated_weight_count", std::to_string((uint64_t)kLayers * kExperts * 3 * 2560 * 640)}};
  auto dict = [&](bool pretty) {
    std::string s;
    const char* nl = pretty ? "\n  " : "";
    const char* nl2 = pretty ? "\n    " : "";
    const char* sep = pretty ? "," : ", ";
    for (const auto& [k, v] : fields) s += (s.empty() ? "" : sep) + std::string(nl) + py_str(k) + ": " + v;
    s += sep + std::string(nl) + "\"source_sha256\": {";
    for (size_t i = 0; i < verified.size(); ++i)
      s += (i ? sep : "") + std::string(nl2) + py_str(verified[i].first) + ": " + py_str(verified[i].second);
    s += (verified.empty() || !pretty ? "" : "\n  ") + std::string("}");
    s += sep + std::string(nl) + "\"layer_sha256\": [";
    for (size_t i = 0; i < layer_hashes.size(); ++i) s += (i ? sep : "") + std::string(nl2) + py_str(layer_hashes[i]);
    s += (layer_hashes.empty() || !pretty ? "" : "\n  ") + std::string("]");
    return s;
  };
  const std::string seconds = py_float(elapsed());
  if (fs::exists(args.output, ec)) throw std::runtime_error("refusing to overwrite an existing expert file");
  fs::rename(temporary, args.output);
  created.clear();
  fs::path summary = args.output;
  summary.replace_extension(".import.json");
  const std::string body = "{" + dict(true) + ",\n  \"seconds\": " + seconds + "\n}";
  FILE* sf = plat::fopen_path(summary, "wb");
  if (!sf || fwrite(body.data(), 1, body.size(), sf) != body.size() || fclose(sf) != 0)
    throw std::runtime_error("cannot write " + summary.string());
  say("{\"import_complete\": " + py_str(py_path(args.output_arg)) + ", " + dict(false) + ", \"seconds\": " + seconds + "}");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* usage =
      "usage: import_autoround_experts SOURCE_DIRECTORY OUTPUT.experts --experimental-format [--wait-for-downloads]\n";
  Args args;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--experimental-format") args.experimental = true;
    else if (a == "--wait-for-downloads") args.wait = true;
    else if (a == "-h" || a == "--help") {
      printf("%sImports the pinned Flash Next AutoRound INT4 experts into an inferzero expert file (AR4_128, a private\n"
             "experimental format: --experimental-format acknowledges it). --wait-for-downloads: wait for shards still\n"
             "downloading.\n", usage);
      return 0;
    } else if (a.size() > 1 && a[0] == '-') {
      LOG_ERR("%simport_autoround_experts: error: unrecognized arguments: %s\n", usage, a.c_str());
      return 2;
    } else pos.push_back(a);
  }
  if (pos.size() != 2) {
    LOG_ERR("%simport_autoround_experts: error: expected SOURCE_DIRECTORY and OUTPUT\n", usage);
    return 2;
  }
  if (!args.experimental) {
    LOG_ERR("%simport_autoround_experts: error: private experimental AR4_128 format: explicitly pass --experimental-format\n", usage);
    return 2;
  }
  args.source = u8p(pos[0]);
  args.output = u8p(pos[1]);
  args.output_arg = pos[1];
  fs::path created;  // (the .partial once it exists: removed when a step after that fails)
  try {
    return run(args, created);
  } catch (const std::exception& error) {
    LOG_ERR("import_autoround_experts: %s\n", error.what());
    if (!created.empty()) {
      std::error_code ec;
      fs::remove(created, ec);
    }
    return 1;
  }
}
