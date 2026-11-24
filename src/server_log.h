// The server's log file by default (2026-11-23, the user: "It should log by default, if not how we spot errors?"; a
// B580 under Linux lost its GPU and its owner found no log: there serverz wrote to stderr only). The same files on
// both systems, as zero-server.ps1 has kept them on Windows (docs/logging.md):
//   <the logs folder>/server.log                       this run's, written anew at each start
//   <the logs folder>/server_<YYYY-MM-DD_HH-MM-SS>.log  earlier runs', named by their last line's local time
//                                                      (" (2)" etc. for the same second); the newest 10
//   <the logs folder>/server_<...>.zip                  older ones, each in a zip of the same name with its time;
//                                                      zips over 90 days old or past 200 MB together removed,
//                                                      oldest first; never more than 30 .log where zipping fails
// The logs folder: --log-dir D, else ARC_LOG_DIR, else <the data folder>/logs (%USERPROFILE%\.inferzero\logs,
// ~/.inferzero/logs; ARC_DATA_DIR moves it with the rest). Created if missing; one that cannot be written: a W line
// and the default folder; the default one too: a W line and no file, the server runs on.
// One owner of the rotation per start: where serverz's stderr already is that server.log (zero-server.ps1's
// redirect, which rotated before it started serverz), serverz neither rotates nor opens it again.
#pragma once
#include "platform.h"
#include "app_dir.h"
#include "gpu_fail.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace slog {
namespace fs = std::filesystem;

inline fs::path default_dir() { return fs::path(app_dir()) / "logs"; }

// d made and a file creatable in it
inline bool usable_dir(const fs::path& d) {
  std::error_code ec;
  fs::create_directories(d, ec);
  if (!fs::is_directory(d, ec)) return false;
  const fs::path probe = d / ".write_test";
  FILE* f = plat::fopen_path(probe, "wb");
  if (!f) return false;
  fclose(f);
  fs::remove(probe, ec);
  return true;
}

// a file's last write as local time, yyyy-MM-dd_HH-mm-ss (zero-server.ps1's stamp); its time_t in *t
inline std::string stamp_of(const fs::path& f, std::time_t* t = nullptr) {
  std::error_code ec;
  const auto ft = fs::last_write_time(f, ec);
  std::time_t tt = std::time(nullptr);
  if (!ec)  // (the file clock shifted by the two clocks' difference now: MSVC has no file_clock::to_sys, g++ 13 no clock_cast)
    tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() +
                                              std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                                  ft - std::filesystem::file_time_type::clock::now()));
  if (t) *t = tt;
  std::tm tm{};
  if (!plat::local_time(tt, tm)) return "0000-00-00_00-00-00";
  char b[32];
  strftime(b, sizeof b, "%Y-%m-%d_%H-%M-%S", &tm);
  return b;
}

// CRC-32 (zip's) in pieces: state 0xFFFFFFFF first, each piece through crc32_update, the result state ^ 0xFFFFFFFF
inline uint32_t crc32_update(uint32_t c, const unsigned char* p, size_t n) {
  static const auto tab = [] {
    std::vector<uint32_t> t(256);
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t v = i;
      for (int k = 0; k < 8; ++k) v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1;
      t[i] = v;
    }
    return t;
  }();
  for (size_t i = 0; i < n; ++i) c = tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c;
}

// src in a zip of its own (one entry, stored: no compression), written to a temporary file and then put in place,
// with src's time; false (and nothing left behind) on failure. Streamed in pieces of 1 MiB (the CRC in a first pass):
// a log of hundreds of MB is never held in RAM
inline bool zip_one(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  const uintmax_t sz = fs::file_size(src, ec);
  if (ec || sz >= 0xFFFFFFFFull) return false;  // (no zip64: such a log stays a .log)
  std::vector<unsigned char> buf(1 << 20);
  uint32_t crc = 0;
  {
    FILE* in = plat::fopen_path(src, "rb");
    if (!in) return false;
    uint32_t c = 0xFFFFFFFFu;
    uintmax_t left = sz;
    while (left) {
      const size_t n = fread(buf.data(), 1, (size_t)std::min<uintmax_t>(left, buf.size()), in);
      if (!n) break;
      c = crc32_update(c, buf.data(), n);
      left -= n;
    }
    fclose(in);
    if (left) return false;
    crc = c ^ 0xFFFFFFFFu;
  }
  std::time_t t = 0;
  stamp_of(src, &t);
  std::tm tm{};
  plat::local_time(t, tm);
  const uint16_t dtime = (uint16_t)((tm.tm_hour << 11) | (tm.tm_min << 5) | (tm.tm_sec / 2));
  const uint16_t ddate = (uint16_t)((std::max(0, tm.tm_year + 1900 - 1980) << 9) | ((tm.tm_mon + 1) << 5) | tm.tm_mday);
  const std::string name = src.filename().string();
  std::vector<unsigned char> o;
  auto u16 = [&](uint32_t v) { o.push_back((unsigned char)(v & 0xFF)), o.push_back((unsigned char)((v >> 8) & 0xFF)); };
  auto u32 = [&](uint32_t v) { u16(v & 0xFFFF), u16(v >> 16); };
  auto header = [&](bool central, uint32_t off) {
    u32(central ? 0x02014b50u : 0x04034b50u);
    if (central) u16(20);   // (made by)
    u16(20), u16(0), u16(0);  // (version needed, flags, method: stored)
    u16(dtime), u16(ddate), u32(crc), u32((uint32_t)sz), u32((uint32_t)sz), u16((uint32_t)name.size()), u16(0);
    if (central) u16(0), u16(0), u16(0), u32(0), u32(off);  // (comment, disk, internal / external attributes, offset)
    o.insert(o.end(), name.begin(), name.end());
  };
  header(false, 0);
  const std::vector<unsigned char> head(o);
  const uintmax_t cd_at = head.size() + sz;
  o.clear();
  header(true, 0);
  const size_t cd_size = o.size();
  u32(0x06054b50u), u16(0), u16(0), u16(1), u16(1), u32((uint32_t)cd_size), u32((uint32_t)cd_at), u16(0);
  if (cd_at >= 0xFFFFFFFFull) return false;
  const fs::path tmp = dst.string() + ".tmp";
  FILE* out = plat::fopen_path(tmp, "wb");
  if (!out) return false;
  bool ok = fwrite(head.data(), 1, head.size(), out) == head.size();
  if (FILE* in = ok ? plat::fopen_path(src, "rb") : nullptr) {
    uintmax_t left = sz;
    while (ok && left) {
      const size_t n = fread(buf.data(), 1, (size_t)std::min<uintmax_t>(left, buf.size()), in);
      ok = n > 0 && fwrite(buf.data(), 1, n, out) == n;
      left -= n;
    }
    fclose(in);
  } else {
    ok = false;
  }
  ok = ok && fwrite(o.data(), 1, o.size(), out) == o.size();
  ok = fclose(out) == 0 && ok;
  if (ok) {
    fs::rename(tmp, dst, ec);
    ok = !ec;
  }
  if (!ok) {
    fs::remove(tmp, ec);
    return false;
  }
  const auto ft = fs::last_write_time(src, ec);
  if (!ec) fs::last_write_time(dst, ft, ec);
  return true;
}

// an earlier run's log moved into dir as server_<its time>.log; false if it could not be moved (in use: a file another
// process holds open, which Windows will not rename or remove; nothing changed then, a copy made on the way removed)
inline bool keep_earlier(const fs::path& dir, const fs::path& log) {
  std::error_code ec;
  const std::string st = stamp_of(log);
  fs::path dest = dir / ("server_" + st + ".log");
  for (int k = 2; fs::exists(dest, ec); ++k) dest = dir / ("server_" + st + " (" + std::to_string(k) + ").log");
  fs::rename(log, dest, ec);
  if (ec) {  // (another drive: a copy, then the original removed)
    ec.clear();
    fs::copy_file(log, dest, ec);
    if (ec) return false;
    const auto ft = fs::last_write_time(log, ec);
    if (!ec) fs::last_write_time(dest, ft, ec);
    ec.clear();
    if (!fs::remove(log, ec) || ec) {  // (still there: in use. The review of ba6b107: this was taken as moved)
      fs::remove(dest, ec);
      return false;
    }
  }
  gpufail::marker_log_moved(log.string(), dest.string());  // (a GPU fault's marker names the log of its run)
  return true;
}

// the newest 10 earlier logs as they are, older ones zipped; at most 30 .log; zips past 90 days or 200 MB removed
inline void prune(const fs::path& dir) {
  std::error_code ec;
  struct F { fs::path p; fs::file_time_type t; uintmax_t n; };
  auto list = [&](const char* ext) {
    std::vector<F> v;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      const std::string n = e.path().filename().string();
      if (n.rfind("server_", 0) != 0 || e.path().extension() != ext) continue;
      std::error_code e2;
      v.push_back({e.path(), fs::last_write_time(e.path(), e2), fs::file_size(e.path(), e2)});
    }
    std::sort(v.begin(), v.end(), [](const F& a, const F& b) { return a.t > b.t; });  // (newest first)
    return v;
  };
  std::vector<F> logs = list(".log");
  for (size_t i = 10; i < logs.size(); ++i) {
    const fs::path z = logs[i].p.parent_path() / (logs[i].p.stem().string() + ".zip");
    if (zip_one(logs[i].p, z)) fs::remove(logs[i].p, ec);
  }
  logs = list(".log");
  for (size_t i = 30; i < logs.size(); ++i) fs::remove(logs[i].p, ec);
  const auto old = fs::file_time_type::clock::now() - std::chrono::hours(24 * 90);
  uintmax_t sum = 0;
  for (const F& z : list(".zip")) {
    sum += z.n;
    if (z.t < old || sum > 200ull * 1024 * 1024) fs::remove(z.p, ec);
  }
}

// The logs folder's lock, logs/server.lock, held for the whole run: one serverz owns server.log and its rotation. A
// second one (started twice, or by hand next to zero-server / a systemd service) used to rotate the first one's live
// log away before failing on the port: on Windows the copy was taken as moved and the live file truncated ("wb"), on
// Linux the live server went on writing into the renamed file (the review of ba6b107). Windows: the file opened
// without write sharing; Linux: flock, not inherited by children (the terminal tool, MCP servers). Freed by the
// system when the process ends, however it ends. True: this process holds it; pid: the holder's, when known.
inline bool take_lock(const fs::path& dir, unsigned long* pid) {
  const fs::path p = dir / "server.lock";
  const std::string me = std::to_string(plat::process_id()) + "\n";
#ifdef _WIN32
  HANDLE h = CreateFileW(p.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  // (only a sharing violation is another server holding it; any other failure (access denied, a read-only folder): no
  //  lock possible, as on Linux, the run goes on as before without the guard. The audit of 2026-11-23)
  if (h == INVALID_HANDLE_VALUE && GetLastError() != ERROR_SHARING_VIOLATION) return true;
  if (h == INVALID_HANDLE_VALUE) {
    if (pid) {
      *pid = 0;
      if (FILE* f = plat::fopen_path(p, "rb")) {
        if (fscanf(f, "%lu", pid) != 1) *pid = 0;
        fclose(f);
      }
    }
    return false;
  }
  SetEndOfFile(h);
  DWORD w = 0;
  WriteFile(h, me.data(), (DWORD)me.size(), &w, nullptr);
  FlushFileBuffers(h);
  // (h kept open until the process ends: that is the lock)
  return true;
#else
  const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return true;  // (no lock file possible: as before, without the guard)
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    if (pid) {
      char b[32] = {};
      *pid = ::read(fd, b, sizeof b - 1) > 0 ? strtoul(b, nullptr, 10) : 0;
    }
    ::close(fd);
    return false;
  }
  if (ftruncate(fd, 0) == 0) {
    const ssize_t w = ::write(fd, me.data(), me.size());
    (void)w;
  }
  // (fd kept open until the process ends: that is the lock)
  return true;
#endif
}

// (the 256 MB roll within a run: a long service at -lv 3 grew one file without end. The log's mutex is held here, so
//  nothing in it logs: the move, the prune and the reopen only. On a failure the file is kept open as it is)
inline fs::path& roll_log() { static fs::path p; return p; }
inline void roll_now() {
  zlog::Config& c = zlog::cfg();
  const fs::path& log = roll_log();
  if (!c.dfile || log.empty()) return;
  fclose(c.dfile);
  c.dfile = nullptr;
  const bool moved = keep_earlier(log.parent_path(), log);
  // (the prune, which zips, in a thread of its own: under the log's mutex it held every log line for seconds, audit
  //  2026-11-23; one at a time, nothing in it logs)
  if (moved) {
    static std::atomic<bool> pruning{false};
    if (!pruning.exchange(true)) {
      std::thread([dir = log.parent_path()] {
        prune(dir);
        pruning.store(false);
      }).detach();
    }
  }
  c.dfile = plat::fopen_path(log, moved ? "wb" : "ab");
  c.dfile_bytes = 0;
  if (!moved) c.dfile_cap *= 2;  // (not again at once)
}
constexpr size_t kRollBytes = 256ull << 20;

// The server's log at its start (see the top): cli_dir is --log-dir's value or "". Sets zlog's dfile and its path.
inline void open_default(const std::string& cli_dir) {
  zlog::Config& c = zlog::cfg();
  if (c.disabled) return;
  fs::path dir;
  const char* env = getenv("ARC_LOG_DIR");
  const std::string asked = !cli_dir.empty() ? cli_dir : env && *env ? std::string(env) : std::string();
  if (!asked.empty()) {
    if (usable_dir(asked)) dir = asked;
    else LOG_WRN("inferzero: the log folder %s cannot be written: the default one\n", asked.c_str());
  }
  if (dir.empty()) {
    dir = default_dir();
    if (!usable_dir(dir)) {
      LOG_WRN("inferzero: the log folder %s cannot be written: no log file (the log on stderr only)\n", dir.string().c_str());
      return;
    }
  }
  const fs::path log = dir / "server.log";
  // (a file of this run's own: server_pid<N>.log, an earlier run's name, so the rotation keeps and prunes it later)
  auto own_file = [&](const char* why) {
    const fs::path mine = dir / ("server_pid" + std::to_string(plat::process_id()) + ".log");
    c.dfile = plat::fopen_path(mine, "wb");
    if (c.dfile) snprintf(c.dfile_path, sizeof c.dfile_path, "%s", mine.string().c_str());
    LOG_WRN("inferzero: %s: this run's log is %s\n", why, c.dfile ? mine.string().c_str() : "on stderr only");
  };
  unsigned long holder = 0;
  if (!take_lock(dir, &holder)) {  // (another serverz owns this folder's server.log: it is left alone)
    const std::string why = "another server (process " + std::to_string(holder) + ") writes " + log.string();
    own_file(why.c_str());
    return;
  }
  snprintf(c.dfile_path, sizeof c.dfile_path, "%s", log.string().c_str());
  // (stderr already that file: zero-server.ps1 rotated it and redirects every line there)
  if (FILE* f = plat::fopen_path(log, "rb")) {
    const bool same = plat::same_file(stderr, f);
    fclose(f);
    if (same) return;
  }
  // (a GPU hang marker this start will refuse on: nothing rotated, so the run it names stays; the refusal follows)
  if (gpufail::marker_live()) {
    c.dfile_path[0] = 0;
    return;
  }
  std::error_code ec;
  // (the place before 2026-11-23: the data folder's top, zero-server.ps1's server.log there; an earlier run now)
  const fs::path top = fs::path(app_dir()) / "server.log";
  if (fs::exists(top, ec) && !fs::equivalent(top, log, ec)) {
    bool same = false;
    if (FILE* f = plat::fopen_path(top, "rb")) {
      same = plat::same_file(stderr, f);
      fclose(f);
    }
    if (!same) keep_earlier(dir, top);
  }
  if (fs::exists(log, ec) && !keep_earlier(dir, log)) {  // (in use: never truncated)
    own_file(("the last run's log " + log.string() + " is in use").c_str());
    return;
  }
  prune(dir);
  c.dfile = plat::fopen_path(log, "wb");
  if (!c.dfile) {
    LOG_WRN("inferzero: cannot open the log file %s: the log on stderr only\n", log.string().c_str());
    c.dfile_path[0] = 0;
    return;
  }
  roll_log() = log;
  c.dfile_cap = kRollBytes;
  if (const char* e = getenv("ARC_LOG_ROLL_KB")) c.dfile_cap = (size_t)(std::max(0.1, atof(e)) * 1024);  // (tests)
  c.dfile_roll = roll_now;
  LOG_INF("inferzero: log file %s (earlier runs' next to it)\n", log.string().c_str());
}
}  // namespace slog
