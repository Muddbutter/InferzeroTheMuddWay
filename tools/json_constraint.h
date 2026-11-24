// Constrained JSON output (response_format json_object / json_schema): a prefix checker that tells whether a text is
// the beginning of a JSON value matching a schema. The sampler only draws tokens whose text keeps the answer such a
// prefix, and ends the answer once the value is complete, so the result always parses and fits the schema.
//
// Schema subset: type (one or a list), properties / required / additionalProperties, items / minItems / maxItems,
// enum, const, anyOf / oneOf (first branch that fits), $ref to #/$defs/... or #/definitions/..., {} (any value).
// With "properties" and no additionalProperties, other keys are not allowed (as OpenAI's strict mode requires).
#pragma once
#include "../src/json.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace jcon {

enum Res { FAIL, PARTIAL, DONE };

// whether an object with these properties (any_key: other keys allowed too) can take a key besides the `seen` ones
inline bool more_keys(const Json& props, bool any_key, const std::vector<std::string>& seen) {
  if (any_key) return true;
  for (const auto& [name, v] : props.entries()) {
    (void)v;
    if (std::find(seen.begin(), seen.end(), name) == seen.end()) return true;
  }
  return false;
}
// whether the schema allows type t ("number" includes "integer"; no type: anything)
inline bool has_type(const Json& sc, const char* t) {
  const Json& ty = sc["type"];
  auto is = [&](const std::string& x) { return x == t || (std::string(t) == "number" && x == "integer"); };
  if (ty.is_string()) return is(ty.str());
  if (ty.is_array()) {
    for (const Json& x : ty.items())
      if (is(x.str())) return true;
    return false;
  }
  return true;
}
// whether the schema's numbers must be integers (type "integer", or a list with "integer" and no "number")
inline bool integer_only(const Json& sc) {
  const Json& ty = sc["type"];
  if (ty.is_string()) return ty.str() == "integer";
  if (!ty.is_array()) return false;
  bool num = false, in = false;
  for (const Json& x : ty.items()) { num = num || x.str() == "number"; in = in || x.str() == "integer"; }
  return in && !num;
}
// code point cp as UTF-8 onto o
inline void put_utf8(std::string& o, uint32_t cp) {
  if (cp < 0x80) {
    o += (char)cp;
  } else if (cp < 0x800) {
    o += (char)(0xC0 | (cp >> 6));
    o += (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    o += (char)(0xE0 | (cp >> 12));
    o += (char)(0x80 | ((cp >> 6) & 0x3F));
    o += (char)(0x80 | (cp & 0x3F));
  } else {
    o += (char)(0xF0 | (cp >> 18));
    o += (char)(0x80 | ((cp >> 12) & 0x3F));
    o += (char)(0x80 | ((cp >> 6) & 0x3F));
    o += (char)(0x80 | (cp & 0x3F));
  }
}
// the code point at byte p of UTF-8 text t (a broken sequence: its first byte as it is)
inline uint32_t utf8_at(const std::string& t, size_t p) {
  const unsigned char c = (unsigned char)t[p];
  const int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
  if (n == 0 || p + n >= t.size()) return c;
  uint32_t cp = c & (0x3F >> n);
  for (int k = 1; k <= n; ++k) {
    const unsigned char d = (unsigned char)t[p + k];
    if ((d & 0xC0) != 0x80) return c;
    cp = (cp << 6) | (d & 0x3F);
  }
  return cp;
}
// whether `pend`, an escape not finished yet (from its backslash: "\", "\u00", "\ud83d\ud"), can still become the
// character at byte p of `name`: one of the ways JSON writes it (\uXXXX in either case, a surrogate pair above
// U+FFFF, or its short form such as \n)
// (a printable ASCII character other than " and \: a \u escape of it is refused, see Checker::string)
inline bool plain_ascii(uint32_t cp) { return cp >= 0x20 && cp < 0x7F && cp != '"' && cp != '\\'; }
inline bool escape_can_be(const std::string& pend, const std::string& name, size_t p) {
  if (p >= name.size()) return false;
  const uint32_t cp = utf8_at(name, p);
  std::vector<std::string> forms;
  char b[16];
  if (cp >= 0x10000) {
    const uint32_t v = cp - 0x10000;
    snprintf(b, sizeof b, "\\u%04x\\u%04x", (unsigned)(0xD800 + (v >> 10)), (unsigned)(0xDC00 + (v & 0x3FF)));
  } else {
    snprintf(b, sizeof b, "\\u%04x", (unsigned)cp);
  }
  if (!plain_ascii(cp)) forms.push_back(b);
  const char* sh = cp == '"' ? "\\\"" : cp == '\\' ? "\\\\" : cp == '/' ? "\\/" : cp == '\b' ? "\\b" : cp == '\f' ? "\\f"
                 : cp == '\n' ? "\\n" : cp == '\r' ? "\\r" : cp == '\t' ? "\\t" : nullptr;
  if (sh) forms.push_back(sh);
  for (const std::string& fm : forms) {
    if (pend.size() > fm.size()) continue;
    bool ok = true;
    for (size_t k = 0; k < pend.size() && ok; ++k) ok = tolower((unsigned char)pend[k]) == tolower((unsigned char)fm[k]);
    if (ok) return true;
  }
  return false;
}
// whether a string being written (its decoded text so far `got`, and `pend`, an unfinished escape or "") can still
// become `name`
inline bool can_become(const std::string& got, const std::string& pend, const std::string& name) {
  if (name.compare(0, got.size(), got) != 0) return false;
  return pend.empty() || escape_can_be(pend, name, got.size());
}
// the literal t from i: DONE when all of it is there, PARTIAL when s ends inside it
inline Res literal(const std::string& s, size_t& i, const char* t) {
  for (const char* p = t; *p; ++p, ++i) {
    if (i >= s.size()) return PARTIAL;
    if (s[i] != *p) return FAIL;
  }
  return DONE;
}

class Checker {
 public:
  // schema: the JSON schema ({} = any value); object_only: the value must be an object (json_object mode)
  Checker(Json schema, bool object_only) : root_(std::move(schema)), object_only_(object_only) {}

  // How `s` (the answer so far) relates to the schema: FAIL (no continuation fits), PARTIAL (it can still grow
  // into a match), DONE (a complete match, with at most trailing whitespace).
  Res check(const std::string& s) const {
    // (the memo below only for a text that starts with the prefix it was made on)
    memo_live_ = base_ > 0 && s.size() >= base_ && s.compare(0, base_, prefix_) == 0;
    size_t i = 0;
    if (!ws(s, i)) return FAIL;  // (the 8-character limit before the value too: no endless leading whitespace)
    if (i == s.size()) return PARTIAL;
    if (object_only_ && s[i] != '{') return FAIL;
    const Res r = value(root_, s, i, 0);
    if (r != DONE) return r;
    size_t j = i;
    ws(s, j);
    return j == s.size() && j - i <= 2 ? DONE : FAIL;  // (at most a little whitespace after the value)
  }
  // The text every check until the next call starts with: the answer so far (#80, 2026-11-22). A value of the schema
  // that the parser completed inside it, at a position and a schema node, ends where it ended for every text that
  // starts with it, so it is kept (memo_) and a later check skips it: a check then costs the member being written,
  // not the whole answer, and nothing it decides changes. The answer only grows from step to step, so the memo stays;
  // a prefix that does not extend the last one (another answer) clears it
  void set_prefix(const std::string& p) {
    if (p.size() < prefix_.size() || p.compare(0, prefix_.size(), prefix_) != 0) memo_.clear();
    prefix_ = p;
    base_ = p.size();
  }

 private:
  Json root_;
  bool object_only_;
  struct MemoKey {
    const Json* sc;
    size_t i;
    bool operator==(const MemoKey& o) const { return sc == o.sc && i == o.i; }
  };
  struct MemoHash {
    size_t operator()(const MemoKey& k) const { return std::hash<const void*>()(k.sc) ^ (k.i * 0x9E3779B97F4A7C15ull); }
  };
  mutable std::unordered_map<MemoKey, size_t, MemoHash> memo_;  // (schema node, start) -> end of a completed value
  struct EnumOpts {
    std::vector<std::string> opts, strs;
    bool made = false;
  };
  mutable std::unordered_map<const Json*, EnumOpts> enum_opts_;  // (an enum / const node's options, made once)
  std::string prefix_;
  size_t base_ = 0;
  mutable bool memo_live_ = false;

  const Json& resolve(const Json& sc) const {
    const std::string ref = sc["$ref"].str();
    if (ref.empty()) return sc;
    for (const char* pre : {"#/$defs/", "#/definitions/"}) {
      const std::string p = pre;
      if (ref.compare(0, p.size(), p) == 0) {
        const Json& defs = root_[p.substr(2, p.size() - 3)];
        return defs[ref.substr(p.size())];
      }
    }
    return root_;  // ("#": the root)
  }
  // whitespace: at most 8 characters in a row (a model could otherwise pad forever)
  static bool ws(const std::string& s, size_t& i) {
    size_t n = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) { ++i; ++n; }
    return n <= 8;
  }

  // value_at with the memo: a value completed before the prefix's end (its last character read, a number's end
  // included, inside the prefix) is the same for every text with that prefix
  Res value(const Json& sc0, const std::string& s, size_t& i, int depth) const {
    if (!memo_live_ || i >= base_) return value_at(sc0, s, i, depth);
    const auto it = memo_.find({&sc0, i});
    if (it != memo_.end()) {
      i = it->second;
      return DONE;
    }
    const size_t i0 = i;
    const Res r = value_at(sc0, s, i, depth);
    if (r == DONE && i < base_) memo_.emplace(MemoKey{&sc0, i0}, i);
    return r;
  }
  Res value_at(const Json& sc0, const std::string& s, size_t& i, int depth) const {
    if (depth > 32) return FAIL;
    const Json& sc = resolve(sc0);
    // anyOf / oneOf: the first branch that is not FAIL from here
    for (const char* k : {"anyOf", "oneOf"}) {
      if (!sc[k].is_array()) continue;
      Res best = FAIL;
      size_t bi = i;
      for (const Json& br : sc[k].items()) {
        size_t j = i;
        const Res r = value(br, s, j, depth + 1);
        if (r == DONE) { i = j; return DONE; }
        if (r == PARTIAL && best == FAIL) { best = PARTIAL; bi = j; }
      }
      i = bi;
      return best;
    }
    if (i >= s.size()) return PARTIAL;
    const char c = s[i];
    // enum / const: the value's text must be (a prefix of) one of the allowed values' JSON text
    if (sc["enum"].is_array() || sc.has("const")) {
      // (strs: the string options' own text, compared decoded, #80; both lists made once per schema node, not for
      //  every check: a 600-value enum was 600 strings built for each candidate token)
      EnumOpts& eo = enum_opts_[&sc];
      if (!eo.made) {
        auto add = [&](const Json& e) {
          if (e.is_string()) eo.strs.push_back(e.str());
          else eo.opts.push_back(e.dump());
        };
        if (sc.has("const")) add(sc["const"]);
        for (const Json& e : sc["enum"].items()) add(e);
        eo.made = true;
      }
      const std::vector<std::string>& opts = eo.opts;
      const std::vector<std::string>& strs = eo.strs;
      if (c == '"' && !strs.empty()) {
        size_t j = i;
        std::string got, pend;
        const Res r = string(s, j, &got, &pend);
        if (r == FAIL) return FAIL;
        if (r == DONE) {
          if (std::find(strs.begin(), strs.end(), got) == strs.end()) return FAIL;
          i = j;
          return DONE;
        }
        for (const std::string& o : strs)
          if (can_become(got, pend, o)) { i = s.size(); return PARTIAL; }
        return FAIL;
      }
      // (the rest of the text compared in place: a copy of it was made for every check)
      const size_t n = s.size() - i;
      bool partial = false;
      for (const std::string& o : opts) {
        if (n >= o.size() && s.compare(i, o.size(), o) == 0) { i += o.size(); return DONE; }
        if (n <= o.size() && o.compare(0, n, s, i, n) == 0) partial = true;
      }
      if (partial) { i = s.size(); return PARTIAL; }
      return FAIL;
    }
    if (c == '{') return has_type(sc, "object") ? object(sc, s, i, depth) : FAIL;
    if (c == '[') return has_type(sc, "array") ? array(sc, s, i, depth) : FAIL;
    if (c == '"') return has_type(sc, "string") ? string(s, i, nullptr) : FAIL;
    if (c == '-' || (c >= '0' && c <= '9')) return has_type(sc, "number") ? number(s, i, integer_only(sc)) : FAIL;
    if (c == 't') return has_type(sc, "boolean") ? literal(s, i, "true") : FAIL;
    if (c == 'f') return has_type(sc, "boolean") ? literal(s, i, "false") : FAIL;
    if (c == 'n') return has_type(sc, "null") ? literal(s, i, "null") : FAIL;
    return FAIL;
  }

  static Res number(const std::string& s, size_t& i, bool integer) {
    if (i < s.size() && s[i] == '-') ++i;
    if (i >= s.size()) return PARTIAL;
    if (s[i] == '0') ++i;
    else if (s[i] >= '1' && s[i] <= '9') { while (i < s.size() && isdigit((unsigned char)s[i])) ++i; }
    else return FAIL;
    if (i >= s.size()) return PARTIAL;
    if (s[i] == '.') {
      if (integer) return FAIL;
      ++i;
      if (i >= s.size()) return PARTIAL;
      if (!isdigit((unsigned char)s[i])) return FAIL;
      while (i < s.size() && isdigit((unsigned char)s[i])) ++i;
      if (i >= s.size()) return PARTIAL;
    }
    if (s[i] == 'e' || s[i] == 'E') {
      if (integer) return FAIL;
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
      if (i >= s.size()) return PARTIAL;
      if (!isdigit((unsigned char)s[i])) return FAIL;
      while (i < s.size() && isdigit((unsigned char)s[i])) ++i;
      if (i >= s.size()) return PARTIAL;
    }
    return DONE;
  }
  // the 4 hex digits of a \u escape at s[p..]: their value, -1 when s ends first, -2 when one is not a hex digit
  static long hex4(const std::string& s, size_t p) {
    long v = 0;
    for (size_t k = 0; k < 4; ++k) {
      if (p + k >= s.size()) return -1;
      const int c = tolower((unsigned char)s[p + k]);
      if (!isxdigit(c)) return -2;
      v = v * 16 + (c <= '9' ? c - '0' : c - 'a' + 10);
    }
    return v;
  }
  // a string; its decoded text (UTF-8, \u escapes and surrogate pairs decoded: #80, 2026-11-22) goes to `out` when
  // given, and on PARTIAL inside an escape, the escape's text so far to `pend` (else it is left empty)
  static Res string(const std::string& s, size_t& i, std::string* out, std::string* pend = nullptr) {
    if (pend) pend->clear();
    if (s[i] != '"') return FAIL;
    ++i;
    auto partial = [&](size_t from) {
      if (pend) *pend = s.substr(from);
      i = s.size();
      return PARTIAL;
    };
    while (i < s.size()) {
      const unsigned char c = (unsigned char)s[i];
      if (c == '"') { ++i; return DONE; }
      if (c < 0x20) return FAIL;
      if (c == '\\') {
        if (i + 1 >= s.size()) return partial(i);
        const char e = s[i + 1];
        if (e == 'u') {
          const long v = hex4(s, i + 2);
          if (v == -2) return FAIL;
          if (v == -1) return partial(i);
          uint32_t cp = (uint32_t)v;
          size_t len = 6;
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // (a high surrogate: its low one follows as \uDC00..\uDFFF)
            if (i + 6 >= s.size()) return partial(i);
            if (s[i + 6] == '\\') {
              if (i + 7 >= s.size()) return partial(i);
              if (s[i + 7] == 'u') {
                const long lo = hex4(s, i + 8);
                if (lo == -2) return FAIL;
                if (lo == -1) return partial(i);
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                  cp = 0x10000 + ((cp - 0xD800) << 10) + ((uint32_t)lo - 0xDC00);
                  len = 12;
                } else {
                  cp = 0xFFFD;
                }
              } else {
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;  // (a lone surrogate: the replacement character, as a decoder gives)
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          } else if (plain_ascii(cp)) {
            // (a printable ASCII character needs no escape: refused, so that a model whose next word does not fit
            //  writes the letter itself, not \u0074 after \u0074 (the 4B on #80's schema did))
            return FAIL;
          }
          if (out) put_utf8(*out, cp);
          i += len;
          continue;
        }
        if (!strchr("\"\\/bfnrt", e)) return FAIL;
        if (out) *out += e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
        i += 2;
        continue;
      }
      if (out) *out += (char)c;
      ++i;
    }
    return PARTIAL;
  }
  Res object(const Json& sc, const std::string& s, size_t& i, int depth) const {
    ++i;  // '{'
    const Json& props = sc["properties"];
    const Json& addl = sc["additionalProperties"];
    // other keys: allowed with additionalProperties true or a schema, or when no properties are listed
    const bool any_key = addl.is_object() || addl.as_bool(false) || (!props.is_object() && !addl.is_bool());
    std::vector<std::string> seen;
    bool first = true;
    while (true) {
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      if (s[i] == '}') {
        for (const Json& r : sc["required"].items())
          if (std::find(seen.begin(), seen.end(), r.str()) == seen.end()) return FAIL;
        ++i;
        return DONE;
      }
      if (!first) {
        if (s[i] != ',') return FAIL;
        if (!more_keys(props, any_key, seen)) return FAIL;  // (a comma only where another key can follow)
        ++i;
        if (!ws(s, i)) return FAIL;
        if (i >= s.size()) return PARTIAL;
      }
      first = false;
      if (s[i] != '"') return FAIL;
      std::string key, pend;
      const Res kr = string(s, i, &key, &pend);
      if (kr == FAIL) return FAIL;
      // the key: one of the listed properties not used yet (one it can still become while it is being written, an
      // unfinished escape included: "message_\u00" is not on its way to "message_kind", #80)
      auto allowed = [&](const std::string& k, bool prefix) {
        if (std::find(seen.begin(), seen.end(), k) != seen.end() && !prefix) return false;
        if (any_key) return true;
        for (const auto& [name, v] : props.entries()) {
          (void)v;
          if (std::find(seen.begin(), seen.end(), name) != seen.end()) continue;
          if (prefix ? can_become(k, pend, name) : name == k) return true;
        }
        return false;
      };
      if (kr == PARTIAL) return allowed(key, true) ? PARTIAL : FAIL;
      if (!allowed(key, false)) return FAIL;
      seen.push_back(key);
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      if (s[i] != ':') return FAIL;
      ++i;
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      static const Json any = Json::object();
      const Json& vs = props[key].is_null() ? (addl.is_object() ? addl : any) : props[key];
      const Res vr = value(vs, s, i, depth + 1);
      if (vr != DONE) return vr;
    }
  }
  Res array(const Json& sc, const std::string& s, size_t& i, int depth) const {
    ++i;  // '['
    static const Json any = Json::object();
    const Json& items = sc["items"].is_object() ? sc["items"] : any;
    const int64_t mn = sc["minItems"].as_int(0), mx = sc["maxItems"].as_int(1 << 30);
    int64_t n = 0;
    while (true) {
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      if (s[i] == ']') {
        if (n < mn) return FAIL;
        ++i;
        return DONE;
      }
      if (n > 0) {
        if (s[i] != ',') return FAIL;
        ++i;
        if (!ws(s, i)) return FAIL;
        if (i >= s.size()) return PARTIAL;
      }
      if (n >= mx) return FAIL;
      const Res r = value(items, s, i, depth + 1);
      if (r != DONE) return r;
      ++n;
    }
  }
};

}  // namespace jcon
