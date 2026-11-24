// Inkling 2's chat format, written out from the template in its GGUF (tokenizer.chat_template of
// unsloth/Inkling-2-Mini-GGUF): native, as chat.h is for Qwen, since the template uses what minja lacks
// ({% generation %}, tojson(sort_keys=..., separators=...), dictsort(case_sensitive=...)). Rendered the way Hugging Face
// renders it (its tojson: json.dumps with ensure_ascii=False, no HTML escaping), and model output parsed back into
// OpenAI-style reasoning / content / tool calls.
//
// The format: every message is ROLE[name]<|content_KIND|>BODY<|end_message|>, ROLE one of <|message_user|>,
// <|message_model|>, <|message_system|>, <|message_tool|>; an assistant turn ends with <|content_model_end_sampling|>
// (the end-of-generation token, 200006). Tools are declared first in a system message ("tool_declare"), and the
// thinking effort is a system message before the first non-system message (at the end when there is none). There is
// no generation prompt: the model writes its own <|message_model|> after the prompt.
#pragma once
#include "chat.h"
#include "json.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace inkling {

// Python's repr of a float (Jinja's {{ num }}): the shortest digits that read back the same; an exponent below 1e-4
inline std::string py_float(double v) {
  if (v == std::floor(v) && std::fabs(v) < 1e16) {
    char b[32];
    snprintf(b, sizeof b, "%.1f", v);
    return b;
  }
  char b[40];
  for (int p = 1; p <= 17; ++p) {
    snprintf(b, sizeof b, "%.*g", p, v);
    if (strtod(b, nullptr) == v) break;
  }
  return b;  // (%g writes 1e-05 and 1e+20 as Python's repr does)
}

// json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sort): the object keys sorted at every level
// when sort is set, in the given order otherwise (Json keeps the order it read)
inline void dump_compact(const Json& v, bool sort, std::string& out) {
  if (v.is_array()) {
    out.push_back('[');
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) out.push_back(',');
      dump_compact(v[i], sort, out);
    }
    out.push_back(']');
    return;
  }
  if (v.is_object()) {
    std::vector<const std::pair<std::string, Json>*> e;
    for (const auto& kv : v.entries()) e.push_back(&kv);
    if (sort) std::stable_sort(e.begin(), e.end(), [](auto* a, auto* b) { return a->first < b->first; });
    out.push_back('{');
    for (size_t i = 0; i < e.size(); ++i) {
      if (i) out.push_back(',');
      Json::escape(out, e[i]->first);
      out.push_back(':');
      dump_compact(e[i]->second, sort, out);
    }
    out.push_back('}');
    return;
  }
  out += v.dump();  // (strings: \uXXXX only below 0x20, as json.dumps with ensure_ascii=False)
}
inline std::string tojson(const Json& v, bool sort = false) {
  std::string s;
  dump_compact(v, sort, s);
  return s;
}

// Jinja truthiness of a message field: a non-empty string, array or object, true, a non-zero number
inline bool truthy(const Json& v) {
  if (v.is_string() || v.is_array() || v.is_object()) return v.size() > 0 || (v.is_string() && !v.as_string().empty());
  if (v.is_bool()) return v.as_bool();
  if (v.is_number()) return v.as_number() != 0;
  return false;
}

// the "Thinking effort level" message: a name of effort_map or a number in [0, 0.99]; none given: 0.9
inline std::string effort_message(const Json& effort) {
  double num = 0.9;
  if (effort.is_string()) {
    const std::string key = chat::trim(effort.as_string());
    static const std::pair<const char*, double> map[] = {{"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7},
                                                         {"high", 0.9}, {"xhigh", 0.99}, {"max", 0.99}};
    bool found = false;
    for (const auto& [k, x] : map)
      if (key == k) { num = x; found = true; }
    if (!found) throw std::runtime_error("Unknown reasoning_effort: " + effort.as_string());
  } else if (effort.is_number()) {
    num = effort.as_number();
  }
  if (num < 0.0 || num > 0.99) throw std::runtime_error("reasoning_effort must be in [0.0, 0.99]");
  return std::string("<|message_system|><|content_text|>Thinking effort level: ") + (num == 0.0 ? "0" : py_float(num)) +
         "<|end_message|>";
}

// The prompt for messages / tools in OpenAI form (tool-call arguments as objects or JSON strings of objects).
// effort: the request's reasoning_effort (null: the template's default 0.9).
inline std::string render(const Json& messages, const Json& tools, const Json& effort = Json()) {
  std::string out;
  if (tools.is_array() && tools.size() > 0) {
    Json specs = Json::array();
    for (const Json& tool : tools.items()) {
      const Json& fn = tool.has("function") ? tool["function"] : tool;
      Json spec = Json::object();
      spec.set("description", truthy(fn["description"]) ? fn["description"] : Json(""));
      spec.set("name", fn["name"]);
      spec.set("parameters", truthy(fn["parameters"]) ? fn["parameters"] : Json::object());
      spec.set("type", truthy(tool["type"]) ? tool["type"] : Json("function"));
      specs.push(spec);
    }
    out += "<|message_system|>tool_declare<|content_xml|>" + tojson(specs, true) + "<|end_message|>";
  }
  static const std::pair<const char*, const char*> roles[] = {{"user", "<|message_user|>"}, {"assistant", "<|message_model|>"},
                                                              {"system", "<|message_system|>"}, {"tool", "<|message_tool|>"}};
  bool effort_done = false;
  for (const Json& m : messages.items()) {
    const std::string role = m["role"].str();
    const char* rt = nullptr;
    for (const auto& [r, t] : roles)
      if (role == r) rt = t;
    if (!rt) throw std::runtime_error("Unknown message role: " + role);
    if (!effort_done && role != "system") {
      out += effort_message(effort);
      effort_done = true;
    }
    std::string rtok = rt;
    if (role == "tool") {
      if (truthy(m["name"])) rtok += m["name"].str();
      else if (truthy(m["tool_call_id"]))  // (every matching call of every assistant message, as the template loops)
        for (const Json& prev : messages.items()) {
          if (prev["role"].str() != "assistant" || !truthy(prev["tool_calls"])) continue;
          for (const Json& tc : prev["tool_calls"].items())
            if (tc["id"].is_string() && tc["id"].as_string() == m["tool_call_id"].str() && tc["function"]["name"].is_string())
              rtok += tc["function"]["name"].as_string();
        }
    } else if (role == "assistant" && truthy(m["name"])) {
      rtok += m["name"].str();
    }
    const bool asst = role == "assistant";
    const Json& content = m["content"];
    if (asst && truthy(m["reasoning_content"]))
      out += rtok + "<|content_thinking|>" + m["reasoning_content"].str() + "<|end_message|>";
    if (content.is_string() && (!content.as_string().empty() || !(asst && (truthy(m["tool_calls"]) || truthy(m["reasoning_content"]))))) {
      out += rtok + "<|content_text|>" + content.as_string() + "<|end_message|>";
    } else if (truthy(content) && content.is_array()) {
      for (const Json& part : content.items()) {
        const std::string ty = part.is_object() && part.has("type") ? part["type"].str() : "";
        if (part.is_string()) out += rtok + "<|content_text|>" + part.as_string() + "<|end_message|>";
        else if (!part.has("type") || ty == "text" || ty == "input_text")
          out += rtok + "<|content_text|>" + (part["text"].is_string() ? part["text"].as_string() : "") + "<|end_message|>";
        else if (ty == "image" || ty == "input_image" || ty == "image_url")
          // (the template's placeholder is <|content_image|><|unused_200054|>: the inkling2v projector is not built)
          throw std::runtime_error("images are not supported for Inkling 2 yet");
        else if (ty == "audio" || ty == "input_audio" || ty == "audio_url")
          throw std::runtime_error("Inkling 2 does not accept audio input");
        else throw std::runtime_error("Unsupported content part type: " + ty);
      }
    }
    if (asst && truthy(m["tool_calls"]))
      for (const Json& tc : m["tool_calls"].items()) {
        const Json& fn = tc["function"];
        if (!fn["name"].is_string()) throw std::runtime_error("tool call function name must be a string");
        const std::string name = fn["name"].as_string();
        if (truthy(m["name"]) && m["name"].str() != name)
          throw std::runtime_error("assistant name " + m["name"].str() + " does not match its tool call " + name);
        // (OpenAI clients send the arguments as a JSON string; the template wants the parsed object)
        const Json args = chat::arguments_object(tc);
        out += "<|message_model|>" + name + "<|content_invoke_tool_json|>{\"name\":" + tojson(Json(name)) + ",\"args\":{";
        std::vector<const std::pair<std::string, Json>*> e;
        for (const auto& kv : args.entries()) e.push_back(&kv);
        std::stable_sort(e.begin(), e.end(), [](auto* a, auto* b) { return a->first < b->first; });  // dictsort
        for (size_t i = 0; i < e.size(); ++i) {
          if (i) out += ",";
          out += tojson(Json(e[i]->first)) + ":" + tojson(e[i]->second);
        }
        out += "}}<|end_message|>";
      }
    if (asst) out += "<|content_model_end_sampling|>";
  }
  if (!effort_done) out += effort_message(effort);
  return out;
}

struct Parsed {
  std::string reasoning, content;
  std::vector<chat::ToolCall> tool_calls;
  bool ended = false;  // <|content_model_end_sampling|> seen
};

// Model output after the prompt (special tokens as their text): its messages <|message_model|>[name]<|content_KIND|>
// BODY<|end_message|>. Thinking bodies -> reasoning, text bodies -> content, <|content_invoke_tool_json|>
// {"name":..., "args":{...}} -> a tool call (a body that is not such an object is kept as content, so nothing is
// lost). <|content_invoke_tool_text|>: its body as the call's single "text" argument (the format is not documented).
// Text outside any message (a model that skipped its role token) counts as content.
// Streaming (final false): a tool call's message still being written gives nothing yet (its body is not content
// until it is known not to be a call), so every part only grows as the text grows.
inline Parsed parse(const std::string& s, bool final = true) {
  Parsed r;
  static const std::string kModel = "<|message_model|>", kEnd = "<|end_message|>", kStop = "<|content_model_end_sampling|>",
                           kThink = "<|content_thinking|>", kText = "<|content_text|>",
                           kToolJ = "<|content_invoke_tool_json|>", kToolT = "<|content_invoke_tool_text|>";
  size_t stop = s.find(kStop);
  if (stop != std::string::npos) r.ended = true;
  const std::string t = s.substr(0, stop);
  size_t i = 0;
  while (i < t.size()) {
    const size_t m = t.find(kModel, i);
    if (m != i) {  // text before the next message
      const std::string loose = t.substr(i, m == std::string::npos ? std::string::npos : m - i);
      r.content += loose;
      if (m == std::string::npos) break;
    }
    size_t h = m + kModel.size();
    // the content kind after an optional name (a tool name for a call)
    const size_t k = t.find("<|content_", h);
    if (k == std::string::npos) break;  // (a message still being written: its header so far)
    const std::string name = t.substr(h, k - h);
    const size_t ke = t.find("|>", k);
    if (ke == std::string::npos) break;
    const std::string kind = t.substr(k, ke + 2 - k);
    const size_t be = t.find(kEnd, ke + 2);
    const std::string body = t.substr(ke + 2, be == std::string::npos ? std::string::npos : be - ke - 2);
    if (be == std::string::npos && !final && (kind == kToolJ || kind == kToolT)) break;  // (a call still being written)
    if (kind == kThink) r.reasoning += body;
    else if (kind == kText) r.content += body;
    else if (kind == kToolJ) {
      Json j;
      if (Json::try_parse(chat::trim(body), j) && j.is_object() && (j["name"].is_string() || !name.empty())) {
        chat::ToolCall tc;
        tc.name = j["name"].is_string() ? j["name"].as_string() : name;
        tc.arguments = j["args"].is_object() ? j["args"] : Json::object();
        r.tool_calls.push_back(tc);
      } else {
        r.content += body;
      }
    } else if (kind == kToolT) {
      chat::ToolCall tc;
      tc.name = name;
      tc.arguments = Json::object();
      tc.arguments.set("text", body);
      r.tool_calls.push_back(tc);
    } else {
      r.content += body;  // (an unknown kind: its text kept)
    }
    i = be == std::string::npos ? t.size() : be + kEnd.size();
  }
  return r;
}

// The server's parser (chat::OutputParserBase) over parse(): the stream sends each part's new end. The format's
// markers are single tokens, so no text is held back for a marker still being written; a tool call comes whole
// once its message ends.
class OutputParser : public chat::OutputParserBase {
 public:
  void update(const std::string& text, bool final, std::string& d_reason, std::string& d_content) override {
    const Parsed p = parse(text, final);
    d_reason = p.reasoning.size() > sent_r_ ? p.reasoning.substr(sent_r_) : std::string();
    d_content = p.content.size() > sent_c_ ? p.content.substr(sent_c_) : std::string();
    sent_r_ = std::max(sent_r_, p.reasoning.size());
    sent_c_ = std::max(sent_c_, p.content.size());
  }
  std::string reasoning_text(const std::string& text) const override { return parse(text).reasoning; }
  std::string content_text(const std::string& text) const override { return parse(text).content; }
  std::vector<chat::ToolCall> tool_calls(const std::string& text) const override { return parse(text).tool_calls; }
  std::vector<chat::PartialToolCall> partial_tool_calls(const std::string& text, bool final) const override {
    std::vector<chat::PartialToolCall> v;
    for (const auto& tc : parse(text, final).tool_calls) {
      chat::PartialToolCall c;
      c.name = tc.name;
      c.args = tc.arguments.dump();
      c.done = true;
      v.push_back(std::move(c));
    }
    return v;
  }

 private:
  size_t sent_r_ = 0, sent_c_ = 0;
};

}  // namespace inkling
