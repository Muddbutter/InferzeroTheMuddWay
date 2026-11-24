// Inkling 2's tokenizer split (GPT-4o's pre-tokenizer, src/tokenizer.h pretokenize_gpt4o) and chat format
// (src/chat_inkling.h): the split against the regex worked by hand, the rendered prompts against the GGUF template
// worked by hand. The token ids against llama.cpp's (llama-tokenize on the real file) are a later check.
#include "../src/chat_inkling.h"
#include "../src/tokenizer.h"
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
static void check(bool ok, const std::string& what) {
  if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++fails; }
}
static std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& w : v) s += "[" + w + "]";
  return s;
}
static void split(const std::string& text, const std::vector<std::string>& want) {
  const auto got = Tokenizer::split_gpt4o(text);
  check(got == want, "split \"" + text + "\": " + join(got) + ", want " + join(want));
}

int main() {
  split("Hello world", {"Hello", " world"});
  split("HELLOworld", {"HELLOworld"});        // (U* then Lo+: one word)
  split("ABC def", {"ABC", " def"});          // (A fails, B: U+)
  split("don't stop", {"don't", " stop"});    // (the contraction stays on its word)
  split("12345", {"123", "45"});
  split("a  b", {"a", " ", " b"});            // (\s+(?!\S) leaves the last space to the word)
  split("hi\n\nthere", {"hi", "\n\n", "there"});
  split("foo/bar", {"foo", "/bar"});          // (a leading non-letter joins the word)
  split("x = 1;", {"x", " =", " ", "1", ";"});
  split("\xC3\x9Cn\xC3\xAF" "code", {"\xC3\x9Cn\xC3\xAF" "code"});  // (non-ASCII letters: upper and lower at once)
  split("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E", {"\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"});
  split("end.\n", {"end", ".\n"});            // (punctuation takes its trailing newline)

  // long runs of one class (200 KB): each must split in linear time (a quadratic case in the U* backtrack would take
  // minutes here), and as the regex does
  {
    const size_t L = 200000;
    const auto sp = Tokenizer::split_gpt4o(std::string(L, ' '));
    check(sp.size() == 1 && sp[0].size() == L, "split: 200 KB of spaces is one word");
    const auto dg = Tokenizer::split_gpt4o(std::string(L - 1, '7'));  // (199,999: 66,666 groups of 3, then 1)
    check(dg.size() == (L - 1 + 2) / 3 && dg.back().size() == 1, "split: 200 KB of digits in groups of 3");
    const auto up = Tokenizer::split_gpt4o(std::string(L, 'A'));
    check(up.size() == 1, "split: 200 KB of capitals is one word");
    std::string mixed;
    for (size_t i = 0; i < L / 4; ++i) mixed += "AB c";
    check(Tokenizer::split_gpt4o(mixed).size() == L / 2, "split: 200 KB of \"AB c\" is two words per repeat");
  }

  // the chat format
  Json msgs = Json::parse(R"([{"role":"system","content":"S"},{"role":"user","content":"Hi"},
    {"role":"assistant","content":"Hello","reasoning_content":"think"},{"role":"user","content":"Q"}])");
  check(inkling::render(msgs, Json()) ==
            "<|message_system|><|content_text|>S<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>"
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>Hello<|end_message|><|content_model_end_sampling|>"
            "<|message_user|><|content_text|>Q<|end_message|>",
        "render: system, effort, turns");
  check(inkling::render(Json::parse(R"([{"role":"user","content":"x"}])"), Json(), Json("low")) ==
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
            "<|message_user|><|content_text|>x<|end_message|>",
        "render: effort low");
  check(inkling::render(Json::parse(R"([{"role":"user","content":"x"}])"), Json(), Json("none")).find("level: 0<|") !=
            std::string::npos,
        "render: effort none is 0");
  check(inkling::render(Json::parse(R"([{"role":"system","content":"S"}])"), Json()) ==
            "<|message_system|><|content_text|>S<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
        "render: effort at the end when every message is a system one");
  Json tools = Json::parse(R"([{"type":"function","function":{"name":"get","description":"d",
    "parameters":{"type":"object","properties":{"b":{"type":"string"},"a":{"type":"integer"}}}}}])");
  Json calls = Json::parse(R"([{"role":"user","content":"q"},
    {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get","arguments":"{\"z\":1,\"a\":\"x\"}"}}]},
    {"role":"tool","tool_call_id":"c1","content":"r"}])");
  check(inkling::render(calls, tools) ==
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"d\",\"name\":\"get\",\"parameters\":"
            "{\"properties\":{\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"string\"}},\"type\":\"object\"},"
            "\"type\":\"function\"}]<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>q<|end_message|>"
            "<|message_model|>get<|content_invoke_tool_json|>{\"name\":\"get\",\"args\":{\"a\":\"x\",\"z\":1}}<|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_tool|>get<|content_text|>r<|end_message|>",
        "render: tools declared sorted, a call with sorted arguments, the tool's name from its call");
  bool threw = false;
  try { inkling::render(Json::parse(R"([{"role":"user","content":"x"}])"), Json(), Json("huge")); } catch (...) { threw = true; }
  check(threw, "render: an unknown effort raises");

  const auto p = inkling::parse("<|message_model|><|content_thinking|>T<|end_message|><|message_model|><|content_text|>A"
                                "<|end_message|><|content_model_end_sampling|>");
  check(p.reasoning == "T" && p.content == "A" && p.ended && p.tool_calls.empty(), "parse: thinking and text");
  const auto q = inkling::parse("<|message_model|>get<|content_invoke_tool_json|>{\"name\":\"get\",\"args\":{\"a\":1}}"
                                "<|end_message|><|content_model_end_sampling|>");
  check(q.tool_calls.size() == 1 && q.tool_calls[0].name == "get" && q.tool_calls[0].arguments["a"].as_int() == 1 &&
            q.content.empty(),
        "parse: a tool call");
  const auto r = inkling::parse("<|message_model|><|content_text|>part");
  check(r.content == "part" && !r.ended, "parse: an unfinished message");
  bool ok = true;
  inkling::Parsed s;
  try {  // (a stream stopped inside a call: the partial JSON kept as content, no exception)
    s = inkling::parse("<|message_model|><|content_thinking|>T<|end_message|><|message_model|>get"
                       "<|content_invoke_tool_json|>{\"name\":\"get\",\"args\":{\"a\"");
  } catch (...) { ok = false; }
  check(ok && s.reasoning == "T" && s.tool_calls.empty() && s.content == "{\"name\":\"get\",\"args\":{\"a\"" && !s.ended,
        "parse: a stream stopped inside a tool call");
  const auto u = inkling::parse("<|message_model|>get<|content_invo");
  check(u.content.empty() && u.tool_calls.empty(), "parse: a stream stopped inside a header");

  {  // the server's parser fed piece by piece (markers whole, as single tokens): the deltas add up to the final split,
     // nothing of a call is content while it is written, the call comes whole once its message ends
    const std::vector<std::string> pieces = {"<|message_model|>", "<|content_thinking|>", "Th", "ink", "<|end_message|>",
                                             "<|message_model|>", "<|content_text|>", "Ans", "wer", "<|end_message|>",
                                             "<|message_model|>", "get", "<|content_invoke_tool_json|>", "{\"name\":\"get\",",
                                             "\"args\":{\"a\":1}}", "<|end_message|>"};
    inkling::OutputParser op;
    std::string text, dr, dc, all_r, all_c;
    bool call_as_content = false;
    size_t calls_before_end = 0;
    for (size_t i = 0; i < pieces.size(); ++i) {
      text += pieces[i];
      op.update(text, false, dr, dc);
      all_r += dr;
      all_c += dc;
      call_as_content = call_as_content || dc.find("name") != std::string::npos;
      if (i + 1 < pieces.size()) calls_before_end += op.partial_tool_calls(text, false).size();
    }
    op.update(text, true, dr, dc);
    all_r += dr;
    all_c += dc;
    const auto calls = op.tool_calls(text);
    const auto pc = op.partial_tool_calls(text, true);
    check(all_r == "Think" && all_c == "Answer" && !call_as_content && calls_before_end == 0 && calls.size() == 1 &&
              calls[0].name == "get" && pc.size() == 1 && pc[0].done && pc[0].args == "{\"a\":1}",
          "OutputParser: streamed thinking, text and a tool call");
    inkling::OutputParser op2;  // (a stream stopped inside a call: its text is content at the end)
    const std::string cut = "<|message_model|>get<|content_invoke_tool_json|>{\"name\":\"get\",\"ar";
    op2.update(cut, false, dr, dc);
    const bool none_yet = dc.empty();
    op2.update(cut, true, dr, dc);
    check(none_yet && dc == "{\"name\":\"get\",\"ar", "OutputParser: an unfinished call becomes content at the end");
  }

  if (fails) { std::printf("test_inkling_chat: %d failed\n", fails); return 1; }
  std::printf("test_inkling_chat: all passed\n");
  return 0;
}
