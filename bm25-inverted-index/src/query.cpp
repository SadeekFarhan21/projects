#include "se/query.hpp"

#include <algorithm>

namespace se {
namespace {

struct Lexeme {
  enum class Type { Word, Phrase, LParen, RParen, And, Or, End } type;
  std::string text;
};

std::vector<Lexeme> lex(std::string_view q) {
  std::vector<Lexeme> out;
  size_t i = 0;
  auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  while (i < q.size()) {
    char c = q[i];
    if (is_space(c)) { ++i; continue; }
    if (c == '(') { out.push_back({Lexeme::Type::LParen, "("}); ++i; continue; }
    if (c == ')') { out.push_back({Lexeme::Type::RParen, ")"}); ++i; continue; }
    if (c == '"') {
      size_t j = q.find('"', i + 1);
      if (j == std::string_view::npos) throw QueryParseError("unbalanced quote");
      out.push_back({Lexeme::Type::Phrase, std::string(q.substr(i + 1, j - i - 1))});
      i = j + 1;
      continue;
    }
    size_t j = i;
    while (j < q.size() && !is_space(q[j]) && q[j] != '(' && q[j] != ')' && q[j] != '"') ++j;
    std::string w(q.substr(i, j - i));
    if (w == "AND") out.push_back({Lexeme::Type::And, w});
    else if (w == "OR") out.push_back({Lexeme::Type::Or, w});
    else out.push_back({Lexeme::Type::Word, w});
    i = j;
  }
  out.push_back({Lexeme::Type::End, ""});
  return out;
}

class Parser {
 public:
  Parser(std::vector<Lexeme> lx, const Tokenizer& tok) : lx_(std::move(lx)), tok_(tok) {}

  std::unique_ptr<QueryNode> parse() {
    auto n = expr();
    if (peek() != Lexeme::Type::End) throw QueryParseError("unexpected '" + lx_[i_].text + "'");
    return n;
  }

 private:
  using T = Lexeme::Type;
  T peek() const { return lx_[i_].type; }

  static bool starts_unary(T t) { return t == T::Word || t == T::Phrase || t == T::LParen; }

  // Builds an And/Or node, dropping empty children and collapsing singletons.
  static std::unique_ptr<QueryNode> combine(QueryNode::Kind k,
                                            std::vector<std::unique_ptr<QueryNode>> kids) {
    std::vector<std::unique_ptr<QueryNode>> keep;
    for (auto& c : kids) {
      if (!c) continue;
      if (c->kind == k) {  // flatten (AND a (AND b c)) into (AND a b c)
        for (auto& g : c->children) keep.push_back(std::move(g));
      } else {
        keep.push_back(std::move(c));
      }
    }
    if (keep.empty()) return nullptr;
    if (keep.size() == 1) return std::move(keep[0]);
    auto n = std::make_unique<QueryNode>();
    n->kind = k;
    n->children = std::move(keep);
    return n;
  }

  std::unique_ptr<QueryNode> expr() {
    std::vector<std::unique_ptr<QueryNode>> kids;
    kids.push_back(and_expr());
    while (peek() == T::Or) {
      ++i_;
      if (!starts_unary(peek())) throw QueryParseError("OR must be followed by a term");
      kids.push_back(and_expr());
    }
    return combine(QueryNode::Kind::Or, std::move(kids));
  }

  std::unique_ptr<QueryNode> and_expr() {
    std::vector<std::unique_ptr<QueryNode>> kids;
    if (!starts_unary(peek())) throw QueryParseError("expected a term");
    kids.push_back(unary());
    for (;;) {
      if (peek() == T::And) {
        ++i_;
        if (!starts_unary(peek())) throw QueryParseError("AND must be followed by a term");
        kids.push_back(unary());
      } else if (starts_unary(peek())) {
        kids.push_back(unary());
      } else {
        break;
      }
    }
    return combine(QueryNode::Kind::And, std::move(kids));
  }

  std::unique_ptr<QueryNode> unary() {
    const Lexeme& l = lx_[i_++];
    switch (l.type) {
      case T::LParen: {
        auto n = expr();
        if (peek() != T::RParen) throw QueryParseError("missing ')'");
        ++i_;
        return n;
      }
      case T::Word:
      case T::Phrase:
        return words(l.text);
      default:
        throw QueryParseError("unexpected '" + l.text + "'");
    }
  }

  std::unique_ptr<QueryNode> words(const std::string& text) {
    std::vector<Token> toks;
    tok_.tokenize(text, toks);
    if (toks.empty()) return nullptr;
    auto n = std::make_unique<QueryNode>();
    if (toks.size() == 1) {
      n->kind = QueryNode::Kind::Term;
      n->terms.push_back(toks[0].term);
      n->offsets.push_back(0);
      return n;
    }
    n->kind = QueryNode::Kind::Phrase;
    const uint32_t base = toks[0].pos;
    for (auto& t : toks) {
      n->terms.push_back(t.term);
      n->offsets.push_back(t.pos - base);
    }
    return n;
  }

  std::vector<Lexeme> lx_;
  const Tokenizer& tok_;
  size_t i_ = 0;
};

}  // namespace

std::unique_ptr<QueryNode> parse_boolean_query(std::string_view q, const Tokenizer& tok) {
  auto lx = lex(q);
  if (lx.size() == 1) return nullptr;  // only End
  return Parser(std::move(lx), tok).parse();
}

std::string to_string(const QueryNode& n) {
  switch (n.kind) {
    case QueryNode::Kind::Term:
      return n.terms[0];
    case QueryNode::Kind::Phrase: {
      std::string s = "\"";
      for (size_t i = 0; i < n.terms.size(); ++i) {
        if (i > 0) {
          for (uint32_t g = n.offsets[i - 1] + 1; g < n.offsets[i]; ++g) s += " _";
          s += ' ';
        }
        s += n.terms[i];
      }
      return s + "\"";
    }
    case QueryNode::Kind::And:
    case QueryNode::Kind::Or: {
      std::string s = n.kind == QueryNode::Kind::And ? "(AND" : "(OR";
      for (const auto& c : n.children) s += " " + to_string(*c);
      return s + ")";
    }
  }
  return {};
}

void collect_terms(const QueryNode& n, std::vector<std::string>& out) {
  for (const auto& t : n.terms)
    if (std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
  for (const auto& c : n.children) collect_terms(*c, out);
}

}  // namespace se
