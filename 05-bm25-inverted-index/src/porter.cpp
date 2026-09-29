// Porter stemmer, a direct port of the reference ANSI C version
// (tartarus.org/martin/PorterStemmer). b[0..k] is the word being stemmed,
// j is a general offset into it set by ends().
#include "se/porter.hpp"

#include <cstring>

namespace se {
namespace {

struct Stemmer {
  std::string& b;
  int k = 0;
  int j = 0;

  bool cons(int i) const {
    switch (b[i]) {
      case 'a': case 'e': case 'i': case 'o': case 'u': return false;
      case 'y': return i == 0 ? true : !cons(i - 1);
      default: return true;
    }
  }
  // m() counts VC sequences in b[0..j]: <c>(vc)^m<v>.
  int m() const {
    int n = 0, i = 0;
    for (;;) {
      if (i > j) return n;
      if (!cons(i)) break;
      ++i;
    }
    ++i;
    for (;;) {
      for (;;) {
        if (i > j) return n;
        if (cons(i)) break;
        ++i;
      }
      ++i;
      ++n;
      for (;;) {
        if (i > j) return n;
        if (!cons(i)) break;
        ++i;
      }
      ++i;
    }
  }
  bool vowel_in_stem() const {
    for (int i = 0; i <= j; ++i)
      if (!cons(i)) return true;
    return false;
  }
  bool doublec(int i) const {
    if (i < 1) return false;
    if (b[i] != b[i - 1]) return false;
    return cons(i);
  }
  // cvc(i) is true when b[i-2..i] is consonant-vowel-consonant and the last
  // consonant is not w, x or y (used to restore an e: hop(e), cav(e)).
  bool cvc(int i) const {
    if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
    char ch = b[i];
    return !(ch == 'w' || ch == 'x' || ch == 'y');
  }
  bool ends(const char* s) {
    int len = static_cast<int>(std::strlen(s));
    if (s[len - 1] != b[k]) return false;
    if (len > k + 1) return false;
    if (std::memcmp(b.data() + k - len + 1, s, static_cast<size_t>(len)) != 0) return false;
    j = k - len;
    return true;
  }
  void setto(const char* s) {
    int len = static_cast<int>(std::strlen(s));
    b.resize(static_cast<size_t>(j + 1));
    b.append(s);
    k = j + len;
  }
  void r(const char* s) {
    if (m() > 0) setto(s);
  }
  void truncate() { b.resize(static_cast<size_t>(k + 1)); }

  void step1ab() {
    if (b[k] == 's') {
      if (ends("sses")) k -= 2;
      else if (ends("ies")) setto("i");
      else if (b[k - 1] != 's') --k;
      truncate();
    }
    if (ends("eed")) {
      if (m() > 0) { --k; truncate(); }
    } else if ((ends("ed") || ends("ing")) && vowel_in_stem()) {
      k = j;
      truncate();
      if (ends("at")) setto("ate");
      else if (ends("bl")) setto("ble");
      else if (ends("iz")) setto("ize");
      else if (doublec(k)) {
        --k;
        char ch = b[k];
        if (ch == 'l' || ch == 's' || ch == 'z') ++k;
        truncate();
      } else if (m() == 1 && cvc(k)) {
        setto("e");
      }
    }
  }
  void step1c() {
    if (ends("y") && vowel_in_stem()) b[k] = 'i';
  }
  void step2() {
    if (k < 1) return;
    switch (b[k - 1]) {
      case 'a':
        if (ends("ational")) { r("ate"); break; }
        if (ends("tional")) { r("tion"); break; }
        break;
      case 'c':
        if (ends("enci")) { r("ence"); break; }
        if (ends("anci")) { r("ance"); break; }
        break;
      case 'e':
        if (ends("izer")) { r("ize"); break; }
        break;
      case 'l':
        if (ends("bli")) { r("ble"); break; }
        if (ends("alli")) { r("al"); break; }
        if (ends("entli")) { r("ent"); break; }
        if (ends("eli")) { r("e"); break; }
        if (ends("ousli")) { r("ous"); break; }
        break;
      case 'o':
        if (ends("ization")) { r("ize"); break; }
        if (ends("ation")) { r("ate"); break; }
        if (ends("ator")) { r("ate"); break; }
        break;
      case 's':
        if (ends("alism")) { r("al"); break; }
        if (ends("iveness")) { r("ive"); break; }
        if (ends("fulness")) { r("ful"); break; }
        if (ends("ousness")) { r("ous"); break; }
        break;
      case 't':
        if (ends("aliti")) { r("al"); break; }
        if (ends("iviti")) { r("ive"); break; }
        if (ends("biliti")) { r("ble"); break; }
        break;
      case 'g':
        if (ends("logi")) { r("log"); break; }
        break;
      default: break;
    }
  }
  void step3() {
    switch (b[k]) {
      case 'e':
        if (ends("icate")) { r("ic"); break; }
        if (ends("ative")) { r(""); break; }
        if (ends("alize")) { r("al"); break; }
        break;
      case 'i':
        if (ends("iciti")) { r("ic"); break; }
        break;
      case 'l':
        if (ends("ical")) { r("ic"); break; }
        if (ends("ful")) { r(""); break; }
        break;
      case 's':
        if (ends("ness")) { r(""); break; }
        break;
      default: break;
    }
  }
  void step4() {
    if (k < 1) return;
    switch (b[k - 1]) {
      case 'a': if (ends("al")) break; return;
      case 'c': if (ends("ance")) break; if (ends("ence")) break; return;
      case 'e': if (ends("er")) break; return;
      case 'i': if (ends("ic")) break; return;
      case 'l': if (ends("able")) break; if (ends("ible")) break; return;
      case 'n':
        if (ends("ant")) break;
        if (ends("ement")) break;
        if (ends("ment")) break;
        if (ends("ent")) break;
        return;
      case 'o':
        if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
        if (ends("ou")) break;
        return;
      case 's': if (ends("ism")) break; return;
      case 't': if (ends("ate")) break; if (ends("iti")) break; return;
      case 'u': if (ends("ous")) break; return;
      case 'v': if (ends("ive")) break; return;
      case 'z': if (ends("ize")) break; return;
      default: return;
    }
    if (m() > 1) { k = j; truncate(); }
  }
  void step5() {
    j = k;
    if (b[k] == 'e') {
      int a = m();
      if (a > 1 || (a == 1 && !cvc(k - 1))) { --k; truncate(); }
    }
    if (b[k] == 'l' && doublec(k) && m() > 1) { --k; truncate(); }
  }
};

}  // namespace

void porter_stem(std::string& word) {
  if (word.size() <= 2) return;
  for (char c : word)
    if (c < 'a' || c > 'z') return;
  Stemmer s{word};
  s.k = static_cast<int>(word.size()) - 1;
  s.step1ab();
  if (s.k > 0) {
    s.step1c();
    s.step2();
    s.step3();
    s.step4();
    s.step5();
  }
  word.resize(static_cast<size_t>(s.k + 1));
}

}  // namespace se
