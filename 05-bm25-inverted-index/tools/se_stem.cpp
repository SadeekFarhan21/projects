// Reads one word per line on stdin, writes "word<TAB>stem" per line.
// Used by scripts/check_porter.py to diff our stemmer against NLTK's.
#include <iostream>
#include <string>

#include "se/porter.hpp"

int main() {
  std::string w;
  while (std::getline(std::cin, w)) {
    std::string s = w;
    se::porter_stem(s);
    std::cout << w << '\t' << s << '\n';
  }
  return 0;
}
