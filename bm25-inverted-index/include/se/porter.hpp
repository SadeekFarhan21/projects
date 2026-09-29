#pragma once
#include <string>

namespace se {

// The original Porter (1980) stemmer, following Martin Porter's reference C
// implementation (including its "bli"/"logi" departures from the paper).
// Input must be lowercase ASCII letters; other input is returned unchanged.
void porter_stem(std::string& word);

}  // namespace se
