cmake_minimum_required(VERSION 3.24)
project(search_engine_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SE_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
option(SE_BUILD_TESTS "Build the GoogleTest suite" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# The project standard is -O2 for release builds (CMake's default is -O3).
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

if(SE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(secore STATIC
  src/varint.cpp
  src/json.cpp
  src/porter.cpp
  src/tokenizer.cpp
  src/corpus.cpp
  src/index_builder.cpp
  src/index.cpp
  src/query.cpp
  src/searcher.cpp
  src/metrics.cpp
)
target_include_directories(secore PUBLIC include)

find_package(Threads REQUIRED)

foreach(tool se_index se_search se_eval se_bench se_stem)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE secore Threads::Threads)
endforeach()

if(SE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(se_tests
    tests/test_varint.cpp
    tests/test_json.cpp
    tests/test_tokenizer.cpp
    tests/test_index.cpp
    tests/test_query.cpp
    tests/test_search.cpp
    tests/test_metrics.cpp
  )
  target_link_libraries(se_tests PRIVATE secore GTest::gtest_main)
  include(GoogleTest)
  # PRE_TEST: list tests at ctest time, not as a post-build step (the post-build
  # listing timed out under ASan on a heavily loaded machine).
  gtest_discover_tests(se_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 120)
endif()
