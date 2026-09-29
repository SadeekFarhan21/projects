cmake_minimum_required(VERSION 3.24)
project(llt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The spec for this project is -O2 for the measured build.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

option(LLT_SANITIZE "Build with -fsanitize=address,undefined" OFF)
option(LLT_TSAN "Build with -fsanitize=thread" OFF)
option(LLT_UBSAN_ONLY "Build with -fsanitize=undefined only" OFF)
option(LLT_BUILD_TESTS "Build tests" ON)
option(LLT_BUILD_BENCH "Build benchmarks" ON)

set(LLT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wno-gnu-zero-variadic-macro-arguments)
if(LLT_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()
if(LLT_UBSAN_ONLY)
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=undefined)
endif()
if(LLT_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# Homebrew LLVM (used by the sanitizer presets) emits DWARF 5 that Apple's
# linker cannot index; DWARF 4 keeps ld quiet and symbolization working.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND APPLE)
  add_compile_options(-gdwarf-4)
endif()

find_package(Threads REQUIRED)

add_library(llt STATIC
  src/capture.cpp
  src/stats.cpp
  src/synth.cpp
  src/thread_util.cpp)
target_include_directories(llt PUBLIC include)
target_compile_options(llt PRIVATE ${LLT_WARNINGS})
target_link_libraries(llt PUBLIC Threads::Threads)

add_executable(itchgen tools/itchgen.cpp)
target_link_libraries(itchgen PRIVATE llt)
target_compile_options(itchgen PRIVATE ${LLT_WARNINGS})

add_executable(trader tools/trader.cpp)
target_link_libraries(trader PRIVATE llt)
target_compile_options(trader PRIVATE ${LLT_WARNINGS})

if(LLT_BUILD_BENCH)
  add_executable(bench_micro bench/bench_micro.cpp)
  target_link_libraries(bench_micro PRIVATE llt)
  target_compile_options(bench_micro PRIVATE ${LLT_WARNINGS})

  add_executable(bench_pipeline bench/bench_pipeline.cpp)
  target_link_libraries(bench_pipeline PRIVATE llt)
  target_compile_options(bench_pipeline PRIVATE ${LLT_WARNINGS})
endif()

if(LLT_BUILD_TESTS)
  include(FetchContent)
  # doctest is header-only. SOURCE_SUBDIR points at a directory with no
  # CMakeLists.txt so only the sources are fetched (its own CMakeLists predates
  # CMake 4's minimum-version policy).
  FetchContent_Declare(doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
    URL_HASH SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186
    SOURCE_SUBDIR do-not-configure)
  FetchContent_MakeAvailable(doctest)

  enable_testing()
  add_executable(unit_tests
    tests/main.cpp
    tests/test_itch.cpp
    tests/test_queues.cpp
    tests/test_order_table.cpp
    tests/test_order_book.cpp
    tests/test_strategy_risk.cpp
    tests/test_pipeline.cpp)
  target_link_libraries(unit_tests PRIVATE llt)
  target_include_directories(unit_tests PRIVATE ${doctest_SOURCE_DIR}/doctest)
  # doctest's macros use __COUNTER__, which newer clang flags under -Wpedantic.
  set(LLT_TEST_WARNINGS ${LLT_WARNINGS} -Wno-unknown-warning-option -Wno-c2y-extensions)
  target_compile_options(unit_tests PRIVATE ${LLT_TEST_WARNINGS})
  add_test(NAME unit_tests COMMAND unit_tests)

  # Separate binary because it replaces global operator new.
  add_executable(hotpath_alloc_test tests/test_hotpath_alloc.cpp)
  target_link_libraries(hotpath_alloc_test PRIVATE llt)
  target_include_directories(hotpath_alloc_test PRIVATE ${doctest_SOURCE_DIR}/doctest)
  target_compile_options(hotpath_alloc_test PRIVATE ${LLT_TEST_WARNINGS})
  add_test(NAME hotpath_alloc_test COMMAND hotpath_alloc_test)
endif()
