cmake_minimum_required(VERSION 3.24)
project(exchange_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The project standard is -O2 for release numbers (CMake defaults to -O3).
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

option(EXCH_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(EXCH_BUILD_TESTS "Build unit and differential tests" ON)
option(EXCH_BUILD_BENCH "Build benchmarks" ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(EXCH_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # Homebrew LLVM emits DWARF 5 that Apple's ld warns about.
    add_compile_options(-gdwarf-4)
  endif()
endif()

set(EXCH_SOURCES src/engine.cpp src/events.cpp src/reference.cpp src/market_data.cpp)

add_library(exch_core ${EXCH_SOURCES})
target_include_directories(exch_core PUBLIC include)

add_executable(exch tools/exch.cpp)
target_link_libraries(exch PRIVATE exch_core)

if(EXCH_BUILD_BENCH)
  add_executable(bench_engine bench/bench_engine.cpp)
  target_link_libraries(bench_engine PRIVATE exch_core)

  # Ablation: identical engine, but the id -> slot index uses std::unordered_map.
  add_library(exch_core_stdmap ${EXCH_SOURCES})
  target_include_directories(exch_core_stdmap PUBLIC include)
  target_compile_definitions(exch_core_stdmap PUBLIC EXCH_STD_UNORDERED_MAP)
  add_executable(bench_engine_stdmap bench/bench_engine.cpp)
  target_link_libraries(bench_engine_stdmap PRIVATE exch_core_stdmap)

  # Ablation: IdMap with an identity hash instead of splitmix64.
  add_library(exch_core_identity ${EXCH_SOURCES})
  target_include_directories(exch_core_identity PUBLIC include)
  target_compile_definitions(exch_core_identity PUBLIC EXCH_IDMAP_IDENTITY_HASH)
  add_executable(bench_engine_identity bench/bench_engine.cpp)
  target_link_libraries(bench_engine_identity PRIVATE exch_core_identity)
endif()

if(EXCH_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(exch_tests
    tests/test_fixed_point.cpp
    tests/test_id_map.cpp
    tests/test_engine.cpp
    tests/test_differential.cpp
    tests/test_replay.cpp)
  target_link_libraries(exch_tests PRIVATE exch_core GTest::gtest_main)
  # gtest headers trip some of our stricter warnings.
  target_compile_options(exch_tests PRIVATE -Wno-conversion -Wno-shadow)

  enable_testing()
  include(GoogleTest)
  gtest_discover_tests(exch_tests DISCOVERY_TIMEOUT 60)
  add_test(NAME replay_cli_roundtrip
           COMMAND ${CMAKE_COMMAND} -DEXCH=$<TARGET_FILE:exch> -DWORK=${CMAKE_CURRENT_BINARY_DIR}/replay_cli
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/replay_cli.cmake)
  # Golden file: the hand-written example session must replay to exactly the
  # reviewed output. Any semantic change shows up here as a diff.
  add_test(NAME golden_basic_session
           COMMAND exch verify --in ${CMAKE_CURRENT_SOURCE_DIR}/examples/basic.log
                   --expect ${CMAKE_CURRENT_SOURCE_DIR}/examples/basic.expected)
endif()
