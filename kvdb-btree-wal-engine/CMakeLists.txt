cmake_minimum_required(VERSION 3.24)
project(kvdb LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Release is -O2 (not CMake's default -O3) as the project standard.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Comma-separated sanitizer list, e.g. "address,undefined" or "undefined".
set(KVDB_SANITIZERS "" CACHE STRING "Sanitizers to enable (-fsanitize=...)")
option(KVDB_BUILD_TESTS "Build unit tests" ON)
option(KVDB_BUILD_BENCH "Build benchmarks" ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
if(KVDB_SANITIZERS)
  add_compile_options(-fsanitize=${KVDB_SANITIZERS} -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=${KVDB_SANITIZERS})
  # Homebrew LLVM emits DWARF 5, which Apple's linker warns it cannot parse.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-gdwarf-4)
  endif()
  # libc++ debug hardening: bounds-checked containers, string_view, etc.
  add_compile_definitions(_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
endif()

add_library(kvdb
  src/common.cc
  src/pager.cc
  src/buffer_pool.cc
  src/node.cc
  src/btree.cc
  src/wal.cc
  src/db.cc)
target_include_directories(kvdb PUBLIC include)

add_executable(kvdb_repl tools/repl.cc)
target_link_libraries(kvdb_repl PRIVATE kvdb)

if(KVDB_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(kvdb_tests
    tests/test_util.h
    tests/node_test.cc
    tests/buffer_pool_test.cc
    tests/btree_test.cc
    tests/wal_test.cc
    tests/recovery_test.cc)
  target_link_libraries(kvdb_tests PRIVATE kvdb GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(kvdb_tests DISCOVERY_TIMEOUT 60)
endif()

if(KVDB_BUILD_BENCH)
  add_executable(kvdb_bench bench/bench.cc)
  target_link_libraries(kvdb_bench PRIVATE kvdb)
  find_package(SQLite3)
  if(SQLite3_FOUND)
    target_compile_definitions(kvdb_bench PRIVATE KVDB_HAVE_SQLITE=1)
    if(TARGET SQLite3::SQLite3)
      target_link_libraries(kvdb_bench PRIVATE SQLite3::SQLite3)
    else()
      target_link_libraries(kvdb_bench PRIVATE SQLite::SQLite3)
    endif()
  endif()
endif()
