cmake_minimum_required(VERSION 3.24)
project(kvd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KV_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(KV_BUILD_TESTS "Build the test suite" ON)

# Release means -O2 here (not CMake's default -O3) so numbers match the spec.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

add_compile_options(-Wall -Wextra -Wpedantic -Wno-gnu-zero-variadic-macro-arguments)
if(KV_SANITIZE)
  # -gdwarf-4: Homebrew clang emits DWARF 5 that Apple's ld warns it cannot parse.
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined -gdwarf-4)
  add_link_options(-fsanitize=address,undefined)
endif()

# Core library: everything except main(), so tests can link it.
add_library(kvcore
  src/resp.cpp
  src/glob.cpp
  src/store.cpp
  src/commands.cpp
  src/aof.cpp
  src/server.cpp
  src/poller_kqueue.cpp
  src/poller_epoll.cpp
)
target_include_directories(kvcore PUBLIC src)

add_executable(kvd src/main.cpp)
target_link_libraries(kvd PRIVATE kvcore)

add_executable(kvbench bench/loadgen.cpp)
find_package(Threads REQUIRED)
target_link_libraries(kvbench PRIVATE Threads::Threads)

if(KV_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(kv_tests
    tests/test_resp.cpp
    tests/test_glob.cpp
    tests/test_store.cpp
    tests/test_commands.cpp
    tests/test_aof.cpp
    tests/test_server.cpp
  )
  target_link_libraries(kv_tests PRIVATE kvcore GTest::gtest_main Threads::Threads)
  include(GoogleTest)
  gtest_discover_tests(kv_tests DISCOVERY_TIMEOUT 30)

  # End-to-end test through the real binary with a stdlib-only Python client.
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_test(NAME e2e_python_client
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/client_test.py --server $<TARGET_FILE:kvd>)
    set_tests_properties(e2e_python_client PROPERTIES TIMEOUT 60)
  endif()
endif()
