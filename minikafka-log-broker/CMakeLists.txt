cmake_minimum_required(VERSION 3.24)
project(minikafka LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# The project standard is -O2 for release builds (CMake's default Release is -O3).
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

option(MK_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
option(MK_TSAN "Build with ThreadSanitizer" OFF)
option(MK_BUILD_TESTS "Build unit and integration tests" ON)
option(MK_BUILD_BENCH "Build the benchmark harness" ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
if(MK_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()
if(MK_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()
if((MK_SANITIZE OR MK_TSAN) AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # Observed on macOS 26.5 with Apple clang 17: the ASan runtime deadlocks during
  # its own initialisation (even for hello world). Homebrew LLVM works.
  message(WARNING "Sanitizer runtimes from Apple clang 17 hang on macOS 26.x; "
                  "configure with -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  # Apple's linker cannot parse DWARF 5 from newer upstream clang; DWARF 4 avoids the warnings.
  add_compile_options(-gdwarf-4)
endif()

find_package(Threads REQUIRED)

add_library(minikafka STATIC
  src/crc32c.cpp
  src/record.cpp
  src/segment.cpp
  src/partition_log.cpp
  src/offset_store.cpp
  src/group_coordinator.cpp
  src/net.cpp
  src/broker.cpp
  src/client.cpp
  src/producer.cpp
  src/consumer.cpp
)
target_include_directories(minikafka PUBLIC include)
target_link_libraries(minikafka PUBLIC Threads::Threads)

add_executable(mk-broker tools/broker_main.cpp)
target_link_libraries(mk-broker PRIVATE minikafka)

add_executable(mk-cli tools/cli_main.cpp)
target_link_libraries(mk-cli PRIVATE minikafka)

if(MK_BUILD_BENCH)
  add_executable(mk-bench bench/bench_main.cpp)
  target_link_libraries(mk-bench PRIVATE minikafka)
endif()

if(MK_BUILD_TESTS)
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
  add_executable(mk-tests
    tests/test_codec.cpp
    tests/test_log.cpp
    tests/test_groups.cpp
    tests/test_broker.cpp
  )
  target_link_libraries(mk-tests PRIVATE minikafka GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(mk-tests DISCOVERY_TIMEOUT 30)
endif()
