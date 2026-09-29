cmake_minimum_required(VERSION 3.24)
project(mdp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(MDP_SANITIZE "Build with address and undefined behavior sanitizers" OFF)
option(MDP_TESTS "Build GoogleTest unit tests" ON)

find_package(ZLIB REQUIRED)

add_library(mdp STATIC src/pcap.cpp src/book.cpp)
target_include_directories(mdp PUBLIC include)
target_link_libraries(mdp PUBLIC ZLIB::ZLIB)
target_compile_options(mdp PUBLIC -Wall -Wextra -Wpedantic -Wno-gnu-zero-variadic-macro-arguments)
if(MDP_SANITIZE)
  target_compile_options(mdp PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mdp PUBLIC -fsanitize=address,undefined)
endif()

foreach(tool mdp_decode mdp_slice mdp_bench)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE mdp)
endforeach()

if(MDP_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(mdp_tests
    tests/test_messages.cpp
    tests/test_pcap.cpp
    tests/test_net.cpp
    tests/test_book.cpp
    tests/test_seq.cpp
    tests/test_roundtrip.cpp)
  target_include_directories(mdp_tests PRIVATE tests)
  target_link_libraries(mdp_tests PRIVATE mdp GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(mdp_tests)
endif()
