cmake_minimum_required(VERSION 3.24)
project(az_lookahead CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall -Wextra -Wpedantic)

option(C4_BUILD_PYTHON "Build the pybind11 module" ON)
option(C4_BUILD_TESTS "Build GoogleTest suite" ON)

add_library(c4 STATIC
  cpp/src/position.cpp
  cpp/src/solver.cpp
  cpp/src/pure_mcts.cpp
  cpp/src/puct.cpp)
target_include_directories(c4 PUBLIC cpp/include)

add_executable(c4_bench cpp/src/bench.cpp)
target_link_libraries(c4_bench PRIVATE c4)

if(C4_BUILD_PYTHON)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(c4core cpp/src/bindings.cpp)
  target_link_libraries(c4core PRIVATE c4)
  set_target_properties(c4core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/az)
endif()

if(C4_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(c4_tests
    cpp/tests/test_position.cpp
    cpp/tests/test_solver.cpp
    cpp/tests/test_mcts.cpp)
  target_link_libraries(c4_tests PRIVATE c4 GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(c4_tests)
endif()
