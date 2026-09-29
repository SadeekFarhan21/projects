cmake_minimum_required(VERSION 3.24)
project(figgie_arena LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FIGGIE_BUILD_TESTS "Build GoogleTest suite" ON)
option(FIGGIE_BUILD_PYTHON "Build pybind11 module" ON)
option(FIGGIE_SANITIZE "Build with address and undefined behaviour sanitizers" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)
if(FIGGIE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(figgie STATIC
  cpp/src/game.cpp
  cpp/src/posterior.cpp
  cpp/src/bots.cpp
  cpp/src/arena.cpp)
target_include_directories(figgie PUBLIC cpp/include)
find_package(Threads REQUIRED)
target_link_libraries(figgie PUBLIC Threads::Threads)

add_executable(figgie_fuzz cpp/tools/fuzz_main.cpp)
target_link_libraries(figgie_fuzz PRIVATE figgie)

add_executable(figgie_bench cpp/tools/bench_main.cpp)
target_link_libraries(figgie_bench PRIVATE figgie)

if(FIGGIE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(figgie_tests
    cpp/tests/test_book.cpp
    cpp/tests/test_game.cpp
    cpp/tests/test_posterior.cpp
    cpp/tests/test_arena.cpp)
  target_link_libraries(figgie_tests PRIVATE figgie GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(figgie_tests)
endif()

if(FIGGIE_BUILD_PYTHON)
  find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_figgie cpp/bindings/module.cpp)
  target_link_libraries(_figgie PRIVATE figgie)
  set_target_properties(_figgie PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/figgie_arena)
endif()
