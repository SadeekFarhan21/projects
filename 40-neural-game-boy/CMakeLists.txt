cmake_minimum_required(VERSION 3.24)
project(neural_game_boy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GB_WITH_SDL "Build the SDL2 windowed frontend" ON)
option(GB_BUILD_TESTS "Build GoogleTest tests" ON)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

find_package(ZLIB REQUIRED)

# Emulator core: no I/O dependencies beyond the C++ standard library.
add_library(gbcore STATIC
  src/core/cartridge.cpp
  src/core/timer.cpp
  src/core/ppu.cpp
  src/core/bus.cpp
  src/core/cpu.cpp
  src/core/gameboy.cpp
  src/core/testrom.cpp
)
target_include_directories(gbcore PUBLIC src)

# Utilities and the dataset logger (need zlib).
add_library(gbtools STATIC
  src/util/png.cpp
  src/logger/recorder.cpp
  src/logger/policy.cpp
)
target_include_directories(gbtools PUBLIC src)
target_link_libraries(gbtools PUBLIC gbcore ZLIB::ZLIB)

add_executable(gbemu apps/gbemu.cpp)
target_link_libraries(gbemu PRIVATE gbtools)

if(GB_WITH_SDL)
  find_package(SDL2 CONFIG QUIET)
  if(SDL2_FOUND)
    target_sources(gbemu PRIVATE apps/sdl_frontend.cpp)
    target_compile_definitions(gbemu PRIVATE GB_HAVE_SDL=1)
    if(TARGET SDL2::SDL2)
      target_link_libraries(gbemu PRIVATE SDL2::SDL2)
    else()
      target_link_libraries(gbemu PRIVATE ${SDL2_LIBRARIES})
      target_include_directories(gbemu PRIVATE ${SDL2_INCLUDE_DIRS})
    endif()
    message(STATUS "SDL2 found, windowed mode enabled")
  else()
    message(STATUS "SDL2 not found, building headless only")
  endif()
endif()

if(GB_BUILD_TESTS)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
  add_executable(gb_tests
    tests/test_cpu.cpp
    tests/test_timer.cpp
    tests/test_cartridge.cpp
    tests/test_ppu.cpp
    tests/test_logger.cpp
    tests/test_roms.cpp
  )
  target_link_libraries(gb_tests PRIVATE gbtools GTest::gtest GTest::gtest_main)
  target_compile_definitions(gb_tests PRIVATE
    GB_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  include(GoogleTest)
  gtest_discover_tests(gb_tests DISCOVERY_TIMEOUT 60)
endif()
