cmake_minimum_required(VERSION 3.24)
project(vio_to_policy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video features)
find_package(GTest REQUIRED)

add_library(vio
  src/imu.cpp
  src/eskf.cpp
  src/geometry.cpp
  src/sim.cpp
  src/euroc.cpp
  src/frontend.cpp
  src/vio_system.cpp)
target_include_directories(vio PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vio PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_compile_options(vio PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(run_euroc apps/run_euroc.cpp)
target_link_libraries(run_euroc PRIVATE vio)
add_executable(sim_nees apps/sim_nees.cpp)
target_link_libraries(sim_nees PRIVATE vio)

enable_testing()
add_executable(vio_tests
  tests/test_so3.cpp
  tests/test_imu.cpp
  tests/test_geometry.cpp
  tests/test_eskf.cpp)
target_link_libraries(vio_tests PRIVATE vio GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(vio_tests)
