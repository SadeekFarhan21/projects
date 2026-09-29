cmake_minimum_required(VERSION 3.20)
project(gpu_kernel_library LANGUAGES CXX)

# CPU references, the SIMT emulator and all CPU tests always build.
# CUDA targets build only when a CUDA compiler is found (and GKL_ENABLE_CUDA=ON).
option(GKL_ENABLE_CUDA "Build CUDA kernels, GPU tests and GPU benchmarks when CUDA is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

set(GKL_HAVE_CUDA OFF)
if(GKL_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 89 90)  # L4 (sm_89) and H100 (sm_90)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GKL_HAVE_CUDA ON)
  endif()
endif()
message(STATUS "gkl: CUDA targets ${GKL_HAVE_CUDA}")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
endif()

# ---------------------------------------------------------------- CPU side
add_library(gkl_cpu src/cpu/reference.cpp src/cpu/algorithms.cpp src/names.cpp)
target_include_directories(gkl_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

# The real kernel sources compiled as C++ against the SIMT emulator.
add_library(gkl_emu src/emu/launchers_emu.cpp)
target_link_libraries(gkl_emu PUBLIC gkl_cpu Threads::Threads)

enable_testing()

add_executable(test_ref tests/test_ref.cpp)
target_link_libraries(test_ref PRIVATE gkl_cpu)
add_test(NAME test_ref COMMAND test_ref)

add_executable(test_emu_kernels tests/test_emu_kernels.cpp)
target_link_libraries(test_emu_kernels PRIVATE gkl_emu)
add_test(NAME test_emu_kernels COMMAND test_emu_kernels)

add_executable(gkl_dump tools/gkl_dump.cpp)
target_link_libraries(gkl_dump PRIVATE gkl_emu)

add_executable(bench_cpu bench/bench_cpu.cpp)
target_link_libraries(bench_cpu PRIVATE gkl_cpu)

# ---------------------------------------------------------------- GPU side
if(GKL_HAVE_CUDA)
  add_library(gkl_cuda src/cuda/launchers.cu)
  target_include_directories(gkl_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(gkl_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --use_fast_math>)
  target_link_libraries(gkl_cuda PUBLIC gkl_cpu CUDA::cudart)

  add_executable(test_gpu_kernels tests/test_gpu_kernels.cu)
  target_link_libraries(test_gpu_kernels PRIVATE gkl_cuda gkl_cpu)
  add_test(NAME test_gpu_kernels COMMAND test_gpu_kernels)

  add_executable(bench_gpu bench/bench_gpu.cu)
  target_link_libraries(bench_gpu PRIVATE gkl_cuda gkl_cpu CUDA::cublas)
endif()
