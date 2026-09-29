cmake_minimum_required(VERSION 3.20)
project(gpu_verilator CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(verilator REQUIRED HINTS $ENV{VERILATOR_ROOT} /opt/homebrew/share/verilator)

# Verilog emitted by `gpu emit-verilog generated/gpu.v`
set(GPU_VERILOG ${CMAKE_CURRENT_SOURCE_DIR}/../generated/gpu.v)

add_executable(gpu_tb tb.cpp)
target_compile_options(gpu_tb PRIVATE -O2)
verilate(gpu_tb SOURCES ${GPU_VERILOG} TOP_MODULE gpu PREFIX Vgpu
  VERILATOR_ARGS -Wno-fatal -O3)
