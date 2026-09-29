cmake_minimum_required(VERSION 3.24)
project(tcm_runtime LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.12 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_metal metal_runtime.mm)
target_compile_options(_metal PRIVATE -fobjc-arc -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(_metal PRIVATE "-framework Metal" "-framework Foundation")

# Drop the extension next to the Python package so `import tcm._metal` works.
set_target_properties(_metal PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/tcm"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../src/tcm")
