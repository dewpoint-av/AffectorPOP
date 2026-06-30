cmake_minimum_required(VERSION 3.24)
project(AffectorPOP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(AffectorPOP SHARED
  src/AffectorPOP.cpp
  src/kernel.cu)

target_include_directories(AffectorPOP PRIVATE
  ${CMAKE_SOURCE_DIR}/sdk
  ${CMAKE_SOURCE_DIR}/src)

set_target_properties(AffectorPOP PROPERTIES
  PREFIX ""
  CUDA_ARCHITECTURES "75;86;89")

target_compile_options(AffectorPOP PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(AffectorPOP PRIVATE CUDA::cudart)

if(WIN32)
  target_compile_definitions(AffectorPOP PRIVATE _WINDOWS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
