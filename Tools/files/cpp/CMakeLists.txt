cmake_minimum_required(VERSION 3.16)
project(tvra_generic_file_hash_accelerator LANGUAGES CXX)

add_executable(file_hash_accelerator file_hash_accelerator.cpp)
target_compile_features(file_hash_accelerator PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(file_hash_accelerator PRIVATE /O2 /W4)
else()
    target_compile_options(file_hash_accelerator PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()