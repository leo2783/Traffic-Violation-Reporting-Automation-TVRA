cmake_minimum_required(VERSION 3.16)
project(FileCompareTool VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

# 依賴 Qt Widgets 模組
find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(FileCompareTool main.cpp)
target_link_libraries(FileCompareTool PRIVATE Qt6::Widgets)

# 若在 Windows 上隱藏終端機視窗：
if(WIN32)
    set_property(TARGET FileCompareTool PROPERTY WIN32_EXECUTABLE ON)
endif()