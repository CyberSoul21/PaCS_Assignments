cmake_minimum_required(VERSION 3.15...3.30)
project(GaussianFilterOpenCL LANGUAGES CXX)

# Compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Include CImg directory
include_directories(${CMAKE_SOURCE_DIR}/CImg)

# Find OpenCL
find_package(OpenCL REQUIRED)

# For X11 (used by CImg when displaying images)
find_package(X11 REQUIRED)

# Executable 1
add_executable(gaussian_filter environ_gaussian_filter.c)
target_link_libraries(gaussian_filter
    m
    pthread
    X11
    OpenCL::OpenCL
)

# # Executable 2
# add_executable(gaussian_filter_cache environ_gaussian_filter_cache.c)
# target_link_libraries(gaussian_filter_cache
#     m
#     pthread
#     X11
#     OpenCL::OpenCL
# )
