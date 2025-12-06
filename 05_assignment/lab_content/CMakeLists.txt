cmake_minimum_required(VERSION 3.15...3.30)
project(
    GaussianFilter
    VERSION 1.0
    LANGUAGES CXX
)

# For all executables in project
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Include directory for CImg
include_directories(${CMAKE_SOURCE_DIR}/CImg)

# Add executable
add_executable(executable environ_gaussian_filter.c)

# Link libraries
target_link_libraries(executable
    m
    pthread
    X11
    OpenCL
)
