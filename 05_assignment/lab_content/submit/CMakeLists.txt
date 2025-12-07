cmake_minimum_required(VERSION 3.15...3.30)
project(GaussianFilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${CMAKE_SOURCE_DIR}/CImg)
find_package(OpenCL REQUIRED)
find_package(X11 REQUIRED)

set_source_files_properties(environ_gaussian_filter.c PROPERTIES LANGUAGE CXX)
add_executable(gaussian_filter environ_gaussian_filter.c)
target_link_libraries(gaussian_filter m pthread X11 OpenCL::OpenCL)