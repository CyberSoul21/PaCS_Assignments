cmake_minimum_required(VERSION 3.15...3.30)
project(GaussianFilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${CMAKE_SOURCE_DIR}/CImg)
find_package(OpenCL REQUIRED)
find_package(X11 REQUIRED)

set_source_files_properties(environ_gaussian_filter_metrics.c PROPERTIES LANGUAGE CXX)
add_executable(gaussian_filter_metrics environ_gaussian_filter_metrics.c)
target_link_libraries(gaussian_filter_metrics m pthread X11 OpenCL::OpenCL)

set_source_files_properties(environ_gaussian_filter_metrics_non_heterogen.c PROPERTIES LANGUAGE CXX)
add_executable(environ_gaussian_filter_metrics_non_heterogen environ_gaussian_filter_metrics_non_heterogen.c)
target_link_libraries(environ_gaussian_filter_metrics_non_heterogen m pthread X11 OpenCL::OpenCL)