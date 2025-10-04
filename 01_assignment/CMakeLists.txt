cmake_minimum_required(VERSION 3.15...3.30)
project(
    Memory
    VERSION 1.0
    LANGUAGES CXX
)


#  817  cd build/
#  818  cmake .. -G "unix Makefiles"
#  819  cmake .. -G "Unix Makefiles"


#For all executables in project 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -I")
add_executable(matrix matrix.cpp)
add_executable(matrix_stack matrix_stack.cpp)
add_executable(eigen_matrix eigen_matrix.cpp)
