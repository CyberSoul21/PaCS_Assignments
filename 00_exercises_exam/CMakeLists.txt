cmake_minimum_required(VERSION 3.15...3.30)
project(
    Problems
    VERSION 1.0
    LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#For all executables in project 
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

add_executable(dotProduct 01_exercise_dot_product.cpp)
add_executable(maxMin 02_maxMin.cpp)
add_executable(prime 05_prime.cpp)


