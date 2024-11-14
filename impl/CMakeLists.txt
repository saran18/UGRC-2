cmake_minimum_required(VERSION 3.10)
project(JarvisParallel LANGUAGES CXX)

# Specify the HIP compiler path
set(CMAKE_CXX_COMPILER "/opt/rocm-6.0.2/bin/hipcc")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the directory for header files
message(STATUS "CMake Source Directory: ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define the target platform
add_definitions(-D__HIP_PLATFORM_AMD__)

# Add the executable and source files
add_executable(dc_parallel
    src/dc.cpp
)

add_executable(hull_merge
    src/hull_merge.cpp
)

add_executable(triangulation
    src/triangulation.cpp
)

add_executable(sample_seq
    src/sample_seq.cpp
)

# Link HIP runtime to the executable
find_package(HIP REQUIRED)
target_link_libraries(dc_parallel PRIVATE ${HIP_LIBRARIES})
