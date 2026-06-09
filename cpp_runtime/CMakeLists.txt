cmake_minimum_required(VERSION 3.10)

project(neptunet_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(neptunet_runtime
    src/main.cpp
    src/neptunet_state_machine.cpp
)

target_include_directories(neptunet_runtime PRIVATE
    include
)