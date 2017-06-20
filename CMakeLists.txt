cmake_minimum_required(VERSION 3.2)
project(DeepLearningCPP)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/network.hpp src/mnist_loader.hpp src/network2.hpp)
add_executable(DeepLearningCPP ${SOURCE_FILES})
find_library(ARMADILLO armadillo)
target_link_libraries(DeepLearningCPP
        ${ARMADILLO})
