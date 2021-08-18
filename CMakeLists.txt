cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(flwr_cpp_client VERSION 0.1.0 LANGUAGES CXX)
find_package(TorchVision REQUIRED)

set(EXECUTABLE_NAME client)
# The first thing do is to tell cmake to find the TorchVision library.
# The package pulls in all the necessary torch libraries,
# so there is no need to also add `find_package(Torch)` here.

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/cifar10.cpp
                                          src/transform.cpp
                                          include/cifar10.h
                                          include/transform.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} TorchVision::TorchVision)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

if(DOWNLOAD_DATASETS)
  add_dependencies(${EXECUTABLE_NAME} cifar10)
endif()

