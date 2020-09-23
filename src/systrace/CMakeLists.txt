cmake_minimum_required(VERSION 2.8.3)
project(systrace)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(
include/systrace
)

cs_add_library(${PROJECT_NAME}
src/tracer.cpp
)

cs_install()
cs_export()
