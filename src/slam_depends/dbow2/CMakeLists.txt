cmake_minimum_required(VERSION 2.8.3)
project(dbow2)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
find_package(OpenCV 3.2.0 REQUIRED)

include_directories(
include
include/DBoW2
include/DUtils
include/DVision
${OpenCV_INCLUDE_DIRS}
)

cs_add_library(${PROJECT_NAME} SHARED
src/VocabularyBinary.cpp
src/DBoW2/BowVector.cpp
src/DBoW2/FBrief.cpp
src/DBoW2/FeatureVector.cpp
src/DBoW2/FORB.cpp
src/DBoW2/QueryResults.cpp
src/DBoW2/ScoringObject.cpp
src/DUtils/Random.cpp
src/DUtils/Timestamp.cpp
src/DVision/BRIEF.cpp
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

cs_install()
cs_export()
