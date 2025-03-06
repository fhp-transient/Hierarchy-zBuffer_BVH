cmake_minimum_required(VERSION 3.27)
project(assignment3)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "F:/opencv/mingw_build/install")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(F:/eigen-3.4.0)
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES
        ${CMAKE_SOURCE_DIR}/src/*.cpp   # 所有 .cpp 文件
)

file(GLOB HEADERS
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/include/*.hpp
)

add_executable(assignment3 ${SOURCES})

target_link_libraries(assignment3 ${OpenCV_LIBS})

#set(CMAKE_BUILD_TYPE "Release")