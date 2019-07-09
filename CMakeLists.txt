cmake_minimum_required(VERSION 3.1)

project(vehicle_tracking)

## Compile as C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Compile with the highest warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

## System dependencies
find_package(OpenCV 3 REQUIRED)

## Specify additional locations of header files
include_directories(${catkin_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

## Declare a C++ executable
add_executable(vehicle_tracking src/vehicle_tracking.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(vehicle_tracking ${OpenCV_LIBS})
