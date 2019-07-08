cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(vehicle_tracking)

find_package(OpenCV REQUIRED)

add_executable(vehicle_tracking src/vehicle_tracking.cpp)
target_link_libraries(vehicle_tracking ${OpenCV_LIBS})
