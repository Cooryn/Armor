cmake_minimum_required(VERSION 3.16)
project(RoboMaster_Vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/$<0:>")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(include ${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} Eigen3::Eigen)
add_compile_definitions(ARMOR_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

set(COMMON_SOURCES src/camera.cpp src/lightbar_detector.cpp src/solver.cpp src/serial.cpp)
add_executable(Armor src/Armor.cpp ${COMMON_SOURCES}
    src/predictor_single.cpp src/predictor_armor.cpp
    src/pose_base.cpp src/gimbal.cpp src/log_output.cpp)
add_executable(Armor_detect src/Armor_detect.cpp ${COMMON_SOURCES})
