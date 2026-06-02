cmake_minimum_required(VERSION 3.10)
project(RoboMaster_Vision LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 OpenCV
find_package(OpenCV REQUIRED)

# 收集 src 目录下的所有 cpp 文件
file(GLOB SOURCES "src/*.cpp")

# 生成可执行文件
add_executable(rm_vision ${SOURCES})

# 包含头文件目录
target_include_directories(rm_vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接 OpenCV 库
target_link_libraries(rm_vision PRIVATE ${OpenCV_LIBS})