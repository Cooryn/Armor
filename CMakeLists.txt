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
add_executable(Armor ${SOURCES})

# ==========================================
# 【核心修改点】：设置可执行文件的输出路径
# ==========================================
# CMAKE_CURRENT_SOURCE_DIR 就是包含这个 CMakeLists.txt 的根目录
# 我们强制所有的构建配置 (Debug, Release 等) 都输出到根目录
set_target_properties(Armor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)

# 包含头文件目录
target_include_directories(Armor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接 OpenCV 库
target_link_libraries(Armor PRIVATE ${OpenCV_LIBS})