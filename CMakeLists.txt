# 1. 声明要求的 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 2. 声明项目名称和语言
project(RoboMaster_Vision LANGUAGES CXX)

# 3. 设置 C++ 标准 (极其重要！)
# 我们之前的代码中用到了智能指针和 modern C++ 特性，建议直接上 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. 寻找 OpenCV 库
# REQUIRED 表示如果找不到 OpenCV，CMake 会直接报错停止
find_package(OpenCV REQUIRED)

# 5. 收集所有的源文件
# 将 src 目录下的所有 .cpp 文件存入变量 SOURCES 中
file(GLOB SOURCES "src/*.cpp")

# 6. 生成可执行文件
# 将所有的 .cpp 文件编译成一个名为 rm_vision 的可执行程序
add_executable(rm_vision ${SOURCES})

# ==========================================
# 7. 配置 Include 目录 (解答你的核心问题)
# ==========================================
# 告诉 rm_vision 这个程序，它的头文件都在根目录下的 include 文件夹里
# PRIVATE 表示这些头文件只给 rm_vision 自己用
target_include_directories(rm_vision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 8. 链接 OpenCV 库
# 将 OpenCV 的动态/静态库链接到我们的程序上
target_link_libraries(rm_vision PRIVATE ${OpenCV_LIBS})

# 9. (可选) 打印一条信息，确认 OpenCV 找对了
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")