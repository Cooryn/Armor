cmake_minimum_required(VERSION 3.10)
project(RoboMaster_Vision LANGUAGES CXX)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 关闭特定编译器的扩展以保证跨平台兼容性
set(CMAKE_CXX_EXTENSIONS OFF) 

# ==========================================
# 1. 寻找第三方依赖库
# ==========================================
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# ==========================================
# 2. 收集源文件 (告别 GLOB 陷阱)
# ==========================================
# 强烈建议显式列出，而不是用 file(GLOB ...)
set(SOURCES
    src/main.cpp
    src/lightbar_detector.cpp
    src/solver.cpp
    # src/ekf_predictor.cpp  # 如果你以后要在 C++ 里加回 EKF，取消注释即可
)

# ==========================================
# 3. 创建可执行文件
# ==========================================
add_executable(Armor ${SOURCES})

# ==========================================
# 4. 配置目标 (头文件、链接库、编译选项) - 集中管理
# ==========================================
# 包含头文件目录
target_include_directories(Armor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 统一链接依赖库 (OpenCV 和 Eigen)
target_link_libraries(Armor PRIVATE 
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# 开启严格的编译警告 (帮你提前揪出 C++ 里的内存或逻辑 Bug)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(Armor PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()

# ==========================================
# 5. 设置可执行文件的输出路径
# ==========================================
set_target_properties(Armor PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
)