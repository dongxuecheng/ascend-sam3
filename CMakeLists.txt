cmake_minimum_required(VERSION 3.19)
project(ascend-sam3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ascend CANN 环境：必须通过环境变量 ASCEND_HOME_PATH 指定，不硬编码默认值
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set. Please set it to your Ascend CANN Toolkit path.")
endif()

include_directories(
    ${ASCEND_HOME}/include/acl
    ${ASCEND_HOME}/include
    ${CMAKE_SOURCE_DIR}/src
)

link_directories(
    ${ASCEND_HOME}/lib64
    ${ASCEND_HOME}/lib64/plugin/opskernel
)

# OpenCV：必须通过环境变量 OPENCV_INSTALL_DIR 指定，不硬编码默认值
if(DEFINED ENV{OPENCV_INSTALL_DIR})
    set(OPENCV_CUSTOM_DIR "$ENV{OPENCV_INSTALL_DIR}")
    if(EXISTS ${OPENCV_CUSTOM_DIR}/include/opencv4)
        set(OpenCV_INCLUDE_DIRS ${OPENCV_CUSTOM_DIR}/include/opencv4)
        file(GLOB OpenCV_LIBS ${OPENCV_CUSTOM_DIR}/lib/libopencv_*.a)
    else()
        message(FATAL_ERROR "OPENCV_INSTALL_DIR=${OPENCV_CUSTOM_DIR} does not contain include/opencv4")
    endif()
else()
    message(FATAL_ERROR "OPENCV_INSTALL_DIR environment variable is not set. Please set it to your OpenCV install path.")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set(COMMON_SRC
    src/common/aclmemory.cpp
    src/common/object.cpp
    src/common/npy_utils.cpp
    src/common/tokenizer.cpp
)

set(INFER_SRC
    src/infer/aclmodel.cpp
    src/infer/infer.cpp
    src/infer/modelVision.cpp
    src/infer/modelText.cpp
    src/infer/modelDecoder.cpp
    src/infer/sam3infer.cpp
)

add_library(sam3_infer STATIC ${COMMON_SRC} ${INFER_SRC})
set(OPENCV_3RDPARTY_LIBS
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/libtegra_hal.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/liblibopenjp2.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/liblibwebp.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/liblibpng.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/liblibtiff.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/liblibjpeg-turbo.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/libIlmImf.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/liblibprotobuf.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/libzlib.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/libade.a
    ${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/libittnotify.a
)

# tokenizers-cpp 支持
set(tokenizers_cpp_DIR "${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp")
if(EXISTS ${tokenizers_cpp_DIR}/CMakeLists.txt)
    message(STATUS "Found tokenizers-cpp at ${tokenizers_cpp_DIR}")
    # 当前环境是本机 aarch64，tokenizers-cpp 默认使用交叉编译前缀的工具名。
    # 在本机构建场景下没有 aarch64-unknown-linux-gnu-gcc，因此创建一个工具链目录，
    # 用同名符号链接指向本机 gcc/g++/ar，并通过 TOOLCHAIN_DIR 告知 tokenizers-cpp。
    set(TOKENIZERS_TOOLCHAIN_DIR "${CMAKE_BINARY_DIR}/toolchain-bin")
    file(MAKE_DIRECTORY ${TOKENIZERS_TOOLCHAIN_DIR})
    # 使用 CMake 已检测到的本机编译器创建交叉编译命名软链接
    set(_cc "${CMAKE_C_COMPILER}")
    set(_cxx "${CMAKE_CXX_COMPILER}")
    set(_ar "${CMAKE_AR}")
    if(NOT _cc)
        set(_cc "/usr/bin/gcc")
    endif()
    if(NOT _cxx)
        set(_cxx "/usr/bin/g++")
    endif()
    if(NOT _ar)
        set(_ar "/usr/bin/ar")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${_cc}  "${TOKENIZERS_TOOLCHAIN_DIR}/aarch64-unknown-linux-gnu-gcc")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${_cxx} "${TOKENIZERS_TOOLCHAIN_DIR}/aarch64-unknown-linux-gnu-g++")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${_ar}  "${TOKENIZERS_TOOLCHAIN_DIR}/aarch64-unknown-linux-gnu-ar")
    set(TOOLCHAIN_DIR "${TOKENIZERS_TOOLCHAIN_DIR}/" CACHE PATH "tokenizers-cpp toolchain dir" FORCE)
    add_subdirectory(${tokenizers_cpp_DIR})
    list(APPEND EXTRA_LIBS tokenizers_cpp)
else()
    message(WARNING "tokenizers-cpp not found at ${tokenizers_cpp_DIR}, text tokenizer disabled")
endif()

# 静态库循环依赖时使用 start-group/end-group
target_link_libraries(sam3_infer
    ascendcl
    runtime
    acl_op_compiler
    -Wl,--start-group
    ${OpenCV_LIBS}
    ${OPENCV_3RDPARTY_LIBS}
    -Wl,--end-group
    ${EXTRA_LIBS}
    z
    pthread
    dl
    stdc++
)

# 示例可执行程序
add_executable(sam3_demo src/main.cpp)
target_link_libraries(sam3_demo sam3_infer)

# 性能测试程序
add_executable(sam3_bench src/bench.cpp)
target_link_libraries(sam3_bench sam3_infer)

# 可选：安装模型目录说明
message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
