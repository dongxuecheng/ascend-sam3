cmake_minimum_required(VERSION 3.19)
project(ascend-sam3 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CANN 的部分算子库目录带有 host 架构后缀。兼容鲲鹏/aarch64 与 x86_64，
# 避免在 CMake 文件中固定为某一种服务器架构。
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_PROCESSOR_LOWER)
if(SYSTEM_PROCESSOR_LOWER MATCHES "^(aarch64|arm64)$")
    set(CANN_HOST_ARCH "aarch64")
elseif(SYSTEM_PROCESSOR_LOWER MATCHES "^(x86_64|amd64)$")
    set(CANN_HOST_ARCH "x86_64")
else()
    message(FATAL_ERROR "Unsupported host architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Ascend CANN 环境：必须通过环境变量 ASCEND_HOME_PATH 指定，不硬编码默认值
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set. Please set it to your Ascend CANN Toolkit path.")
endif()

include_directories(
    ${ASCEND_HOME}/include/acl
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/include/aclnnop
    ${CMAKE_SOURCE_DIR}/src
)

link_directories(
    ${ASCEND_HOME}/lib64
    ${ASCEND_HOME}/lib64/plugin/opskernel
    ${ASCEND_HOME}/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${CANN_HOST_ARCH}
)

# OpenCV：必须通过环境变量 OPENCV_INSTALL_DIR 指定，不硬编码默认值
if(DEFINED ENV{OPENCV_INSTALL_DIR})
    set(OPENCV_CUSTOM_DIR "$ENV{OPENCV_INSTALL_DIR}")
    if(EXISTS ${OPENCV_CUSTOM_DIR}/include/opencv4)
        set(OpenCV_INCLUDE_DIRS ${OPENCV_CUSTOM_DIR}/include/opencv4)
    elseif(EXISTS ${OPENCV_CUSTOM_DIR}/include/opencv)
        set(OpenCV_INCLUDE_DIRS ${OPENCV_CUSTOM_DIR}/include/opencv)
    else()
        message(FATAL_ERROR "OPENCV_INSTALL_DIR=${OPENCV_CUSTOM_DIR} does not contain OpenCV headers")
    endif()
    file(GLOB OpenCV_LIBS
        ${OPENCV_CUSTOM_DIR}/lib/libopencv_*.a
        ${OPENCV_CUSTOM_DIR}/lib/libopencv_*.so
        ${OPENCV_CUSTOM_DIR}/lib/aarch64-linux-gnu/libopencv_*.so
        ${OPENCV_CUSTOM_DIR}/lib/x86_64-linux-gnu/libopencv_*.so
    )
    if(NOT OpenCV_LIBS)
        message(FATAL_ERROR "No OpenCV libraries found under ${OPENCV_CUSTOM_DIR}/lib")
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
    src/infer/maskPostprocessCann.cpp
    src/infer/modelVision.cpp
    src/infer/modelText.cpp
    src/infer/modelDecoder.cpp
    src/infer/sam3infer.cpp
)

add_library(ascendsam3 STATIC ${COMMON_SRC} ${INFER_SRC})
# 静态 OpenCV 时需要额外链接 3rdparty；动态 OpenCV 时此目录可能不存在
file(GLOB OPENCV_3RDPARTY_LIBS "${OPENCV_CUSTOM_DIR}/lib/opencv4/3rdparty/*.a")
if(NOT OPENCV_3RDPARTY_LIBS)
    set(OPENCV_3RDPARTY_LIBS "")
endif()

# tokenizers-cpp 支持
# 当前环境是本机 aarch64，tokenizers-cpp 默认使用交叉编译前缀的工具名。
# 在本机构建场景下没有 aarch64-unknown-linux-gnu-gcc，因此创建一个工具链目录，
# 用同名符号链接指向本机 gcc/g++/ar，并通过 TOOLCHAIN_DIR 告知 tokenizers-cpp。
if(CANN_HOST_ARCH STREQUAL "aarch64")
    set(TOKENIZERS_TOOLCHAIN_DIR "${CMAKE_BINARY_DIR}/toolchain-bin")
    file(MAKE_DIRECTORY ${TOKENIZERS_TOOLCHAIN_DIR})
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
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 优先使用本地 vendored 的 tokenizers-cpp；不存在时从 GitHub 拉取
set(TOKENIZERS_CPP_LOCAL_DIR "${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp")
if(EXISTS ${TOKENIZERS_CPP_LOCAL_DIR}/CMakeLists.txt)
    message(STATUS "Found local tokenizers-cpp at ${TOKENIZERS_CPP_LOCAL_DIR}")
    add_subdirectory(${TOKENIZERS_CPP_LOCAL_DIR})
else()
    message(STATUS "Local tokenizers-cpp not found, fetching via FetchContent...")
    include(FetchContent)
    if(NOT DEFINED TOKENIZERS_GIT_REPOSITORY)
        set(TOKENIZERS_GIT_REPOSITORY "https://github.com/mlc-ai/tokenizers-cpp.git"
            CACHE STRING "tokenizers-cpp git repository URL")
    endif()
    if(NOT DEFINED TOKENIZERS_GIT_TAG)
        set(TOKENIZERS_GIT_TAG "c586c52" CACHE STRING "tokenizers-cpp git tag/commit")
    endif()
    FetchContent_Declare(
        tokenizers-cpp
        GIT_REPOSITORY ${TOKENIZERS_GIT_REPOSITORY}
        GIT_TAG        ${TOKENIZERS_GIT_TAG}
        GIT_SUBMODULES "sentencepiece;msgpack"
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(tokenizers-cpp)
endif()
list(APPEND EXTRA_LIBS tokenizers_cpp)

# 当前实现只调用 AscendCL/ACL Runtime API；mask 后处理走 ACL memcpy +
# OpenCV，不直接调用 aclnn。仅链接实际使用的库，避免 CANN 版本变化时因
# acl_op_compiler、nnopbase 或拆分后的 aclnn 库名称变化导致构建失败。
find_library(CANN_ASCENDCL_LIB NAMES ascendcl PATHS ${ASCEND_HOME}/lib64 NO_DEFAULT_PATH)
find_library(CANN_ACLRT_LIB NAMES runtime acl_rt PATHS ${ASCEND_HOME}/lib64 NO_DEFAULT_PATH)
if(NOT CANN_ASCENDCL_LIB)
    message(FATAL_ERROR "Cannot find libascendcl under ${ASCEND_HOME}/lib64")
endif()
set(CANN_LIBS ${CANN_ASCENDCL_LIB})
if(CANN_ACLRT_LIB)
    list(APPEND CANN_LIBS ${CANN_ACLRT_LIB})
endif()

# 静态库循环依赖时使用 start-group/end-group
target_link_libraries(ascendsam3
    ${CANN_LIBS}
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
add_executable(ascendsam3_demo src/main.cpp)
target_link_libraries(ascendsam3_demo ascendsam3)

# 性能测试程序
add_executable(ascendsam3_bench src/bench.cpp)
target_link_libraries(ascendsam3_bench ascendsam3)

# pybind11 Python 绑定
# 帮助 pybind11 定位当前 Python 解释器，避免多版本/自定义安装路径导致检测失败
if(NOT DEFINED PYTHON_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_Interpreter_FOUND)
        set(PYTHON_EXECUTABLE ${Python_EXECUTABLE} CACHE FILEPATH "Python interpreter")
    endif()
endif()

# 优先使用本地已安装的 pybind11（例如 pip install pybind11），找不到再回退到 FetchContent
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found locally, downloading via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(ascendsam3_py src/python/bindings.cpp)
set_target_properties(ascendsam3_py PROPERTIES OUTPUT_NAME ascendsam3 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(ascendsam3_py PRIVATE
    ascendsam3
    ${CANN_LIBS}
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


# 可选：安装模型目录说明
message(STATUS "ASCEND_HOME: ${ASCEND_HOME}")
message(STATUS "CANN_HOST_ARCH: ${CANN_HOST_ARCH}")
message(STATUS "CANN_LIBS: ${CANN_LIBS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
