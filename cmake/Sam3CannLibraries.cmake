# Link against the toolkit's development libraries, never the real runtime or
# simulator libraries. Docker builds do not have the host driver mounted.
# ASCEND_HOME and CANN_HOST_ARCH must be set by the caller.
if(NOT ASCEND_HOME OR NOT CANN_HOST_ARCH MATCHES "^(aarch64|x86_64)$")
    message(FATAL_ERROR "Set ASCEND_HOME and a supported CANN_HOST_ARCH before selecting CANN libraries")
endif()

# Older CANN packages may use arm64-linux instead of aarch64-linux.
set(_sam3_cann_arch_roots "${ASCEND_HOME}/${CANN_HOST_ARCH}-linux")
if(CANN_HOST_ARCH STREQUAL "aarch64")
    list(APPEND _sam3_cann_arch_roots "${ASCEND_HOME}/arm64-linux")
endif()
set(_sam3_cann_dev_dirs
    "${ASCEND_HOME}/devlib/linux/${CANN_HOST_ARCH}"
    "${ASCEND_HOME}/devlib/${CANN_HOST_ARCH}"
    "${ASCEND_HOME}/devlib")
foreach(_root IN LISTS _sam3_cann_arch_roots)
    list(APPEND _sam3_cann_dev_dirs
        "${_root}/devlib/linux/${CANN_HOST_ARCH}"
        "${_root}/devlib/${CANN_HOST_ARCH}"
        "${_root}/devlib")
endforeach()
# Compatibility with older AscendCL toolkit layouts.
list(APPEND _sam3_cann_dev_dirs
    "${ASCEND_HOME}/runtime/lib64/stub"
    "${ASCEND_HOME}/acllib/lib64/stub"
    "${ASCEND_HOME}/lib64/stub")
foreach(_root IN LISTS _sam3_cann_arch_roots)
    list(APPEND _sam3_cann_dev_dirs
        "${_root}/runtime/lib64/stub"
        "${_root}/acllib/lib64/stub"
        "${_root}/lib64/stub")
endforeach()

# find_library caches its result. Migrate build trees that previously selected
# lib64/libruntime.so and reselect if the toolkit or architecture changes.
foreach(_var CANN_ASCENDCL_LIB CANN_ACLRT_LIB CANN_LIB_DIR)
    unset(${_var} CACHE)
    unset(${_var})
endforeach()
foreach(_dir IN LISTS _sam3_cann_dev_dirs)
    find_library(CANN_ASCENDCL_LIB NAMES ascendcl PATHS "${_dir}" NO_DEFAULT_PATH)
    if(CANN_ASCENDCL_LIB)
        set(CANN_LIB_DIR "${_dir}")
        break()
    endif()
endforeach()
if(NOT CANN_ASCENDCL_LIB)
    message(FATAL_ERROR
        "Cannot find CANN development libascendcl for ${CANN_HOST_ARCH} under ${ASCEND_HOME}. "
        "Install the matching toolkit devlib/lib64/stub libraries; real lib64/runtime and "
        "simulator libraries are deliberately not used for linking without a driver.")
endif()

set(CANN_LIBS "${CANN_ASCENDCL_LIB}")
# Recent CANN splits ACL runtime APIs into acl_rt. Older monolithic AscendCL
# stubs do not need this extra library. Never mix libraries from different dirs.
find_library(CANN_ACLRT_LIB NAMES acl_rt PATHS "${CANN_LIB_DIR}" NO_DEFAULT_PATH)
if(CANN_ACLRT_LIB)
    list(APPEND CANN_LIBS "${CANN_ACLRT_LIB}")
endif()
message(STATUS "CANN link directory (development only): ${CANN_LIB_DIR}")

function(sam3_configure_cann_target target)
    # rpath-link is a linker-only lookup path for indirect dependencies; it is
    # not embedded in the output. Runtime LD_LIBRARY_PATH selects the real libs.
    target_link_options(${target} PRIVATE "-Wl,-rpath-link,${CANN_LIB_DIR}")
    set_target_properties(${target} PROPERTIES
        SKIP_BUILD_RPATH TRUE
        BUILD_RPATH ""
        BUILD_WITH_INSTALL_RPATH FALSE
        INSTALL_RPATH ""
        INSTALL_RPATH_USE_LINK_PATH FALSE)
endfunction()
