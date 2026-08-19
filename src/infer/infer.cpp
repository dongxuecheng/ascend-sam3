#include "infer/infer.hpp"
#include "infer/sam3infer.hpp"
#include <acl/acl.h>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace
{

std::once_flag g_acl_init_once;
aclError g_acl_init_result = static_cast<aclError>(-1);

int configured_device_id()
{
    const char* value = std::getenv("ASCEND_DEVICE_ID");
    if (value == nullptr || *value == '\0')
    {
        return 0;
    }

    errno = 0;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX)
    {
        std::cerr << "Invalid ASCEND_DEVICE_ID='" << value << "'" << std::endl;
        return -1;
    }
    return static_cast<int>(parsed);
}

} // namespace

std::shared_ptr<Infer> load(const ModelPaths& paths)
{
    std::call_once(g_acl_init_once, []() {
        g_acl_init_result = aclInit(nullptr);
    });
    if (g_acl_init_result != ACL_SUCCESS)
    {
        std::cerr << "aclInit failed: " << g_acl_init_result << std::endl;
        return nullptr;
    }

    int device_id = configured_device_id();
    if (device_id < 0)
    {
        return nullptr;
    }

    uint32_t device_count = 0;
    aclError ret = aclrtGetDeviceCount(&device_count);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "aclrtGetDeviceCount failed: " << ret << std::endl;
        return nullptr;
    }
    std::cout << "Visible Ascend device count: " << device_count << std::endl;
    if (device_count == 0)
    {
        std::cerr << "No Ascend device is visible to the process" << std::endl;
        return nullptr;
    }
    if (static_cast<uint32_t>(device_id) >= device_count)
    {
        std::cerr << "ASCEND_DEVICE_ID=" << device_id
                  << " is outside the visible range [0, "
                  << device_count - 1 << "]" << std::endl;
        return nullptr;
    }

    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE)
    {
        std::cerr << "aclrtSetDevice(" << device_id << ") failed: " << ret << std::endl;
        return nullptr;
    }
    std::cout << "Using Ascend device " << device_id << std::endl;

    auto infer = std::make_shared<Sam3Infer>(paths, device_id);
    if (!infer->initialize())
    {
        std::cerr << "Sam3Infer initialize failed" << std::endl;
        return nullptr;
    }

    return infer;
}
