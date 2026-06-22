#include "infer/infer.hpp"
#include "infer/sam3infer.hpp"
#include <acl/acl.h>
#include <iostream>

static bool g_acl_initialized = false;

std::shared_ptr<Infer> load(const ModelPaths& paths)
{
    if (!g_acl_initialized)
    {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "aclInit failed: " << ret << std::endl;
            return nullptr;
        }
        g_acl_initialized = true;
    }

    aclError ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE)
    {
        std::cerr << "aclrtSetDevice failed: " << ret << std::endl;
        return nullptr;
    }

    auto infer = std::make_shared<Sam3Infer>(paths);
    if (!infer->initialize())
    {
        std::cerr << "Sam3Infer initialize failed" << std::endl;
        return nullptr;
    }

    return infer;
}
