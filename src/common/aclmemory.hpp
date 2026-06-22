#pragma once

#ifndef ACLMEMORY_HPP
#define ACLMEMORY_HPP

#include <acl/acl.h>
#include <iostream>

#define CHECK_ACL_RUN(call)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        auto __call__ret_code = (call);                                                                                \
        if (__call__ret_code != ACL_SUCCESS)                                                                           \
        {                                                                                                              \
            std::cerr << "ACL RUNTIME " << #call << " FAILED, ERROR CODE : " << __call__ret_code << std::endl;         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

/**
 * @brief 对ascend 内存操作的二次封装，简化流程，方便使用
 *
 */
class AclBaseMemory
{
  public:
    AclBaseMemory() = default;
    AclBaseMemory(void* cpu, size_t cpu_bytes, void* npu, size_t npu_bytes);
    virtual ~AclBaseMemory();
    virtual void* npu_realloc(size_t bytes);
    virtual void* cpu_realloc(size_t bytes);
    void release_npu();
    void release_cpu();
    void release();
    inline bool owner_cpu() const
    {
        return owner_cpu_;
    }
    inline bool owner_npu() const
    {
        return owner_npu_;
    }
    inline size_t cpu_bytes() const
    {
        return cpu_bytes_;
    }
    inline size_t npu_bytes() const
    {
        return npu_bytes_;
    }
    inline size_t cpu_capacity() const
    {
        return cpu_capacity_;
    }
    inline size_t npu_capacity() const
    {
        return npu_capacity_;
    }
    virtual inline void* get_cpu() const
    {
        return cpu_;
    }
    virtual inline void* get_npu() const
    {
        return npu_;
    }

    void reference(void* cpu, size_t cpu_bytes, void* npu, size_t npu_bytes);

  protected:
    void* cpu_        = nullptr;
    size_t cpu_bytes_ = 0, cpu_capacity_ = 0;
    bool owner_cpu_ = true;

    void* npu_        = nullptr;
    size_t npu_bytes_ = 0, npu_capacity_ = 0;
    bool owner_npu_ = true;
};

template <typename _DT> class AclMemory : public AclBaseMemory
{
  public:
    AclMemory()                            = default;
    AclMemory(const AclMemory&)            = delete;
    AclMemory(AclMemory&&)                 = delete;
    AclMemory& operator=(const AclMemory&) = delete;

    virtual _DT* npu(size_t size)
    {
        return (_DT*)npu_realloc(size * sizeof(_DT));
    }
    virtual _DT* cpu(size_t size)
    {
        return (_DT*)cpu_realloc(size * sizeof(_DT));
    }

    inline size_t cpu_size() const
    {
        return cpu_bytes_ / sizeof(_DT);
    }
    inline size_t npu_size() const
    {
        return npu_bytes_ / sizeof(_DT);
    }

    virtual inline _DT* npu() const
    {
        return (_DT*)get_npu();
    }
    virtual inline _DT* cpu() const
    {
        return (_DT*)get_cpu();
    }
};

#endif