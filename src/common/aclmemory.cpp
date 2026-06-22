#include "common/aclmemory.hpp"

AclBaseMemory::AclBaseMemory(void* cpu, size_t cpu_bytes, void* npu, size_t npu_bytes)
{
    reference(cpu, cpu_bytes, npu, npu_bytes);
}

void AclBaseMemory::reference(void* cpu, size_t cpu_bytes, void* npu, size_t npu_bytes)
{
    release();
    if (cpu == nullptr || cpu_bytes == 0)
    {
        cpu       = nullptr;
        cpu_bytes = 0;
    }
    if (npu == nullptr || npu_bytes == 0)
    {
        npu       = nullptr;
        npu_bytes = 0;
    }
    this->cpu_          = cpu;
    this->cpu_bytes_    = cpu_bytes;
    this->cpu_capacity_ = cpu_bytes;

    this->npu_          = npu;
    this->npu_bytes_    = npu_bytes;
    this->npu_capacity_ = npu_bytes;

    this->owner_cpu_ = !(cpu && cpu_bytes > 0);
    this->owner_npu_ = !(npu && npu_bytes > 0);
}

AclBaseMemory::~AclBaseMemory()
{
    release();
}

void* AclBaseMemory::npu_realloc(size_t bytes)
{
    if (npu_capacity_ < bytes)
    {
        release_npu();

        npu_capacity_ = bytes;
        CHECK_ACL_RUN(aclrtMalloc(&npu_, npu_capacity_, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    npu_bytes_ = bytes;
    return npu_;
}

void* AclBaseMemory::cpu_realloc(size_t bytes)
{
    if (cpu_capacity_ < bytes)
    {
        release_cpu();

        cpu_capacity_ = bytes;
        CHECK_ACL_RUN(aclrtMallocHost(&cpu_, cpu_capacity_));
    }
    cpu_bytes_ = bytes;
    return cpu_;
}

void AclBaseMemory::release_npu()
{
    if (npu_)
    {
        if (owner_npu_)
        {
            CHECK_ACL_RUN(aclrtFree(npu_));
        }
        npu_ = nullptr;
    }
    npu_bytes_    = 0;
    npu_capacity_ = 0;
}

void AclBaseMemory::release_cpu()
{
    if (cpu_)
    {
        if (owner_cpu_)
        {
            CHECK_ACL_RUN(aclrtFreeHost(cpu_));
        }
        cpu_ = nullptr;
    }
    cpu_bytes_    = 0;
    cpu_capacity_ = 0;
}

void AclBaseMemory::release()
{
    release_npu();
    release_cpu();
}