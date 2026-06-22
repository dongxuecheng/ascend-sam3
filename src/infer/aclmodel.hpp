#pragma once

#ifndef ACLMODEL_HPP__
#define ACLMODEL_HPP__

#include "common/aclmemory.hpp"
#include <acl/acl.h>
#include <memory>
#include <string>
#include <vector>

#define CHECK_ACL(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        auto __ret = (call);                                                                                           \
        if (__ret != ACL_SUCCESS)                                                                                      \
        {                                                                                                              \
            std::cerr << "ACL " << #call << " FAILED, ERROR: " << __ret << " at " << __FILE__ << ":" << __LINE__      \
                      << std::endl;                                                                                    \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

/**
 * @brief Ascend ACL 模型推理基类（仅使用原生 ACL 接口，不依赖 AclLite）
 *
 * 负责 OM 模型加载、输入/输出 dataset 创建与销毁、执行推理。
 * 所有输入/输出内存均分配在 NPU 显存上，便于子模型之间零拷贝传递。
 */
class AclModel
{
  public:
    AclModel();
    virtual ~AclModel();

    AclModel(const AclModel&)            = delete;
    AclModel& operator=(const AclModel&) = delete;

    /**
     * @brief 从文件加载模型并创建输入输出 dataset
     * @param model_path om 模型路径
     * @return ACL_SUCCESS 成功，其他失败
     */
    aclError init(const std::string& model_path);

    /**
     * @brief 执行一次推理
     * @return ACL_SUCCESS 成功
     */
    aclError execute();

    /**
     * @brief 同步 stream
     */
    void synchronize() const;

    inline uint32_t input_count() const
    {
        return input_count_;
    }
    inline uint32_t output_count() const
    {
        return output_count_;
    }

    void* input_buffer(size_t idx) const;
    void* output_buffer(size_t idx) const;
    size_t input_size(size_t idx) const;
    size_t output_size(size_t idx) const;

    inline aclmdlDesc* desc() const
    {
        return model_desc_;
    }

  protected:
    void destroy_resource();

    uint32_t model_id_      = 0;
    aclmdlDesc* model_desc_ = nullptr;
    aclmdlDataset* input_dataset_  = nullptr;
    aclmdlDataset* output_dataset_ = nullptr;

    uint32_t input_count_  = 0;
    uint32_t output_count_ = 0;

    std::vector<void*> input_buffers_;
    std::vector<void*> output_buffers_;
    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;

    aclrtStream stream_ = nullptr;
};

#endif // ACLMODEL_HPP__
