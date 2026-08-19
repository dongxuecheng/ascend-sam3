#include "infer/aclmodel.hpp"
#include <iostream>

AclModel::AclModel()
{
}

AclModel::~AclModel()
{
    destroy_resource();
}

aclError AclModel::init(const std::string& model_path)
{
    aclError ret = aclmdlLoadFromFile(model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Load model " << model_path << " failed, error: " << ret << std::endl;
        return ret;
    }
    model_loaded_ = true;

    model_desc_ = aclmdlCreateDesc();
    if (model_desc_ == nullptr)
    {
        std::cerr << "Create model desc failed" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Get model desc failed, error: " << ret << std::endl;
        return ret;
    }

    input_count_  = aclmdlGetNumInputs(model_desc_);
    output_count_ = aclmdlGetNumOutputs(model_desc_);

    input_dataset_  = aclmdlCreateDataset();
    output_dataset_ = aclmdlCreateDataset();
    if (input_dataset_ == nullptr || output_dataset_ == nullptr)
    {
        std::cerr << "Create dataset failed" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    input_buffers_.resize(input_count_, nullptr);
    output_buffers_.resize(output_count_, nullptr);
    input_sizes_.resize(input_count_, 0);
    output_sizes_.resize(output_count_, 0);

    for (uint32_t i = 0; i < input_count_; ++i)
    {
        size_t size = aclmdlGetInputSizeByIndex(model_desc_, i);
        input_sizes_[i] = size;
        void* dev_buf = nullptr;
        ret = aclrtMalloc(&dev_buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Malloc input " << i << " buffer failed, size=" << size << ", error=" << ret << std::endl;
            return ret;
        }
        input_buffers_[i] = dev_buf;

        aclDataBuffer* data_buf = aclCreateDataBuffer(dev_buf, size);
        if (data_buf == nullptr)
        {
            std::cerr << "Create input data buffer " << i << " failed" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }
        ret = aclmdlAddDatasetBuffer(input_dataset_, data_buf);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Add input dataset buffer " << i << " failed" << std::endl;
            return ret;
        }
    }

    for (uint32_t i = 0; i < output_count_; ++i)
    {
        size_t size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        output_sizes_[i] = size;
        void* dev_buf = nullptr;
        ret = aclrtMalloc(&dev_buf, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Malloc output " << i << " buffer failed, size=" << size << ", error=" << ret << std::endl;
            return ret;
        }
        output_buffers_[i] = dev_buf;

        aclDataBuffer* data_buf = aclCreateDataBuffer(dev_buf, size);
        if (data_buf == nullptr)
        {
            std::cerr << "Create output data buffer " << i << " failed" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }
        ret = aclmdlAddDatasetBuffer(output_dataset_, data_buf);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Add output dataset buffer " << i << " failed" << std::endl;
            return ret;
        }
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Create stream failed, error: " << ret << std::endl;
        return ret;
    }

    return ACL_SUCCESS;
}

aclError AclModel::execute()
{
    aclError ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Execute model failed, error: " << ret << std::endl;
        return ret;
    }
    return ACL_SUCCESS;
}

void AclModel::synchronize() const
{
    if (stream_ != nullptr)
    {
        aclrtSynchronizeStream(stream_);
    }
}

void* AclModel::input_buffer(size_t idx) const
{
    return idx < input_buffers_.size() ? input_buffers_[idx] : nullptr;
}

void* AclModel::output_buffer(size_t idx) const
{
    return idx < output_buffers_.size() ? output_buffers_[idx] : nullptr;
}

size_t AclModel::input_size(size_t idx) const
{
    return idx < input_sizes_.size() ? input_sizes_[idx] : 0;
}

size_t AclModel::output_size(size_t idx) const
{
    return idx < output_sizes_.size() ? output_sizes_[idx] : 0;
}

void AclModel::destroy_resource()
{
    if (model_loaded_)
    {
        aclmdlUnload(model_id_);
        model_id_ = 0;
        model_loaded_ = false;
    }

    if (input_dataset_ != nullptr)
    {
        size_t buf_cnt = aclmdlGetDatasetNumBuffers(input_dataset_);
        for (size_t i = 0; i < buf_cnt; ++i)
        {
            aclDataBuffer* buf = aclmdlGetDatasetBuffer(input_dataset_, i);
            if (buf != nullptr)
            {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(input_dataset_);
        input_dataset_ = nullptr;
    }

    if (output_dataset_ != nullptr)
    {
        size_t buf_cnt = aclmdlGetDatasetNumBuffers(output_dataset_);
        for (size_t i = 0; i < buf_cnt; ++i)
        {
            aclDataBuffer* buf = aclmdlGetDatasetBuffer(output_dataset_, i);
            if (buf != nullptr)
            {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(output_dataset_);
        output_dataset_ = nullptr;
    }

    for (void* ptr : input_buffers_)
    {
        if (ptr != nullptr)
        {
            aclrtFree(ptr);
        }
    }
    input_buffers_.clear();

    for (void* ptr : output_buffers_)
    {
        if (ptr != nullptr)
        {
            aclrtFree(ptr);
        }
    }
    output_buffers_.clear();

    if (model_desc_ != nullptr)
    {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }

    if (stream_ != nullptr)
    {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}
