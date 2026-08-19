#include "infer/modelDecoder.hpp"
#include <chrono>
#include <iostream>

static double now_ms()
{
    using namespace std::chrono;
    return duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

aclError DecoderModel::decode(const std::array<void*, 3>& vision_features,
                              const std::array<size_t, 3>& vision_sizes,
                              void* fpn_pos_2,
                              size_t fpn_pos_2_size,
                              void* text_features,
                              size_t text_features_size,
                              void* text_mask,
                              size_t text_mask_size)
{
    if (input_count() != 6)
    {
        std::cerr << "DecoderModel expects 6 inputs, got " << input_count() << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    const std::array<size_t, 6> provided_sizes{
        vision_sizes[0], vision_sizes[1], vision_sizes[2],
        fpn_pos_2_size, text_features_size, text_mask_size};
    for (size_t i = 0; i < provided_sizes.size(); ++i)
    {
        if (provided_sizes[i] != input_size(i))
        {
            std::cerr << "Decoder input " << i << " size mismatch: OM expects " << input_size(i)
                      << " bytes, provided " << provided_sizes[i] << " bytes" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    aclmdlDataset* external_input = aclmdlCreateDataset();
    if (external_input == nullptr)
    {
        std::cerr << "Create external input dataset failed" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    auto add_buffer = [&](void* ptr, size_t size) -> aclError {
        aclDataBuffer* buf = aclCreateDataBuffer(ptr, size);
        if (buf == nullptr)
        {
            std::cerr << "Create external data buffer failed" << std::endl;
            return ACL_ERROR_INVALID_PARAM;
        }
        aclError ret = aclmdlAddDatasetBuffer(external_input, buf);
        if (ret != ACL_SUCCESS)
        {
            aclDestroyDataBuffer(buf);
            return ret;
        }
        return ACL_SUCCESS;
    };

    aclError ret = ACL_SUCCESS;
    for (int i = 0; i < 3 && ret == ACL_SUCCESS; ++i)
    {
        ret = add_buffer(vision_features[i], vision_sizes[i]);
    }
    if (ret == ACL_SUCCESS)
    {
        ret = add_buffer(fpn_pos_2, fpn_pos_2_size);
    }
    if (ret == ACL_SUCCESS)
    {
        ret = add_buffer(text_features, text_features_size);
    }
    if (ret == ACL_SUCCESS)
    {
        ret = add_buffer(text_mask, text_mask_size);
    }

    if (ret != ACL_SUCCESS)
    {
        size_t buf_cnt = aclmdlGetDatasetNumBuffers(external_input);
        for (size_t i = 0; i < buf_cnt; ++i)
        {
            aclDataBuffer* buf = aclmdlGetDatasetBuffer(external_input, i);
            if (buf != nullptr)
            {
                aclDestroyDataBuffer(buf);
            }
        }
        aclmdlDestroyDataset(external_input);
        return ret;
    }

    aclmdlDataset* origin_input = input_dataset_;
    input_dataset_              = external_input;

    double t0 = now_ms();
    ret = execute();
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "DecoderModel execute failed" << std::endl;
    }

    synchronize();
    double t1 = now_ms();
    std::cout << "[Time] Decoder inference: " << (t1 - t0) << " ms" << std::endl;

    input_dataset_ = origin_input;

    size_t buf_cnt = aclmdlGetDatasetNumBuffers(external_input);
    for (size_t i = 0; i < buf_cnt; ++i)
    {
        aclDataBuffer* buf = aclmdlGetDatasetBuffer(external_input, i);
        if (buf != nullptr)
        {
            aclDestroyDataBuffer(buf);
        }
    }
    aclmdlDestroyDataset(external_input);

    return ret;
}

void* DecoderModel::pred_masks_ptr() const
{
    return output_buffer(0);
}

void* DecoderModel::pred_boxes_ptr() const
{
    return output_buffer(1);
}

void* DecoderModel::pred_logits_ptr() const
{
    return output_buffer(2);
}

void* DecoderModel::presence_logits_ptr() const
{
    return output_buffer(3);
}
