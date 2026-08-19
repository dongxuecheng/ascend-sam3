#include "infer/modelText.hpp"
#include <chrono>
#include <iostream>

static double now_ms()
{
    using namespace std::chrono;
    return duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename T>
static void fill_input(void* dev_buf, const std::vector<int64_t>& values)
{
    std::vector<T> typed(values.begin(), values.end());
    CHECK_ACL(aclrtMemcpy(dev_buf, typed.size() * sizeof(T), typed.data(), typed.size() * sizeof(T),
                          ACL_MEMCPY_HOST_TO_DEVICE));
}

aclError TextModel::encode(const std::vector<int64_t>& input_ids, const std::vector<int64_t>& attention_mask)
{
    if (input_ids.size() != 32 || attention_mask.size() != 32)
    {
        std::cerr << "TextModel input_ids/attention_mask length must be 32" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (input_count() != 2)
    {
        std::cerr << "TextModel expects 2 inputs, got " << input_count() << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    size_t input0_size = input_size(0);
    size_t input1_size = input_size(1);
    if (input0_size != input1_size)
    {
        std::cerr << "TextModel input size mismatch: input_ids=" << input0_size
                  << ", attention_mask=" << input1_size << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }
    if (input0_size == 32 * sizeof(int32_t))
    {
        fill_input<int32_t>(input_buffer(0), input_ids);
        fill_input<int32_t>(input_buffer(1), attention_mask);
    }
    else if (input0_size == 32 * sizeof(int64_t))
    {
        fill_input<int64_t>(input_buffer(0), input_ids);
        fill_input<int64_t>(input_buffer(1), attention_mask);
    }
    else
    {
        std::cerr << "TextModel input size " << input0_size << " not match int32/int64" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    double t0 = now_ms();
    aclError ret = execute();
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "TextModel execute failed" << std::endl;
        return ret;
    }

    synchronize();
    double t1 = now_ms();
    std::cout << "[Time] Text inference: " << (t1 - t0) << " ms" << std::endl;
    return ACL_SUCCESS;
}

void* TextModel::text_features_ptr() const
{
    return output_buffer(0);
}

void* TextModel::text_mask_ptr() const
{
    return output_buffer(1);
}

size_t TextModel::text_features_size() const
{
    return output_size(0);
}

size_t TextModel::text_mask_size() const
{
    return output_size(1);
}
