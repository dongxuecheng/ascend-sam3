#include "infer/sam3infer.hpp"
#include "common/npy_utils.hpp"
#include "common/object.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

static double now_ms()
{
    using namespace std::chrono;
    return duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static constexpr int MAX_MASKS   = 200;
static constexpr int MASK_SIZE   = 288;
static constexpr int TEXT_LENGTH = 32;
static constexpr int TEXT_DIM    = 256;

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

Sam3Infer::Sam3Infer(const ModelPaths& paths)
    : paths_(paths)
    , vision_model_(std::make_unique<VisionModel>())
    , text_model_(std::make_unique<TextModel>())
    , decoder_model_(std::make_unique<DecoderModel>())
{
}

Sam3Infer::~Sam3Infer()
{
    if (fpn_pos_2_buf_ != nullptr)
    {
        aclrtFree(fpn_pos_2_buf_);
        fpn_pos_2_buf_ = nullptr;
    }
    if (ext_text_features_buf_ != nullptr)
    {
        aclrtFree(ext_text_features_buf_);
        ext_text_features_buf_ = nullptr;
    }
    if (ext_text_mask_buf_ != nullptr)
    {
        aclrtFree(ext_text_mask_buf_);
        ext_text_mask_buf_ = nullptr;
    }
}

static bool file_exists(const std::string& path)
{
    std::ifstream f(path);
    return f.good();
}

bool Sam3Infer::initialize()
{
    // 每个模型都通过命令行传入的固定文件路径加载，不存在则直接报错
    const std::string& vision_path  = paths_.vision_model;
    const std::string& text_path    = paths_.text_model;
    const std::string& decoder_path = paths_.decoder_model;

    if (!file_exists(vision_path))
    {
        std::cerr << "Vision model not found: " << vision_path << std::endl;
        return false;
    }
    if (!file_exists(text_path))
    {
        std::cerr << "Text model not found: " << text_path << std::endl;
        return false;
    }
    if (!file_exists(decoder_path))
    {
        std::cerr << "Decoder model not found: " << decoder_path << std::endl;
        return false;
    }

    std::cout << "Using models: " << vision_path << ", " << text_path << ", " << decoder_path << std::endl;

    aclError ret = vision_model_->init(vision_path);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Init vision model failed: " << ret << std::endl;
        return false;
    }

    ret = text_model_->init(text_path);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Init text model failed: " << ret << std::endl;
        return false;
    }

    ret = decoder_model_->init(decoder_path);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Init decoder model failed: " << ret << std::endl;
        return false;
    }

    if (!load_fpn_pos_2())
    {
        std::cerr << "Load fpn_pos_2_constant.npy failed" << std::endl;
        return false;
    }

    return true;
}

bool Sam3Infer::load_fpn_pos_2()
{
    const std::string& npy_path = paths_.fpn_pos2;
    std::vector<float> data;
    std::vector<size_t> shape;
    if (!load_npy_float32(npy_path, data, shape))
    {
        std::cerr << "Failed to load " << npy_path << std::endl;
        return false;
    }

    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 256 || shape[2] != 72 || shape[3] != 72)
    {
        std::cerr << "fpn_pos_2 shape mismatch, expect (1,256,72,72), got (";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::cerr << shape[i] << (i + 1 == shape.size() ? "" : ",");
        }
        std::cerr << ")" << std::endl;
        return false;
    }

    fpn_pos_2_size_ = data.size() * sizeof(float);
    aclError ret    = aclrtMalloc(&fpn_pos_2_buf_, fpn_pos_2_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Malloc fpn_pos_2 device buffer failed: " << ret << std::endl;
        return false;
    }

    ret = aclrtMemcpy(fpn_pos_2_buf_, fpn_pos_2_size_, data.data(), fpn_pos_2_size_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Upload fpn_pos_2 failed: " << ret << std::endl;
        return false;
    }

    std::cout << "Load fpn_pos_2_constant.npy success, shape (1,256,72,72), size " << fpn_pos_2_size_ << " bytes"
              << std::endl;
    return true;
}

aclError Sam3Infer::upload_external_text(const ExternalTextFeature& ext, void*& text_features_buf, void*& text_mask_buf)
{
    size_t need_feature_size = 1ULL * TEXT_LENGTH * TEXT_DIM * sizeof(float);
    size_t need_mask_size    = 1ULL * TEXT_LENGTH * sizeof(uint8_t);

    if (ext.text_features.size() != TEXT_LENGTH * TEXT_DIM || ext.text_mask.size() != TEXT_LENGTH)
    {
        std::cerr << "External text feature size mismatch" << std::endl;
        return ACL_ERROR_INVALID_PARAM;
    }

    if (ext_text_features_buf_ == nullptr || ext_text_features_size_ < need_feature_size)
    {
        if (ext_text_features_buf_ != nullptr)
        {
            aclrtFree(ext_text_features_buf_);
        }
        CHECK_ACL(aclrtMalloc(&ext_text_features_buf_, need_feature_size, ACL_MEM_MALLOC_HUGE_FIRST));
        ext_text_features_size_ = need_feature_size;
    }

    if (ext_text_mask_buf_ == nullptr || ext_text_mask_size_ < need_mask_size)
    {
        if (ext_text_mask_buf_ != nullptr)
        {
            aclrtFree(ext_text_mask_buf_);
        }
        CHECK_ACL(aclrtMalloc(&ext_text_mask_buf_, need_mask_size, ACL_MEM_MALLOC_HUGE_FIRST));
        ext_text_mask_size_ = need_mask_size;
    }

    CHECK_ACL(aclrtMemcpy(ext_text_features_buf_, need_feature_size,
                          ext.text_features.data(), need_feature_size,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(ext_text_mask_buf_, need_mask_size,
                          ext.text_mask.data(), need_mask_size,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    text_features_buf = ext_text_features_buf_;
    text_mask_buf     = ext_text_mask_buf_;
    return ACL_SUCCESS;
}

object::DetectionBoxArray Sam3Infer::forward(std::shared_ptr<Sam3Input> input)
{
    double total_t0 = now_ms();
    object::DetectionBoxArray results;
    if (input == nullptr || input->image.empty())
    {
        std::cerr << "Sam3Infer forward got empty input" << std::endl;
        return results;
    }

    // 1. Vision encoder：每次 forward 都执行
    aclError ret = vision_model_->encode(input->image);
    if (ret != ACL_SUCCESS)
    {
        std::cerr << "Vision encode failed: " << ret << std::endl;
        return results;
    }

    std::array<void*, 3> vision_features{
        vision_model_->feature_ptr(0),
        vision_model_->feature_ptr(1),
        vision_model_->feature_ptr(2)};
    std::array<size_t, 3> vision_sizes{
        vision_model_->output_size(0),
        vision_model_->output_size(1),
        vision_model_->output_size(2)};

    auto process_one = [&](const std::string& class_name,
                           void* text_features, void* text_mask,
                           size_t text_feature_size, size_t text_mask_size) -> bool {
        aclError ret = decoder_model_->decode(vision_features, vision_sizes,
                                              fpn_pos_2_buf_, fpn_pos_2_size_,
                                              text_features, text_feature_size,
                                              text_mask, text_mask_size);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Decoder execute failed: " << ret << std::endl;
            return false;
        }

        object::DetectionBoxArray batch = postprocess(
            input->image,
            input->confidence_threshold,
            input->need_mask,
            class_name,
            decoder_model_->pred_masks_ptr(),
            decoder_model_->pred_boxes_ptr(),
            decoder_model_->pred_logits_ptr(),
            decoder_model_->presence_logits_ptr());

        results.insert(results.end(), batch.begin(), batch.end());
        return true;
    };

    if (!input->external_text_features.empty())
    {
        for (const auto& ext : input->external_text_features)
        {
            void* text_features = nullptr;
            void* text_mask     = nullptr;
            aclError ret = upload_external_text(ext, text_features, text_mask);
            if (ret != ACL_SUCCESS)
            {
                std::cerr << "Upload external text feature failed: " << ret << std::endl;
                continue;
            }
            process_one(ext.class_name.empty() ? "object" : ext.class_name,
                        text_features, text_mask,
                        ext_text_features_size_, ext_text_mask_size_);
        }
    }
    else if (!input->text_prompts.empty())
    {
        for (const auto& prompt : input->text_prompts)
        {
            aclError ret = text_model_->encode(prompt.input_ids, prompt.attention_mask);
            if (ret != ACL_SUCCESS)
            {
                std::cerr << "Text encode failed: " << ret << std::endl;
                continue;
            }
            process_one(prompt.text.empty() ? "object" : prompt.text,
                        text_model_->text_features_ptr(), text_model_->text_mask_ptr(),
                        text_model_->text_features_size(), text_model_->text_mask_size());
        }
    }
    else
    {
        std::cerr << "No text prompt or external text feature provided" << std::endl;
    }

    double total_t1 = now_ms();
    std::cout << "[Time] Sam3Infer forward total: " << (total_t1 - total_t0) << " ms" << std::endl;
    return results;
}

object::DetectionBoxArray Sam3Infer::postprocess(const cv::Mat& original_image,
                                                  float confidence_threshold,
                                                  bool need_mask,
                                                  const std::string& class_name,
                                                  void* pred_masks_buf,
                                                  void* pred_boxes_buf,
                                                  void* pred_logits_buf,
                                                  void* presence_logits_buf)
{
    object::DetectionBoxArray boxes;

    const int orig_h = original_image.rows;
    const int orig_w = original_image.cols;

    std::vector<float> logits(MAX_MASKS);
    std::vector<float> boxes_raw(MAX_MASKS * 4);
    float presence_logit = 0.0f;

    CHECK_ACL(aclrtMemcpy(logits.data(), logits.size() * sizeof(float),
                          pred_logits_buf, logits.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(boxes_raw.data(), boxes_raw.size() * sizeof(float),
                          pred_boxes_buf, boxes_raw.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(&presence_logit, sizeof(float),
                          presence_logits_buf, sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    float presence_score = sigmoid(presence_logit);

    size_t mask_bytes = 1ULL * MASK_SIZE * MASK_SIZE * sizeof(float);

    std::vector<std::pair<float, int>> score_indices;
    for (int idx = 0; idx < MAX_MASKS; ++idx)
    {
        float score = sigmoid(logits[idx]) * presence_score;
        if (score > confidence_threshold)
        {
            score_indices.emplace_back(score, idx);
        }
    }

    std::sort(score_indices.begin(), score_indices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& [score, idx] : score_indices)
    {
        const float* box_ptr = boxes_raw.data() + idx * 4;
        float x1 = box_ptr[0] * orig_w;
        float y1 = box_ptr[1] * orig_h;
        float x2 = box_ptr[2] * orig_w;
        float y2 = box_ptr[3] * orig_h;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        object::DetectionBox det;
        det.type       = object::ObjectType::OBJECT;
        det.score      = score;
        det.box        = object::Box(x1, y1, x2, y2);
        det.class_id   = 0;
        det.class_name = class_name.empty() ? "object" : class_name;

        // 解码 mask：复制 -> 阈值 -> resize 到原图 -> 裁剪到目标框
        if (need_mask)
        {
            std::vector<float> mask_raw(MASK_SIZE * MASK_SIZE);
            CHECK_ACL(aclrtMemcpy(mask_raw.data(), mask_bytes,
                                  static_cast<char*>(pred_masks_buf) + idx * mask_bytes,
                                  mask_bytes,
                                  ACL_MEMCPY_DEVICE_TO_HOST));

            cv::Mat mask_mat(MASK_SIZE, MASK_SIZE, CV_32FC1, mask_raw.data());
            cv::Mat binary_mask;
            cv::threshold(mask_mat, binary_mask, 0.0f, 255.0f, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8UC1);

            cv::Mat resized_mask;
            cv::resize(binary_mask, resized_mask, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

            // 只保留目标框内的 mask
            int roi_x = static_cast<int>(std::max(0.0f, x1));
            int roi_y = static_cast<int>(std::max(0.0f, y1));
            int roi_w = static_cast<int>(std::min(static_cast<float>(orig_w), x2)) - roi_x;
            int roi_h = static_cast<int>(std::min(static_cast<float>(orig_h), y2)) - roi_y;
            if (roi_w > 0 && roi_h > 0)
            {
                cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
                object::Segmentation seg;
                seg.mask = resized_mask(roi).clone();
                seg.keep_largest_part();
                det.segmentation = seg;
            }
        }

        boxes.push_back(det);
    }

    return boxes;
}
