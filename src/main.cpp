#include "infer/infer.hpp"
#include "infer/sam3type.hpp"
#include "common/tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

/**
 * @brief 简单示例：加载 SAM3 模型，对单张图片推理并可视化保存结果
 *
 * 用法：
 *   ./ascendsam3_demo <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> [prompt_text] [output_path]
 *
 * 其中：
 *   vision_model   - vision encoder .om
 *   text_model     - text encoder .om
 *   decoder_model  - decoder .om
 *   fpn_pos2_npy   - fpn_pos_2_constant.npy
 *   tokenizer_json - HuggingFace tokenizer.json
 *   prompt_text    - 文本 prompt，例如 "a person"；为空时使用默认占位 prompt
 */

const int tokenizer_length = 32;

static void visualize(cv::Mat& image, const object::DetectionBoxArray& boxes)
{
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 114, 189),
        cv::Scalar(217, 83, 25),
        cv::Scalar(237, 177, 32),
        cv::Scalar(126, 47, 142),
        cv::Scalar(119, 172, 48),
    };

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& box = boxes[i];
        const cv::Scalar& color = colors[i % colors.size()];

        int left   = static_cast<int>(box.box.left);
        int top    = static_cast<int>(box.box.top);
        int right  = static_cast<int>(box.box.right);
        int bottom = static_cast<int>(box.box.bottom);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 2);

        std::string label = box.class_name + " " + std::to_string(box.score).substr(0, 4);
        int baseline      = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(left, top - text_size.height - 4),
                      cv::Point(left + text_size.width, top),
                      color,
                      -1);
        cv::putText(image,
                    label,
                    cv::Point(left, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    1);

        if (box.segmentation.has_value())
        {
            const cv::Mat& mask = box.segmentation.value().mask;
            if (!mask.empty())
            {
                // mask 已经是相对于目标框的，需要先对齐到原图坐标再叠加
                object::Segmentation aligned = box.segmentation.value().align_to_left_top(
                    left, top, image.cols, image.rows);
                if (!aligned.mask.empty() && aligned.mask.size() == image.size())
                {
                    cv::Mat color_mask(image.size(), image.type(), color);
                    cv::Mat overlay;
                    color_mask.copyTo(overlay, aligned.mask);
                    cv::addWeighted(image, 1.0, overlay, 0.5, 0.0, image);
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 7 || argc > 9)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> [prompt_text] [output_path]"
                  << std::endl;
        return 1;
    }

    ModelPaths paths;
    paths.vision_model  = argv[1];
    paths.text_model    = argv[2];
    paths.decoder_model = argv[3];
    paths.fpn_pos2      = argv[4];
    std::string tokenizer_path = argv[5];
    std::string image_path     = argv[6];
    std::string prompt_arg     = (argc >= 8) ? argv[7] : "";
    std::string output_path    = (argc == 9) ? argv[8] : "workspace/result.jpg";

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    std::shared_ptr<Infer> infer = load(paths);
    if (infer == nullptr)
    {
        std::cerr << "Failed to load models" << std::endl;
        return 1;
    }

    auto input = std::make_shared<Sam3Input>();
    input->image = image;
    input->confidence_threshold = 0.3f;
    if (const char* env_conf = std::getenv("SAM3_CONFIDENCE"))
    {
        input->confidence_threshold = std::stof(env_conf);
    }
    if (const char* env_mask = std::getenv("SAM3_NEED_MASK"))
    {
        input->need_mask = (std::atoi(env_mask) != 0);
    }

    // 使用 tokenizers-cpp 从文本生成 token ids
    sam3::ClipTokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path))
    {
        std::cerr << "Failed to load tokenizer from " << tokenizer_path << std::endl;
        return 1;
    }

    if (!prompt_arg.empty())
    {
        // 支持多个类别用英文逗号分隔，例如 "person,car"
        std::stringstream ss(prompt_arg);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (token.empty()) continue;
            TextPrompt prompt;
            prompt.text = token;
            std::tie(prompt.input_ids, prompt.attention_mask) = tokenizer.encode(token, tokenizer_length);
            input->text_prompts.push_back(prompt);
        }
    }
    else
    {
        // 默认占位 prompt，仅用于测试模型能否跑通，不会得到真实检测结果
        TextPrompt prompt;
        prompt.text    = "a person";
        prompt.input_ids.resize(tokenizer_length, 0);
        prompt.attention_mask.resize(tokenizer_length, 1);
        prompt.input_ids[0] = 49406; // <|startoftext|>
        prompt.input_ids[1] = 320;   // a
        prompt.input_ids[2] = 2533;  // person
        prompt.input_ids[3] = 49407; // 
        input->text_prompts.push_back(prompt);
    }

    object::DetectionBoxArray boxes = infer->forward(input);

    std::cout << "Detected " << boxes.size() << " objects:" << std::endl;
    for (const auto& box : boxes)
    {
        std::cout << box << std::endl;
    }

    cv::Mat vis_image = image.clone();
    visualize(vis_image, boxes);

    if (cv::imwrite(output_path, vis_image))
    {
        std::cout << "Result saved to " << output_path << std::endl;
    }
    else
    {
        std::cerr << "Failed to save result to " << output_path << std::endl;
    }

    infer.reset();
    return 0;
}
