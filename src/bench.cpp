#include "infer/infer.hpp"
#include "infer/sam3type.hpp"
#include "common/tokenizer.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <tuple>
#include <vector>

static double now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    if (argc != 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <vision_model> <text_model> <decoder_model> <fpn_pos2_npy> <tokenizer_json> <image_path> <prompt_text>"
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
    std::string prompt_arg     = argv[7];

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

    sam3::ClipTokenizer tokenizer;
    if (!tokenizer.load(tokenizer_path))
    {
        std::cerr << "Failed to load tokenizer from " << tokenizer_path << std::endl;
        return 1;
    }

    auto input = std::make_shared<Sam3Input>();
    input->image = image;
    input->confidence_threshold = 0.3f;

    TextPrompt prompt;
    prompt.text = prompt_arg;
    std::tie(prompt.input_ids, prompt.attention_mask) = tokenizer.encode(prompt_arg, 32);
    input->text_prompts.push_back(prompt);

    const int warmup_iters = 10;
    const int bench_iters  = 100;

    std::cout << "Warmup " << warmup_iters << " iters..." << std::endl;
    for (int i = 0; i < warmup_iters; ++i)
    {
        infer->forward(input);
    }

    std::cout << "Benchmark " << bench_iters << " iters..." << std::endl;
    std::vector<double> times;
    times.reserve(bench_iters);

    double total_start = now_ms();
    for (int i = 0; i < bench_iters; ++i)
    {
        double t0 = now_ms();
        infer->forward(input);
        double t1 = now_ms();
        times.push_back(t1 - t0);
    }
    double total_end = now_ms();

    double sum = 0.0;
    double min_t = times[0];
    double max_t = times[0];
    for (double t : times)
    {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double avg = sum / bench_iters;

    std::cout << "========== Benchmark Result ==========" << std::endl;
    std::cout << "Total time: " << (total_end - total_start) << " ms" << std::endl;
    std::cout << "Average:    " << avg << " ms" << std::endl;
    std::cout << "Min:        " << min_t << " ms" << std::endl;
    std::cout << "Max:        " << max_t << " ms" << std::endl;
    std::cout << "FPS:        " << (1000.0 / avg) << std::endl;
    std::cout << "======================================" << std::endl;

    infer.reset();
    return 0;
}
