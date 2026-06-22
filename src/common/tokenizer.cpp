#include "common/tokenizer.hpp"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sam3
{

ClipTokenizer::~ClipTokenizer() = default;

static std::string load_file_as_string(const std::string& path)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f.is_open())
    {
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool ClipTokenizer::load(const std::string& tokenizer_json_path)
{
    std::string blob = load_file_as_string(tokenizer_json_path);
    if (blob.empty())
    {
        std::cerr << "Failed to load tokenizer.json: " << tokenizer_json_path << std::endl;
        return false;
    }

    try
    {
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to create tokenizer from JSON: " << e.what() << std::endl;
        return false;
    }

    if (tokenizer_ == nullptr)
    {
        std::cerr << "Tokenizer::FromBlobJSON returned null" << std::endl;
        return false;
    }

    // 读取 CLIP 特殊 token id，失败则使用默认值
    bos_id_ = tokenizer_->TokenToId("<|startoftext|>");
    eos_id_ = tokenizer_->TokenToId("<|endoftext|>");
    pad_id_ = 0;

    if (bos_id_ < 0)
    {
        std::cerr << "Warning: <|startoftext|> not found in tokenizer, using default 49406" << std::endl;
        bos_id_ = 49406;
    }
    if (eos_id_ < 0)
    {
        std::cerr << "Warning: <|endoftext|> not found in tokenizer, using default 49407" << std::endl;
        eos_id_ = 49407;
    }

    std::cout << "Tokenizer loaded: " << tokenizer_json_path
              << ", bos=" << bos_id_ << ", eos=" << eos_id_ << ", pad=" << pad_id_ << std::endl;
    return true;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> ClipTokenizer::encode(
    const std::string& text, size_t max_length) const
{
    std::vector<int64_t> input_ids(max_length, pad_id_);
    std::vector<int64_t> attention_mask(max_length, 0);

    if (tokenizer_ == nullptr || max_length == 0)
    {
        return {input_ids, attention_mask};
    }

    // 先不带特殊 token 编码文本，再手动拼接 BOS/EOS
    std::vector<int32_t> tokens = tokenizer_->Encode(text);

    std::vector<int32_t> ids;
    ids.reserve(max_length);
    ids.push_back(bos_id_);
    for (int32_t t : tokens)
    {
        ids.push_back(t);
    }
    ids.push_back(eos_id_);

    if (ids.size() > max_length)
    {
        ids.resize(max_length);
        ids.back() = eos_id_; // 截断后确保最后是 EOS
    }

    for (size_t i = 0; i < ids.size() && i < max_length; ++i)
    {
        input_ids[i]     = static_cast<int64_t>(ids[i]);
        attention_mask[i] = 1;
    }

    return {input_ids, attention_mask};
}

} // namespace sam3
