#ifndef SAM3_TOKENIZER_HPP__
#define SAM3_TOKENIZER_HPP__

#include <cstdint>
#include <memory>
#include <string>
#include <tokenizers_cpp.h>
#include <utility>
#include <vector>

namespace sam3
{

/**
 * @brief CLIP 风格 tokenizer 封装（基于 tokenizers-cpp）
 *
 * 加载 HuggingFace tokenizer.json，对输入文本做如下处理：
 *   BOS + tokens + EOS + PAD(0)
 * 并生成对应长度的 attention_mask。
 */
class ClipTokenizer
{
  public:
    ClipTokenizer() = default;
    ~ClipTokenizer();

    /**
     * @brief 从 tokenizer.json 文件路径加载
     * @param tokenizer_json_path HuggingFace tokenizer.json 路径
     * @return true 成功
     */
    bool load(const std::string& tokenizer_json_path);

    /**
     * @brief 是否已经加载成功
     */
    bool ok() const { return tokenizer_ != nullptr; }

    /**
     * @brief 编码文本
     * @param text 输入文本，例如 "a person"
     * @param max_length 固定输出长度，默认 32
     * @return pair<input_ids, attention_mask>，长度均为 max_length
     */
    std::pair<std::vector<int64_t>, std::vector<int64_t>> encode(
        const std::string& text, size_t max_length = 32) const;

  private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    int32_t bos_id_ = 49406;
    int32_t eos_id_ = 49407;
    int32_t pad_id_ = 0;
};

} // namespace sam3

#endif // SAM3_TOKENIZER_HPP__
