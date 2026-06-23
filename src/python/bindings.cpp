#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>

#include "infer/infer.hpp"
#include "infer/sam3type.hpp"
#include "common/tokenizer.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

namespace py = pybind11;

class Sam3PyModel
{
public:
    Sam3PyModel(const std::string& vision_model,
                const std::string& text_model,
                const std::string& decoder_model,
                const std::string& fpn_pos2,
                const std::string& tokenizer_path)
    {
        ModelPaths paths;
        paths.vision_model  = vision_model;
        paths.text_model    = text_model;
        paths.decoder_model = decoder_model;
        paths.fpn_pos2      = fpn_pos2;

        infer_ = load(paths);
        if (infer_ == nullptr)
        {
            throw std::runtime_error("Failed to load SAM3 models");
        }

        if (!tokenizer_.load(tokenizer_path))
        {
            throw std::runtime_error("Failed to load tokenizer from " + tokenizer_path);
        }
    }

    py::list detect(const py::bytes& image_bytes,
                    const std::vector<std::string>& class_names,
                    float confidence,
                    bool return_mask)
    {
        std::string raw = image_bytes;
        std::vector<uint8_t> buf(raw.begin(), raw.end());
        cv::Mat image = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw std::runtime_error("Failed to decode image");
        }

        auto input = std::make_shared<Sam3Input>();
        input->image = image;
        input->confidence_threshold = confidence;
        input->need_mask = return_mask;

        if (class_names.empty())
        {
            input->text_prompts.push_back(make_prompt("person"));
        }
        else
        {
            for (const auto& name : class_names)
            {
                input->text_prompts.push_back(make_prompt(name));
            }
        }

        object::DetectionBoxArray boxes = infer_->forward(input);

        py::list result;
        for (const auto& box : boxes)
        {
            py::dict item;
            item["class_name"] = box.class_name;
            item["score"]      = box.score;

            py::dict bbox;
            bbox["left"]   = box.box.left;
            bbox["top"]    = box.box.top;
            bbox["right"]  = box.box.right;
            bbox["bottom"] = box.box.bottom;
            item["box"] = bbox;

            if (return_mask && box.segmentation.has_value())
            {
                const cv::Mat& mask = box.segmentation.value().mask;
                if (!mask.empty())
                {
                    std::vector<uint8_t> png_buf;
                    cv::imencode(".png", mask, png_buf);
                    item["mask_png"] = py::bytes(reinterpret_cast<const char*>(png_buf.data()), png_buf.size());
                    item["mask_width"]  = mask.cols;
                    item["mask_height"] = mask.rows;
                }
            }

            result.append(item);
        }

        return result;
    }

private:
    TextPrompt make_prompt(const std::string& text)
    {
        TextPrompt prompt;
        prompt.text = text;
        std::tie(prompt.input_ids, prompt.attention_mask) = tokenizer_.encode(text, 32);
        return prompt;
    }

    std::shared_ptr<Infer> infer_;
    sam3::ClipTokenizer tokenizer_;
};

PYBIND11_MODULE(ascendsam3, m)
{
    m.doc() = "SAM3 Ascend NPU inference Python bindings";

    py::class_<Sam3PyModel>(m, "Sam3Model")
        .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&, const std::string&>(),
             py::arg("vision_model"),
             py::arg("text_model"),
             py::arg("decoder_model"),
             py::arg("fpn_pos2"),
             py::arg("tokenizer_path"))
        .def("detect", &Sam3PyModel::detect,
             py::arg("image_bytes"),
             py::arg("class_names") = std::vector<std::string>{"person"},
             py::arg("confidence") = 0.3f,
             py::arg("return_mask") = true,
             "Detect objects in an image. Returns a list of detection dicts.");
}
