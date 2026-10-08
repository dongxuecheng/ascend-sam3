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

        object::DetectionBoxArray boxes;
        {
            py::gil_scoped_release release;
            boxes = infer_->forward(input);
        }
        return serialize(boxes, return_mask);
    }

    py::dict detect_obj_refine(const py::bytes& image_bytes,
                               const std::vector<std::string>& pre_labels,
                               const std::vector<std::string>& refine_labels,
                               float confidence, bool return_mask, bool merge_results,
                               const sam3::refine::CropConfig& config,
                               float pre_detect_confidence)
    {
        config.validate();
        if (pre_labels.empty()) throw std::invalid_argument("pre_detect_labels must not be empty");
        const std::string raw = image_bytes;
        const std::vector<uint8_t> bytes(raw.begin(), raw.end());
        auto input = std::make_shared<Sam3Input>();
        input->image = cv::imdecode(bytes, cv::IMREAD_COLOR);
        if (input->image.empty()) throw std::invalid_argument("Failed to decode image");
        input->obj_refine = true;
        input->confidence_threshold = confidence;
        input->pre_detect_confidence = pre_detect_confidence;
        input->need_mask = return_mask;
        input->merge_results = merge_results;
        input->crop_config = config;
        for (const auto& label : pre_labels) input->pre_detect_prompts.push_back(make_prompt(label));
        for (const auto& label : refine_labels) input->text_prompts.push_back(make_prompt(label));
        object::DetectionBoxArray boxes;
        {
            // ACL executes outside the GIL; one instance is still serialized by
            // Sam3Infer's mutex. Health endpoints can run during long refinements.
            py::gil_scoped_release release;
            boxes = infer_->forward(input);
        }
        const auto& stats = input->refine_stats;
        py::dict timings;
        timings["pre_detect"] = stats.pre_detect_ms;
        timings["full_refine"] = stats.full_refine_ms;
        timings["crop_plan"] = stats.crop_plan_ms;
        timings["crop_refine"] = stats.crop_refine_ms;
        timings["nms"] = stats.nms_ms;
        timings["total"] = stats.total_ms;
        py::dict metadata;
        metadata["pre_detections"] = stats.pre_detections;
        metadata["candidate_crops"] = stats.candidate_crops;
        metadata["crops_processed"] = stats.crops_processed;
        metadata["limited"] = stats.limited;
        metadata["merge_results"] = merge_results;
        metadata["timings_ms"] = timings;
        py::dict result;
        result["results"] = serialize(boxes, return_mask);
        result["refinement"] = metadata;
        return result;
    }

private:
    static py::list serialize(const object::DetectionBoxArray& boxes, bool return_mask)
    {

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

    using sam3::refine::CropConfig;
    py::class_<CropConfig>(m, "CropConfig")
        .def(py::init<>())
        .def_readwrite("max_size", &CropConfig::max_size)
        .def_readwrite("padding", &CropConfig::padding)
        .def_readwrite("w_diou", &CropConfig::w_diou)
        .def_readwrite("w_expansion", &CropConfig::w_expansion)
        .def_readwrite("count_penalty", &CropConfig::count_penalty)
        .def_readwrite("nms_threshold", &CropConfig::nms_threshold)
        .def_readwrite("enable_ar_fix", &CropConfig::enable_ar_fix)
        .def_readwrite("target_ar", &CropConfig::target_ar)
        .def_readwrite("max_crops", &CropConfig::max_crops)
        .def_readwrite("max_pre_detections", &CropConfig::max_pre_detections);

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
             "Detect objects in an image. Returns a list of detection dicts.")
        .def("detect_obj_refine", &Sam3PyModel::detect_obj_refine,
             py::arg("image_bytes"), py::arg("pre_detect_labels"), py::arg("refine_labels"),
             py::arg("confidence") = 0.5f, py::arg("return_mask") = false,
             py::arg("merge_results") = true, py::arg("crop_config") = CropConfig{},
             py::arg("pre_detect_confidence") = -1.0f,
             "Pre-detect, cluster crops, refine and run same-class NMS. Returns results and diagnostics.");
}
