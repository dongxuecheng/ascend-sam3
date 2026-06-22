#ifndef INFER_HPP__
#define INFER_HPP__

#include "common/object.hpp"
#include "infer/sam3type.hpp"
#include <memory>
#include <string>

struct ModelPaths
{
    std::string vision_model;   // vision encoder .om
    std::string text_model;     // text encoder .om
    std::string decoder_model;  // decoder .om
    std::string fpn_pos2;       // fpn_pos_2_constant.npy
};

class Infer
{
  public:
    virtual object::DetectionBoxArray forward(std::shared_ptr<Sam3Input> input) = 0;
    virtual ~Infer() = default;
};

std::shared_ptr<Infer> load(const ModelPaths& paths);

#endif
