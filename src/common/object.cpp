#include "common/object.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace object
{
    // Box 的构造函数实现
    Box::Box(float l, float t, float r, float b)
        : left(l), top(t), right(r), bottom(b) {}

    // Box 的输出流操作符重载，用于打印
    std::ostream &operator<<(std::ostream &os, const Box &box) {
        os << "{ \"left\": " << box.left
           << ", \"top\": " << box.top
           << ", \"right\": " << box.right
           << ", \"bottom\": " << box.bottom
           << " }";
        return os;
    }

    Segmentation& Segmentation::operator=(const Segmentation& other)
    {
        if (this == &other) {
            return *this;
        }
        this->mask = other.mask.clone();
        return *this;
    }

    void Segmentation::keep_largest_part()
    {
        if (mask.empty()) return;

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) return;

        // 找到最大轮廓
        auto max_contour = std::max_element(contours.begin(), contours.end(),
                                             [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
                                                 return cv::contourArea(a) < cv::contourArea(b);
                                             });

        // 创建一个新的掩码，只保留最大轮廓
        cv::Mat new_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(new_mask, std::vector<std::vector<cv::Point>>{*max_contour}, -1, cv::Scalar(255), cv::FILLED);
        mask = new_mask;
    }

    Segmentation Segmentation::align_to_left_top(int left, int top, int width, int height) const
    {
        object::Segmentation aligned_seg;
        // 原本的mask是相对于left,top的
        // 现在我们需要创建一个新的mask，大小为width x height，并将原mask放置在新的mask中
        cv::Mat aligned_mask = cv::Mat::zeros(height, width, mask.type());
        if (mask.empty()) return aligned_seg;
        // 计算放置位置
        int x_offset = std::max(0, left);
        int y_offset = std::max(0, top);
        // 计算原mask在新mask中的有效区域
        int copy_width = std::min(mask.cols, width - x_offset);
        int copy_height = std::min(mask.rows, height - y_offset);
        if (copy_width > 0 && copy_height > 0) {
            cv::Rect src_roi(0, 0, copy_width, copy_height);
            cv::Rect dst_roi(x_offset, y_offset, copy_width, copy_height);
            mask(src_roi).copyTo(aligned_mask(dst_roi));
        }
        aligned_seg.mask = aligned_mask;
        return aligned_seg;
    }

    //================================================================================
    // 已修改: DetectionBox 的输出流操作符重载
    // 这是最重要的修改。它不再使用僵化的 switch 结构，
    // 而是先打印所有必需字段，然后逐一检查每个 optional 成员是否存在值。
    // 这种方式更灵活，能准确地反映出任意组合的数据。
    //================================================================================
    std::ostream &operator<<(std::ostream &os, const DetectionBox &box) {
        os << "{";
        
        // --- 打印核心/必需字段 ---
        os << " \"type\": \"" << ObjectTypeToString(box.type) << "\""
           << ", \"class_id\": " << box.class_id
           << ", \"class_name\": \"" << box.class_name << "\""
           << ", \"score\": " << box.score
           << ", \"box\": " << box.box;
        if (box.segmentation.has_value()) {
            // 直接打印整个掩码矩阵不现实，所以我们只打印它的尺寸信息
            const auto& mask = box.segmentation.value().mask;
            os << ", \"segmentation\": { \"width\": " << mask.cols 
               << ", \"height\": " << mask.rows << " }";
        }
        os << " }";
        return os;
    }

    cv::Mat segmentMapToMat(const std::shared_ptr<SegmentMap>& map)
    {
        // 检查输入数据是否有效
        if (map->data == nullptr || map->width <= 0 || map->height <= 0) 
        {
            // 如果SegmentMap无效，返回一个空的cv::Mat
            return cv::Mat();
        }

        // 使用SegmentMap的数据指针、高度和宽度创建一个cv::Mat对象头。
        // CV_8UC1 表示这是一个8位无符号单通道图像，这通常是分割掩码的格式。
        // cv::Mat的这个构造函数不会复制数据，而是直接使用提供的指针。
        return cv::Mat(map->height, map->width, CV_8UC1, map->data);
    }

} // namespace object