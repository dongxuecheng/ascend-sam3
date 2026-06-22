#ifndef OBJECT_HPP
#define OBJECT_HPP

#include <string>
#include <vector>
#include <tuple>
#include <ostream>
#include <optional>
#include "opencv2/opencv.hpp"

namespace object
{
    enum class ObjectType
    {
        UNKNOW = 0,
        OBJECT = 1,
        PERSON = 2,
        CAR    = 3,
    };

    inline const char* ObjectTypeToString(ObjectType type)
    {
        switch (type)
        {
            case ObjectType::OBJECT:
                return "object";
            case ObjectType::PERSON:
                return "person";
            case ObjectType::CAR:
                return "car";
            default:
                return "unknown";
        }
    }

    struct Box
    {
        float left = 0.0f;
        float top = 0.0f;
        float right = 0.0f;
        float bottom = 0.0f;

        Box() = default;
        Box(float l, float t, float r, float b);

        float width() const noexcept { return right - left; }
        float height() const noexcept { return bottom - top; }
        float center_x() const noexcept { return (left + right) / 2; }
        float center_y() const noexcept { return (top + bottom) / 2; }
        float area() const noexcept { return width() * height(); }

        friend std::ostream &operator<<(std::ostream &os, const Box &box);
    };

    struct SegmentMap
    {
        int width = 0, height = 0;     // width % 8 == 0
        unsigned char *data = nullptr; // is width * height memory

        SegmentMap(int width, int height);
        
        virtual ~SegmentMap();

        // 1. Delete Copy Constructor
        SegmentMap(const SegmentMap &) = delete;

        // 2. Delete Copy Assignment Operator
        SegmentMap &operator=(const SegmentMap &) = delete;

        // 3. Move Constructor
        SegmentMap(SegmentMap &&other) noexcept
            : width(std::exchange(other.width, 0)),    // Transfer ownership and reset source
            height(std::exchange(other.height, 0)),  // Transfer ownership and reset source
            data(std::exchange(other.data, nullptr)) // Transfer ownership and reset source
        {
            // The moved-from object 'other' is now in a valid, empty state
        }

        // 4. Move Assignment Operator
        SegmentMap &operator=(SegmentMap &&other) noexcept;
    };

    struct Segmentation
    {
        cv::Mat mask;
        // 分割区域可能有多个部分，使用findContours函数将最大的部分保留为mask
        void keep_largest_part();
        Segmentation align_to_left_top(int left, int top, int width, int height) const;
        Segmentation& operator=(const Segmentation& other);

    };

    struct DetectionBox
    {
        ObjectType type = ObjectType::UNKNOW;
        Box box;
        float score = 0.0f;
        int class_id = -1;
        std::string class_name;
        std::optional<Segmentation> segmentation;

        // 友元函数声明
        friend std::ostream &operator<<(std::ostream &os, const DetectionBox &box);
    };

    cv::Mat segmentMapToMat(const std::shared_ptr<SegmentMap>& map);

    using DetectionBoxArray = std::vector<DetectionBox>;

} // namespace object

#endif // OBJECT_HPP