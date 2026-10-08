// Vision-only benchmark: no tokenizer, Text, Decoder, HTTP or mask processing.
#include "infer/modelVision.hpp"
#include "common/visionBenchMetrics.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using sam3::bench::Difference;
using sam3::bench::Summary;

namespace
{
constexpr std::array<std::size_t, 3> feature_elements{256ULL*288*288, 256ULL*144*144, 256ULL*72*72};

void check(aclError error, const char* action)
{
    if (error != ACL_SUCCESS)
        throw std::runtime_error(std::string(action) + " failed: " + std::to_string(error));
}

std::string quoted(const std::string& text)
{
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text)
    {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

struct Options
{
    std::string model, manifest, output, features, reference;
    int device = 0, warmup = 3, iterations = 10;
    double atol = 0.001, rtol = 0.001;
};

Options options(int argc, char** argv)
{
    const std::string usage = "ascendsam3_vision_bench --model FILE --manifest FILE --json-output FILE "
        "[--device 0 --warmup 3 --iterations 10 --features-dir DIR --reference-dir DIR --atol 0.001 --rtol 0.001]";
    if (argc == 2 && std::string(argv[1]) == "--help") { std::cout << usage << '\n'; std::exit(0); }
    std::map<std::string, std::string> args;
    const std::vector<std::string> known{"--model", "--manifest", "--json-output", "--device", "--warmup",
        "--iterations", "--features-dir", "--reference-dir", "--atol", "--rtol"};
    for (int i = 1; i < argc; i += 2)
    {
        const std::string key = argv[i];
        if (i + 1 >= argc || std::find(known.begin(), known.end(), key) == known.end() || args.count(key))
            throw std::invalid_argument(usage);
        args[key] = argv[i + 1];
    }
    Options opt;
    opt.model = args["--model"]; opt.manifest = args["--manifest"]; opt.output = args["--json-output"];
    opt.features = args["--features-dir"]; opt.reference = args["--reference-dir"];
    auto number = [&](const std::string& key, double fallback) {
        if (!args.count(key)) return fallback;
        std::size_t consumed = 0;
        const double value = std::stod(args[key], &consumed);
        if (consumed != args[key].size() || !std::isfinite(value)) throw std::invalid_argument("Invalid " + key);
        return value;
    };
    auto integer = [&](const std::string& key, int fallback, int minimum, int maximum) {
        const double value = number(key, fallback);
        if (value < minimum || value > maximum || std::floor(value) != value) throw std::invalid_argument("Invalid " + key);
        return static_cast<int>(value);
    };
    opt.device = integer("--device", 0, 0, 1024);
    opt.warmup = integer("--warmup", 3, 0, 100000);
    opt.iterations = integer("--iterations", 10, 1, 100000);
    opt.atol = number("--atol", opt.atol); opt.rtol = number("--rtol", opt.rtol);
    if (opt.atol < 0 || opt.rtol < 0 || opt.model.empty() || opt.manifest.empty() || opt.output.empty())
        throw std::invalid_argument(usage);
    if (!opt.features.empty() && !opt.reference.empty()) throw std::invalid_argument("Dump or compare, not both");
    return opt;
}

struct Case
{
    std::string image;
    bool cropped = false;
    cv::Rect roi;
};

std::vector<Case> read_manifest(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot read manifest: " + path);
    std::vector<Case> cases;
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto split = line.find('\t');
        Case item;
        item.image = line.substr(0, split);
        if (split != std::string::npos)
        {
            item.cropped = true;
            std::istringstream fields(line.substr(split + 1));
            std::string extra;
            if (!(fields >> item.roi.x >> item.roi.y >> item.roi.width >> item.roi.height) || (fields >> extra))
                throw std::runtime_error("Invalid manifest ROI: " + line);
            if (item.roi.x < 0 || item.roi.y < 0 || item.roi.width <= 0 || item.roi.height <= 0)
                throw std::runtime_error("Invalid manifest ROI: " + line);
        }
        cases.push_back(item);
    }
    if (cases.empty()) throw std::runtime_error("Empty manifest");
    return cases;
}

void emit_summary(std::ostream& out, const std::vector<double>& values)
{
    const Summary s = sam3::bench::summarize(values);
    out << "{\"mean\":" << s.mean << ",\"p50\":" << s.p50 << ",\"p95\":" << s.p95
        << ",\"min\":" << s.minimum << ",\"max\":" << s.maximum << '}';
}

void emit_timings(std::ostream& out, const std::vector<VisionModel::Timings>& values)
{
    out << '{';
    const std::array<const char*, 4> names{"sampling_ms", "upload_ms", "inference_ms", "total_ms"};
    const std::array<double VisionModel::Timings::*, 4> members{&VisionModel::Timings::sampling_ms,
        &VisionModel::Timings::upload_ms, &VisionModel::Timings::inference_ms, &VisionModel::Timings::total_ms};
    for (std::size_t i = 0; i < names.size(); ++i)
    {
        if (i) out << ',';
        out << quoted(names[i]) << ':';
        std::vector<double> samples;
        for (const auto& value : values) samples.push_back(value.*members[i]);
        emit_summary(out, samples);
    }
    out << '}';
}

Difference feature(VisionModel& model, std::size_t index, std::size_t case_index, const Options& opt)
{
    const auto bytes = feature_elements[index] * sizeof(float);
    const fs::path name = std::to_string(case_index) + "-fpn" + std::to_string(index) + ".f32";
    std::ifstream reference;
    std::ofstream dump;
    if (!opt.reference.empty())
    {
        const auto path = fs::path(opt.reference) / name;
        if (!fs::is_regular_file(path) || fs::file_size(path) != bytes)
            throw std::runtime_error("Missing/wrong-size reference: " + path.string());
        reference.open(path, std::ios::binary);
        if (!reference) throw std::runtime_error("Cannot open reference");
    }
    if (!opt.features.empty())
    {
        const auto path = fs::path(opt.features) / name;
        if (fs::exists(path)) throw std::runtime_error("Refusing to overwrite feature: " + path.string());
        dump.open(path, std::ios::binary);
        if (!dump) throw std::runtime_error("Cannot create feature dump");
    }
    Difference diff;
    // Chunked reads bound host memory and only inspect the logical tensors
    // consumed by Decoder, not unused capacity/padding in OM output buffers.
    std::vector<float> actual(65536), expected(65536);
    for (std::size_t offset = 0; offset < feature_elements[index]; offset += actual.size())
    {
        const auto count = std::min(actual.size(), feature_elements[index] - offset);
        const auto size = count * sizeof(float);
        check(aclrtMemcpy(actual.data(), size, static_cast<char*>(model.feature_ptr(index)) + offset*sizeof(float),
                          size, ACL_MEMCPY_DEVICE_TO_HOST), "Read Vision features");
        if (reference.is_open())
        {
            reference.read(reinterpret_cast<char*>(expected.data()), size);
            if (!reference) throw std::runtime_error("Cannot read feature reference");
        }
        if (dump.is_open())
        {
            dump.write(reinterpret_cast<const char*>(actual.data()), size);
            if (!dump) throw std::runtime_error("Cannot write feature dump (disk full?)");
        }
        for (std::size_t j = 0; j < count; ++j)
            diff.add(actual[j], reference.is_open() ? expected[j] : actual[j], opt.atol, opt.rtol);
    }
    if (dump.is_open()) { dump.flush(); if (!dump) throw std::runtime_error("Cannot flush feature dump"); }
    return diff;
}

int run(const Options& opt)
{
    const auto cases = read_manifest(opt.manifest);
    if (fs::exists(opt.output)) throw std::runtime_error("Refusing to overwrite report: " + opt.output);
    if (!opt.features.empty()) fs::create_directories(opt.features);
    VisionModel model;
    check(model.init(opt.model), "Load Vision OM");
    model.set_timing_log(false);
    const char* soc = aclrtGetSocName();
    if (!soc) throw std::runtime_error("Cannot query actual chip type");
    if (model.input_count() != 1 || model.input_size(0) != 1008ULL*1008*3 ||
        aclmdlGetInputDataType(model.desc(), 0) != ACL_UINT8 ||
        (model.output_count() != 3 && model.output_count() != 4))
        throw std::runtime_error("Expected 1008x1008 static AIPP uint8 Vision OM with 3/4 outputs");
    for (std::size_t i = 0; i < 3; ++i)
    {
        const auto format = aclmdlGetOutputFormat(model.desc(), i);
        const char* name = aclmdlGetOutputNameByIndex(model.desc(), i);
        if (aclmdlGetOutputDataType(model.desc(), i) != ACL_FLOAT ||
            (format != ACL_FORMAT_NCHW && format != ACL_FORMAT_ND) ||
            model.output_size(i) < feature_elements[i]*sizeof(float) || !name ||
            std::string(name).find("fpn_feat_" + std::to_string(i)) == std::string::npos)
            throw std::runtime_error("Incompatible Vision output " + std::to_string(i));
    }
    std::ostringstream report;
    report << std::setprecision(12) << "{\"schema_version\":1,\"model\":" << quoted(opt.model)
           << ",\"device\":" << opt.device << ",\"soc\":" << quoted(soc)
           << ",\"warmup\":" << opt.warmup << ",\"iterations\":" << opt.iterations
           << ",\"atol\":" << opt.atol << ",\"rtol\":" << opt.rtol
           << ",\"comparison_enabled\":" << (opt.reference.empty() ? "false" : "true") << ",\"cases\":[";
    std::vector<VisionModel::Timings> all;
    bool passed = true;
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        const auto& item = cases[i];
        const cv::Mat image = cv::imread(item.image, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Cannot decode image: " + item.image);
        auto encode = [&]() {
            check(item.cropped ? model.encode_crop(image, item.roi) : model.encode(image), "Vision encode");
        };
        for (int j = 0; j < opt.warmup; ++j) encode();
        std::vector<VisionModel::Timings> timings;
        for (int j = 0; j < opt.iterations; ++j)
        {
            encode();
            timings.push_back(model.last_timings());
            all.push_back(model.last_timings());
        }
        if (i) report << ',';
        report << "{\"image\":" << quoted(item.image) << ",\"roi\":";
        if (item.cropped) report << '[' << item.roi.x << ',' << item.roi.y << ',' << item.roi.width << ',' << item.roi.height << ']';
        else report << "null";
        report << ",\"timings\":";
        emit_timings(report, timings);
        report << ",\"samples\":[";
        for (std::size_t j = 0; j < timings.size(); ++j)
        {
            if (j) report << ',';
            const auto& t = timings[j];
            report << '[' << t.sampling_ms << ',' << t.upload_ms << ',' << t.inference_ms << ',' << t.total_ms << ']';
        }
        report << "],\"features\":[";
        for (std::size_t k = 0; k < 3; ++k)
        {
            if (k) report << ',';
            const Difference diff = feature(model, k, i, opt);
            passed = passed && diff.passed();
            report << "{\"index\":" << k << ",\"elements\":" << diff.elements
                   << ",\"max_abs\":" << diff.max_abs << ",\"mae\":" << diff.mae()
                   << ",\"rmse\":" << diff.rmse() << ",\"cosine\":" << diff.cosine()
                   << ",\"mismatches\":" << diff.mismatches << ",\"nonfinite\":" << diff.nonfinite
                   << ",\"passed\":" << (diff.passed() ? "true" : "false") << '}';
        }
        report << "]}";
        std::cout << "Vision case " << i+1 << '/' << cases.size() << " complete\n" << std::flush;
    }
    report << "],\"timings\":";
    emit_timings(report, all);
    report << ",\"sample_count\":" << all.size() << ",\"features_passed\":" << (passed ? "true" : "false") << "}\n";
    std::ofstream file(opt.output);
    if (!file) throw std::runtime_error("Cannot create report");
    file << report.str(); file.flush();
    if (!file) throw std::runtime_error("Cannot write report");
    std::cout << "Vision JSON: " << opt.output << '\n';
    return passed ? 0 : 2;
}
} // namespace

int main(int argc, char** argv)
{
    bool initialized = false, bound = false;
    int device = 0, status = 1;
    try
    {
        const auto opt = options(argc, argv);
        device = opt.device;
        check(aclInit(nullptr), "aclInit"); initialized = true;
        std::uint32_t count = 0;
        check(aclrtGetDeviceCount(&count), "aclrtGetDeviceCount");
        if (device >= static_cast<int>(count)) throw std::runtime_error("Device outside visible range");
        check(aclrtSetDevice(device), "aclrtSetDevice"); bound = true;
        status = run(opt); // Model destruction occurs before reset/finalize.
    }
    catch (const std::exception& error) { std::cerr << "Vision benchmark failed: " << error.what() << '\n'; }
    if (bound && aclrtResetDevice(device) != ACL_SUCCESS) status = 1;
    if (initialized && aclFinalize() != ACL_SUCCESS) status = 1;
    return status;
}
