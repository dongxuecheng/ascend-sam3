#include "common/npy_utils.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static bool parse_shape(const std::string& header, std::vector<size_t>& shape)
{
    size_t pos = header.find("'shape':");
    if (pos == std::string::npos)
    {
        std::cerr << "Cannot find shape in npy header" << std::endl;
        return false;
    }

    size_t start = header.find('(', pos);
    size_t end   = header.find(')', start);
    if (start == std::string::npos || end == std::string::npos)
    {
        std::cerr << "Invalid shape format in npy header" << std::endl;
        return false;
    }

    std::string inner = header.substr(start + 1, end - start - 1);
    shape.clear();
    std::stringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        // 去除空白
        size_t b = token.find_first_not_of(" \t");
        size_t e = token.find_last_not_of(" \t");
        if (b == std::string::npos)
        {
            continue;
        }
        std::string num = token.substr(b, e - b + 1);
        if (num.empty())
        {
            continue;
        }
        shape.push_back(static_cast<size_t>(std::stoull(num)));
    }
    return true;
}

static bool parse_descr(const std::string& header)
{
    size_t pos = header.find("'descr':");
    if (pos == std::string::npos)
    {
        std::cerr << "Cannot find descr in npy header" << std::endl;
        return false;
    }

    size_t quote1 = header.find('\'', pos + 8);
    size_t quote2 = header.find('\'', quote1 + 1);
    if (quote1 == std::string::npos || quote2 == std::string::npos)
    {
        std::cerr << "Invalid descr format" << std::endl;
        return false;
    }

    std::string descr = header.substr(quote1 + 1, quote2 - quote1 - 1);
    // 仅支持小端 float32: '<f4'
    if (descr != "<f4" && descr != "|f4")
    {
        std::cerr << "Unsupported npy dtype: " << descr << ", only float32 is supported" << std::endl;
        return false;
    }
    return true;
}

bool load_npy_float32(const std::string& path, std::vector<float>& data, std::vector<size_t>& shape)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Cannot open npy file: " << path << std::endl;
        return false;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < 10)
    {
        std::cerr << "Invalid npy file: too small" << std::endl;
        return false;
    }

    char magic[6] = {0};
    file.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        std::cerr << "Invalid npy magic" << std::endl;
        return false;
    }

    uint8_t major = 0, minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    size_t header_len = 0;
    if (major == 1 && minor == 0)
    {
        uint16_t len = 0;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        header_len = len;
    }
    else
    {
        uint32_t len = 0;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        header_len = len;
    }

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    if (!parse_descr(header) || !parse_shape(header, shape))
    {
        return false;
    }

    size_t expect_count = 1;
    for (auto s : shape)
    {
        expect_count *= s;
    }
    size_t expect_bytes = expect_count * sizeof(float);

    std::streamsize data_offset = file.tellg();
    std::streamsize remain      = file_size - data_offset;
    if (remain < static_cast<std::streamsize>(expect_bytes))
    {
        std::cerr << "Npy data size mismatch, expect " << expect_bytes << ", got " << remain << std::endl;
        return false;
    }

    data.resize(expect_count);
    file.read(reinterpret_cast<char*>(data.data()), expect_bytes);
    return file.good();
}
