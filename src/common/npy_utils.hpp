#pragma once

#ifndef NPY_UTILS_HPP__
#define NPY_UTILS_HPP__

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief 极简 numpy .npy 加载器，仅支持 float32 连续存储数组
 *
 * @param path  npy 文件路径
 * @param data  输出数据（host 端）
 * @param shape 输出 shape
 * @return true 成功
 */
bool load_npy_float32(const std::string& path, std::vector<float>& data, std::vector<size_t>& shape);

#endif // NPY_UTILS_HPP__
