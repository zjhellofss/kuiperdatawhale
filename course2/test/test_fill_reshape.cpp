//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(test_fill_reshape, fill1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  std::vector<float> values(2 * 3 * 4);
  // 将1到12填充到values中
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.Fill(values);
  f1.Show();
}

TEST(test_fill_reshape, reshape1) {
  using namespace kuiper_infer;
  LOG(INFO) << "-------------------Reshape-------------------";
  Tensor<float> f1(2, 3, 4);
  std::vector<float> values(2 * 3 * 4);
  // 将1到12填充到values中
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.Fill(values);
  f1.Show();
  /// 将大小调整为(4, 3, 2)
  f1.Reshape({4, 3, 2}, true);
  LOG(INFO) << "-------------------After Reshape-------------------";
  f1.Show();
}