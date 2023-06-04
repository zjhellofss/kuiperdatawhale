//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, tensor_init1D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4);
  f1.Fill(1.f);
  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);
  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(test_tensor, tensor_init2D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_3) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t channels = raw_shapes.at(0);
  const uint32_t rows = raw_shapes.at(1);
  const uint32_t cols = raw_shapes.at(2);

  LOG(INFO) << "data channels: " << channels;
  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_2) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 2, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_1) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 1, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);

  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}
