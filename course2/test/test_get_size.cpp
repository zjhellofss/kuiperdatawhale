//
// Created by fss on 23-6-4.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
TEST(test_tensor_size, tensor_size1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  LOG(INFO) << "-----------------------Tensor Get Size-----------------------";
  LOG(INFO) << "channels: " << f1.channels();
  LOG(INFO) << "rows: " << f1.rows();
  LOG(INFO) << "cols: " << f1.cols();
}