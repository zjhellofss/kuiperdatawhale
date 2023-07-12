//
// Created by fss on 23-7-12.
//
#include <gtest/gtest.h>
#include "layer/abstract/layer_factory.hpp"
using namespace kuiper_infer;

TEST(test_registry, create_layer_find) {
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.Sigmoid";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  // 评价是否注册成功
  ASSERT_NE(layer, nullptr);
}

TEST(test_registry, create_layer_sigmoid_forward) {
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "nn.Sigmoid";
  std::shared_ptr<Layer> layer;
  ASSERT_EQ(layer, nullptr);
  layer = LayerRegisterer::CreateLayer(op);
  ASSERT_NE(layer, nullptr);

  sftensor input_tensor = std::make_shared<ftensor>(3, 4, 4);
  input_tensor->Rand();

  std::vector<sftensor> inputs(1);
  std::vector<sftensor> outputs(1);
  inputs.at(0) = input_tensor;
  layer->Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  sftensor output_tensor = outputs.front();
  ASSERT_EQ(output_tensor->empty(), false);
  ASSERT_EQ(output_tensor->size(), input_tensor->size());

  uint32_t size = output_tensor->size();
  // 评价sigmoid的计算结果是否正确
  for (uint32_t i = 0; i < size; ++i) {
    float input_value = input_tensor->index(i);
    float output_value = output_tensor->index(i);
    ASSERT_EQ(output_value, 1 / (1.f + expf(-input_value)));
  }
}

