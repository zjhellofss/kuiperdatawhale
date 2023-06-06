//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
static std::string ShapeStr(const std::vector<int> &shapes) {
  std::ostringstream ss;
  for (int i = 0; i < shapes.size(); ++i) {
    ss << shapes.at(i);
    if (i != shapes.size() - 1) {
      ss << " x ";
    }
  }
  return ss.str();
}
TEST(test_ir, pnnx_graph_ops) {
  using namespace kuiper_infer;
  /**
   * 如果这里加载失败，请首先考虑相对路径的正确性问题
   */
  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  // 如果这里加载失败，请首先考虑相对路径(bin_path和param_path)的正确性问题
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    LOG(INFO) << ops.at(i)->name;
  }
}

// 输出运算数
TEST(test_ir, pnnx_graph_operands) {
  using namespace kuiper_infer;
  /**
   * 如果这里加载失败，请首先考虑相对路径的正确性问题
   */
  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  // 如果这里加载失败，请首先考虑相对路径(bin_path和param_path)的正确性问题
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    LOG(INFO) << "OP Name: " << op->name;
    LOG(INFO) << "OP Inputs";
    for (int j = 0; j < op->inputs.size(); ++j) {
      LOG(INFO) << "Input name: " << op->inputs.at(j)->name
                << " shape: " << ShapeStr(op->inputs.at(j)->shape);
    }

    LOG(INFO) << "OP Output";
    for (int j = 0; j < op->outputs.size(); ++j) {
      LOG(INFO) << "Output name: " << op->outputs.at(j)->name
                << " shape: " << ShapeStr(op->outputs.at(j)->shape);
    }
    LOG(INFO) << "---------------------------------------------";
  }
}

// 输出运算数和参数
TEST(test_ir, pnnx_graph_operands_and_params) {
  using namespace kuiper_infer;
  /**
   * 如果这里加载失败，请首先考虑相对路径的正确性问题
   */
  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");
  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  // 如果这里加载失败，请首先考虑相对路径(bin_path和param_path)的正确性问题
  ASSERT_EQ(load_result, 0);
  const auto &ops = graph->ops;
  for (int i = 0; i < ops.size(); ++i) {
    const auto &op = ops.at(i);
    if (op->name != "linear") {
      continue;
    }
    LOG(INFO) << "OP Name: " << op->name;
    LOG(INFO) << "OP Inputs";
    for (int j = 0; j < op->inputs.size(); ++j) {
      LOG(INFO) << "Input name: " << op->inputs.at(j)->name
                << " shape: " << ShapeStr(op->inputs.at(j)->shape);
    }

    LOG(INFO) << "OP Output";
    for (int j = 0; j < op->outputs.size(); ++j) {
      LOG(INFO) << "Output name: " << op->outputs.at(j)->name
                << " shape: " << ShapeStr(op->outputs.at(j)->shape);
    }

    LOG(INFO) << "Params";
    for (const auto &attr : op->params) {
      LOG(INFO) << attr.first;
    }

    LOG(INFO) << "Weight: ";
    for (const auto &weight : op->attrs) {
      LOG(INFO) << weight.first << " : " << ShapeStr(weight.second.shape);
    }
    LOG(INFO) << "---------------------------------------------";
  }
}
