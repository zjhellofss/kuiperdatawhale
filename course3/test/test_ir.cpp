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
      LOG(INFO) << attr.first << " type " << attr.second.type;
    }

    LOG(INFO) << "Weight: ";
    for (const auto &weight : op->attrs) {
      LOG(INFO) << weight.first << " : " << ShapeStr(weight.second.shape)
                << " type " << weight.second.type;
    }
    LOG(INFO) << "---------------------------------------------";
  }
}

TEST(test_ir, pnnx_graph_operands_customer_producer) {
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
  const auto &operands = graph->operands;
  for (int i = 0; i < operands.size(); ++i) {
    const auto &operand = operands.at(i);
    LOG(INFO) << "Operand Name: #" << operand->name;
    LOG(INFO) << "Customers: ";
    for (const auto &customer : operand->consumers) {
      LOG(INFO) << customer->name;
    }

    LOG(INFO) << "Producer: " << operand->producer->name;
  }
}

TEST(test_ir, pnnx_graph_all) {
  using namespace kuiper_infer;
  /**
   * 如果这里加载失败，请首先考虑相对路径的正确性问题
   */
  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "op name: " << operator_->name << " type: " << operator_->type;
    LOG(INFO) << "attribute:";
    for (const auto &[name, attribute_] : operator_->attribute) {
      LOG(INFO) << name << " type: " << int(attribute_->type)
                << " shape: " << ShapeStr(attribute_->shape);
      const auto &weight_data = attribute_->weight_data;
      ASSERT_EQ(weight_data.empty(), false); // 判断权重是否为空
    }
    LOG(INFO) << "inputs: ";
    for (const auto &input : operator_->input_operands) {
      LOG(INFO) << "name: " << input.first
                << " shape: " << ShapeStr(input.second->shapes);
    }

    LOG(INFO) << "outputs: ";
    for (const auto &output : operator_->output_names) {
      LOG(INFO) << "name: " << output;
    }
    LOG(INFO) << "--------------------------------------";
  }
}

TEST(test_ir, pnnx_graph_all_homework) {
  using namespace kuiper_infer;
  /**
   * 如果这里加载失败，请首先考虑相对路径的正确性问题
   */
  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    if (operator_->name == "linear") {
      const auto &params = operator_->params;
      ASSERT_EQ(params.size(), 3);
        /////////////////////////////////
      ASSERT_EQ(params.count("bias"), 1);
      RuntimeParameter *parameter_bool = params.at("bias");
      ASSERT_NE(parameter_bool, nullptr);
      ASSERT_EQ((dynamic_cast<RuntimeParameterBool *>(parameter_bool)->value),
                true);
      /////////////////////////////////
      ASSERT_EQ(params.count("in_features"), 1);
      RuntimeParameter *parameter_in_features = params.at("in_features");
      ASSERT_NE(parameter_in_features, nullptr);
      ASSERT_EQ(
          (dynamic_cast<RuntimeParameterInt *>(parameter_in_features)->value),
          32);

      /////////////////////////////////
      ASSERT_EQ(params.count("out_features"), 1);
      RuntimeParameter *parameter_out_features = params.at("out_features");
      ASSERT_NE(parameter_out_features, nullptr);
      ASSERT_EQ(
          (dynamic_cast<RuntimeParameterInt *>(parameter_out_features)->value),
          128);
    }
  }
}