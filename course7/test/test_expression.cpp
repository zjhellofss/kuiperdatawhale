//
// Created by fss on 23-7-22.
//
#include "layer/abstract/layer_factory.hpp"
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/expression.hpp"

#include "parser/parse_expression.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace kuiper_infer;

TEST(test_parser, tokenizer) {
  using namespace kuiper_infer;
  const std::string &str = "add(@0,mul(@1,@2))";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto &tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.token_strs();
  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(2), "@0");
  ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(3), ",");
  ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(4), "mul");
  ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenMul);

  ASSERT_EQ(token_strs.at(5), "(");
  ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(6), "@1");
  ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(7), ",");
  ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(8), "@2");
  ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(9), ")");
  ASSERT_EQ(tokens.at(9).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(10), ")");
  ASSERT_EQ(tokens.at(10).token_type, TokenType::TokenRightBracket);
}

TEST(test_parser, generate1) {
  using namespace kuiper_infer;
  const std::string &str = "add(@0,@1)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  int index = 0; // 从0位置开始构建语法树
  // 抽象语法树:
  //
  //    add
  //    /  \
  //  @0    @1

  const auto &node = parser.Generate_(index);
  ASSERT_EQ(node->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(node->left->num_index, 0);
  ASSERT_EQ(node->right->num_index, 1);
}

TEST(test_parser, generate2) {
  using namespace kuiper_infer;
  const std::string &str = "add(mul(@0,@1),@2)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  int index = 0; // 从0位置开始构建语法树
  // 抽象语法树:
  //
  //       add
  //       /  \
  //     mul   @2
  //    /   \
  //  @0    @1

  const auto &node = parser.Generate_(index);
  ASSERT_EQ(node->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(node->left->num_index, int(TokenType::TokenMul));
  ASSERT_EQ(node->left->left->num_index, 0);
  ASSERT_EQ(node->left->right->num_index, 1);

  ASSERT_EQ(node->right->num_index, 2);
}

TEST(test_parser, reverse_polish) {
  using namespace kuiper_infer;
  const std::string &str = "add(mul(@0,@1),@2)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  // 抽象语法树:
  //
  //       add
  //       /  \
  //     mul   @2
  //    /   \
  //  @0    @1

  const auto &vec = parser.Generate();
  for (const auto &item : vec) {
    if (item->num_index == -5) {
      LOG(INFO) << "Mul";
    } else if (item->num_index == -6) {
      LOG(INFO) << "Add";
    } else {
      LOG(INFO) << item->num_index;
    }
  }
}

TEST(test_expression, complex1) {
  using namespace kuiper_infer;
  const std::string &str = "mul(@2,add(@0,@1))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f);

  std::shared_ptr<Tensor<float>> input3 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  input3->Fill(4.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  std::shared_ptr<Tensor<float>> output2 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(20.f);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-5));
}