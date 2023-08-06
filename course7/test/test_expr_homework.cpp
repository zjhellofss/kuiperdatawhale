//
// Created by fss on 23-8-5.
//
#include <gtest/gtest.h>
#include <vector>
#include "../source/layer/details/expression.hpp"

using namespace kuiper_infer;

#include "parser/parse_expression.hpp"
TEST(test_parser, tokenizer_sin) {
  using namespace kuiper_infer;
  const std::string &str = "add(sin(@0),@1)";
  ExpressionParser parser(str);
  parser.Tokenizer();
  const auto &tokens = parser.tokens();
  ASSERT_EQ(tokens.empty(), false);

  const auto &token_strs = parser.token_strs();
  ASSERT_EQ(token_strs.at(0), "add");
  ASSERT_EQ(tokens.at(0).token_type, TokenType::TokenAdd);

  ASSERT_EQ(token_strs.at(1), "(");
  ASSERT_EQ(tokens.at(1).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(2), "sin");
  ASSERT_EQ(tokens.at(2).token_type, TokenType::TokenSin);

  ASSERT_EQ(token_strs.at(3), "(");
  ASSERT_EQ(tokens.at(3).token_type, TokenType::TokenLeftBracket);

  ASSERT_EQ(token_strs.at(4), "@0");
  ASSERT_EQ(tokens.at(4).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(5), ")");
  ASSERT_EQ(tokens.at(5).token_type, TokenType::TokenRightBracket);

  ASSERT_EQ(token_strs.at(6), ",");
  ASSERT_EQ(tokens.at(6).token_type, TokenType::TokenComma);

  ASSERT_EQ(token_strs.at(7), "@1");
  ASSERT_EQ(tokens.at(7).token_type, TokenType::TokenInputNumber);

  ASSERT_EQ(token_strs.at(8), ")");
  ASSERT_EQ(tokens.at(8).token_type, TokenType::TokenRightBracket);
}

TEST(test_parser, generate_sin) {
  using namespace kuiper_infer;
  const std::string &str = "add(sin(@0),@1)";

  int index = 0;
  /**
        add
        /   \
      sin    @1
       |
      @0
   */
  ExpressionParser parser(str);
  const auto &node = parser.Generate_(index);
  ASSERT_EQ(node->num_index, int(TokenType::TokenAdd));
  ASSERT_EQ(node->left->num_index, int(TokenType::TokenSin));
  ASSERT_EQ(node->left->left->num_index, 0);
  ASSERT_EQ(node->right->num_index, 1);
}

TEST(test_expression, complex2) {
  using namespace kuiper_infer;
  const std::string &str = "mul(@1,sin(@0))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(2.f); // @0
  std::shared_ptr<Tensor<float>> input2 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  input2->Fill(3.f); //@1

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  outputs.at(0) = std::make_shared<Tensor<float>>(3, 224, 224);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);

  float val = 2.f;
  float res = std::sin(val) * 3.f;
  std::shared_ptr<Tensor<float>> output2 =
      std::make_shared<Tensor<float>>(3, 224, 224);
  output2->Fill(res);
  std::shared_ptr<Tensor<float>> output1 = outputs.front();

  ASSERT_TRUE(
      arma::approx_equal(output1->data(), output2->data(), "absdiff", 1e-3));
}