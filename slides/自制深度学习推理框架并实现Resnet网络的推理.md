# 第八课-自制深度学习推理框架并实现Resnet网络的推理

> 本课程赞助方：`Datawhale`
>
> 作者：傅莘莘
>
> 主项目：https://github.com/zjhellofss/KuiperInfer 欢迎大家点赞和PR.
>
> 课程代码：https://github.com/zjhellofss/kuiperdatawhale/course8

![logo.png](https://i.imgur.com/dXqvgNI.png)

![](https://i.imgur.com/qO2yKaH.jpg)

## 模型执行函数

我们已经在之前的课时中讲解了所有算子的执行顺序排序，因为我们在对模型的所有算子进行拓扑排序后，就可以得到一个算子序列 `topo_operators_`， 所以在执行函数`Forward`时，我们只需**按顺序依次执行**算子序列（`topo_operators`）中的每一个算子即可。

```cpp
std::vector<std::shared_ptr<Tensor<float>>> RuntimeGraph::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug) {  
    ...
    ...
    for (const auto& current_op : topo_operators_) {
        if (current_op->type == "pnnx.Input") {
            current_op->has_forward = true;
            ProbeNextLayer(current_op, inputs);
        } else if (current_op->type == "pnnx.Output") {
            current_op->has_forward = true;
            CHECK(current_op->input_operands_seq.size() == 1);
            current_op->output_operands = current_op->input_operands_seq.front();
        } else {
            InferStatus status = current_op->layer->Forward();
            CHECK(status == InferStatus::kInferSuccess)
                << current_op->layer->layer_name()
                << " layer forward failed, error code: " << int(status);
            current_op->has_forward = true;
            ProbeNextLayer(current_op, current_op->output_operands->datas);
        }
    }
```

在算子执行的阶段，我们依次取出`topo_operators`数组中的所有算子。我们可以将算子分为三类：输入类型、输出类型和普通算子。普通算子包括卷积、池化、线性激活等。

### 执行输入类型算子

首先，让我们来分析执行输入类型算子的这个分支

```cpp
current_op->has_forward = true;
ProbeNextLayer(current_op, inputs);
```

在`ProbeNextLayer`函数中，我们对当前节点的所有后继节点进行依次的遍历，并将**当前节点的==输出==**赋值给**后继节点的==输入==**。

首先，我们获取当前节点的所有后继节点`next_ops`；然后对这些后继节点进行遍历操作，在每次遍历时，我们获取当前节点的`current_op`的后继节点`next_rt_operator`的输入张量`next_input_operands`.

```cpp
void RuntimeGraph::ProbeNextLayer(
    const std::shared_ptr<RuntimeOperator> &current_op,
    const std::vector<std::shared_ptr<Tensor<float>>> &layer_output_datas) {
    const auto &next_ops = current_op->output_operators;
    for (const auto &[_, next_rt_operator] : next_ops) {
        // 得到后继节点的输入next_input_operands
        const auto &next_input_operands = next_rt_operator->input_operands;
        ...
```

随后，我们再找到后继节点`next_rt_operator`中**关于`current_op`节点的输入空间`next_input_datas`。**这是因为后继节点`next_rt_operator`的输入可能有多个，我们需要将**当前节点的输出填入到后继节点对应的输入空间中。**

```cpp
void RuntimeGraph::ProbeNextLayer(
    const std::shared_ptr<RuntimeOperator> &current_op,
    const std::vector<std::shared_ptr<Tensor<float>>> &layer_output_datas){
    ...
    ...
    if (next_input_operands.find(current_op->name) !=
        next_input_operands.end()) {

        std::vector<std::shared_ptr<ftensor>> &next_input_datas =
            next_input_operands.at(current_op->name)->datas;
        CHECK(next_input_datas.size() == layer_output_datas.size());
        // 将当前current_op的输出赋值到next_input_datas中
        for (int i = 0; i < next_input_datas.size(); ++i) {
            next_input_datas.at(i) = layer_output_datas.at(i);
        }
    }
```

如下示意图所示，我们需要将当前节点（`current_op`）的输出填入到后继节点`next_rt_operator`关于`current_op.name`对应的位置中。

当我们遍历到`next_rt_operator`的其他前驱节点时，我们同样需要把该前驱节点（`other_op`）的输出张量放入到`next_rt_operator`对应的输入张量空间中。

```
next_input_operands:
{
    输入1 -- current_op.name: current_op对应的输出空间
    输入2 -- other_op.name: other_op对应的输出空间
}
```

在找到后继节点之后，我们将当前节点的输出数据赋值给下一个节点的输入数据。

```cpp
void RuntimeGraph::ProbeNextLayer(
    const std::shared_ptr<RuntimeOperator> &current_op,
    const std::vector<std::shared_ptr<Tensor<float>>> &layer_output_datas){
	...
    ...
    std::vector<std::shared_ptr<ftensor>> &next_input_datas =
        next_input_operands.at(current_op->name)->datas;
    CHECK(next_input_datas.size() == layer_output_datas.size());
    // 将当前current_op的输出赋值到next_input_datas中
    for (int i = 0; i < next_input_datas.size(); ++i) {
        next_input_datas.at(i) = layer_output_datas.at(i);
    }
```

### 执行普通类型算子

```cpp
else {
    InferStatus status = current_op->layer->Forward();
    CHECK(status == InferStatus::kInferSuccess)
        << current_op->layer->layer_name()
        << " layer forward failed, error code: " << int(status);
    current_op->has_forward = true;
    ProbeNextLayer(current_op, current_op->output_operands->datas);
}
```

执行普通类型的算子和执行输入类型的算子流程大致相同。但是在执行普通类型的算子时，会首先调用当前算子的`Forward`版本。在当前算子执行完毕后，**通过`ProbeNextLayer`函数将其输出赋值给后继节点的输入张量。**

### 返回模型推理的输出

执行普通类型的算子和执行输入类型的算子流程大致相同。但是，在执行普通类型的算子时，会首先调用当前算子的`Forward`版本。在当前算子执行完毕后，通过`ProbeNextLayer`函数将其输出赋值给后继节点的输入张量。

```cpp
if (operators_maps_.find(output_name_) != operators_maps_.end()) {
    const auto& output_op = operators_maps_.at(output_name_);
    CHECK(output_op->output_operands != nullptr)
        << "Output from" << output_op->name << " is empty";
    const auto& output_operand = output_op->output_operands;
    return output_operand->datas;
}
```

在所有算子执行完毕后，我们需要首先查询与`output_name`名称对应的算子（`output_op`），然后获取该算子对应的输出张量，即模型的相应输出`output_operand`.

## Resnet网络需要的算子

Resnet模型是一种非常经典的图像分类模型，对于了解过深度学习的同学应该不陌生。虽然它属于较复杂的模型，需要多个算子构成整个网络结构，但这些算子大多是我们以前讲过的，比如卷积层、最大池化层、ReLU激活函数等。

此外，Resnet还需要一些我们之前没讲到的算子，比如自适应池化层、全连接层等。下面我将列举构成Resnet模型的所有主要算子：

1. 卷积层
2. ReLU激活层
3. 自适应平均池化层（Adaptive Average Pooling）
4. 全连接层（Fully Connected Layer）也叫线性层（Linear Layer）
5. 最大池化层

## Linear算子的编写和注册

### Linear算子的初始化

Linear算子的初始化函数接口如下:

```cpp
ParseParameterAttrStatus LinearLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& linear_layer) {
```

首先需要获得线性层(Linear Layer)的所有相关参数。

具体来说,需要获得以下内容:

1. 权重矩阵（weight）：该矩阵表示输入特征与输出特征之间的连接权重，其维度为输出特征数 × 输入特征数，也就是$output\,features\times input \,features$。权重矩阵反映了线性层中每个输入特征对每个输出特征的贡献程度；
2. 偏置向量（bias）：该向量的长度为输出特征数，表示线性层中每个输出特征的偏置量。偏置用于调节线性层的输出结果，其维度是$output \,features$；
3. 是否使用偏置向量(use_bias)，如果是的话，我们再去读取偏置向量相关的权重。

随后，我们就通过线性转换公式“输出特征 = 权重矩阵 × 输入特征 + 偏置向量”，也就是$output\,features=weight\times input\, features+bias$，可以计算得到线性层的输出特征。

```cpp
const auto& params = op->params;
if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the use bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
}
const auto& use_bias_param =
    dynamic_cast<RuntimeParameterBool*>(params.at("bias"));
if (use_bias_param == nullptr) {
    LOG(ERROR) << "Can not find the use bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
}
```

获取`RuntimeOperator`中的权重信息，并将权重赋值给`Linear Layer`.

```cpp
const auto& attr = op->attribute;
CHECK(!attr.empty()) << "Operator attributes is empty";

if (attr.find("weight") == attr.end()) {
    LOG(ERROR) << "Can not find the weight parameter";
    return ParseParameterAttrStatus::kAttrMissingWeight;
}

...
const auto& weight = attr.at("weight");
linear_layer->set_weights(weight->get<float>());
```

获取`RuntimeOperator`中的偏移量权重，并且在这个算子使用偏移量的情况下，将偏移量权重赋值给`Linear Layer`.

```cpp
const auto& bias = attr.at("bias");
if (use_bias) {
    linear_layer->set_bias(bias->get<float>());
}
```

怎么实例化一个`Linear Layer`？我们从`weight`权重中获取了输出特征数`out_features`和输入特征数`in_features`变量，用于初始化`Linear Layer`。

```cpp
const auto& weight = attr.at("weight");
...
const auto& shapes = weight->shape;
int32_t out_features = shapes.at(0);
int32_t in_features = shapes.at(1);
linear_layer =
std::make_shared<LinearLayer>(in_features, out_features, use_bias);
```

### Linear层的初始化

[PyTorch Linear层定义](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html )，我们从这里可以看到Linear层的计算方式为$y=xA^{T}+b$，其中A表示权重矩阵。值得注意的是，在计算之前需要先对权重进行转置，在`Kuiper Infer`中，线性层的计算过程体现在`Forward`函数中，我们接下来将对这个函数进行逐行分析。

```cpp
InferStatus LinearLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
    ...
    ...
    uint32_t batch = inputs.size();
    const std::shared_ptr<Tensor<float>>& weight = weights_.front();
    arma::fmat weight_data(weight->raw_ptr(), out_features_, in_features_, false,true);
    const arma::fmat& weight_data_t = weight_data.t();

```

在以上的代码中，我们将权重矩阵加载到`weight_data`变量中，并对其进行了转置，得到转置后的权重矩阵`weight_data_t`，转置后的权重矩阵维度为$in\_features\times out\_features$.

```cpp
arma::fmat input_vec((float*)input->raw_ptr(), feature_dims, in_features_,
                     false, true);
```

在以上的代码中，我们将输入数据加载到`input_vec`中，它的维度由`feature_dims`和`in_features`确定。在这里，我们可以观察到`input`的维度为$? \times in\_features$，而权重矩阵的维度为$in\_features\times out\_features$。这样它们的维度就完成了对齐，从而可以在接下来步骤中完成$x\times W^{T}$.

```cpp
arma::fmat& result = output->slice(0);
result = input_vec * weight_data_t;
if (use_bias_) {
    ...
    ...
    const auto& bias_data = bias_.front()->data();
	...
    const auto& bias_tensor = bias_data.slice(0);
    for (uint32_t row = 0; row < result.n_rows; ++row) {
        result.row(row) += bias_tensor;
    }
```

我们在执行函数Forward中只要依次对这个算子序列中的所有算子依次执行就可以了。在以上代码中，我们把输入向量`input_vec`与权重矩阵`weight_data`进行矩阵乘法，计算得到结果`result`。如果这个线性层使用了偏置`bias`，我们还需要将其加到结果`result`中。

## Resnet分类网络的推理

> 从本节开始，我们将编写一个使用推理框架进行图像分类的示例程序。该程序的功能是输入一张图像，并输出图像所属的类别。

下面让我们看看在Python中如何使用PyTorch对输入图像进行分类，大致可以分为以下几步：

1. 加载预训练的模型
2. 读取输入图像并进行预处理
3. 使用模型对图像进行前向传播,得到预测结果
4. 对预测结果进行后处理,得到图像的类别
5. 输出分类结果

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    print(torch.version)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()

    img = Image.open(r'imgs/d.jpeg')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(img)
    input_batch = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(top5_catid[i], top5_prob[i].item())
```

### Resnet网络在KuiperInfer中的加载

我们在以下的代码中，完成了对`Resnet`模型的加载，并且在`Build`函数中指定了模型的输入和输出节点。

```cpp
using namespace kuiper_infer;
const std::string& param_path = "course8/model_file/resnet18_batch1.param";
const std::string& weight_path = "course8/model_file/resnet18_batch1.pnnx.bin";
RuntimeGraph graph(param_path, weight_path);
graph.Build("pnnx_input_0", "pnnx_output_0");
```

### 数据的预处理

```cpp
kuiper_infer::sftensor PreProcessImage(const cv::Mat &image) {
    ...
    ...
}
```

以下的代码是`KuiperInfer`对输入图像的预处理，在代码中首先对输入的图像进行归一化。

```cpp
kuiper_infer::sftensor PreProcessImage(const cv::Mat &image){   
	...
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(224, 224));
}
```

以下的代码对加载的图像完成颜色空间的转换，从BGR格式转换到RGB格式。

```cpp
kuiper_infer::sftensor PreProcessImage(const cv::Mat &image){   
    ...
    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

    rgb_image.convertTo(rgb_image, CV_32FC3);
	...
```

到目前为止，颜色空间格式在内存中的存储形式是RGBRGBRGB的排列。为了适配PyTorch，我们还需要将颜色空间格式转换为RRRGGGBBB的排列形式。换句话说，我们就是将图像格式从NHWC转换为NCHW.

```cpp
kuiper_infer::sftensor PreProcessImage(const cv::Mat &image){ 
    ...
    ...
    cv::split(rgb_image, split_images);
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t input_c = 3;
    sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

    uint32_t index = 0;
    for (const auto &split_image: split_images) {
        assert(split_image.total() == input_w * input_h);
        const cv::Mat &split_image_t = split_image.t();
        memcpy(input->slice(index).memptr(), split_image_t.data,
               sizeof(float) * split_image.total());
        index += 1;
    }
    ...
```

首先使用**cv::split**函数将输入的图像拆分为三个通道，分别为R通道、G通道和B通道。每个通道数据的维度为$1 \times input\_w \times input\_w$. 随后再将逐通道的数据依次拷贝到输入张量$input$中。

```cpp
kuiper_infer::sftensor PreProcessImage(const cv::Mat &image){ 
    ...
    ...
    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;

    float var_r = 0.229f;
    float var_g = 0.224f;
    float var_b = 0.225f;
    assert(input->channels() == 3);
    input->data() = input->data() / 255.f;
    input->slice(0) = (input->slice(0) - mean_r) / var_r;
    input->slice(1) = (input->slice(1) - mean_g) / var_g;
    input->slice(2) = (input->slice(2) - mean_b) / var_b;
    return input;
}
```

接下来，我们对输入张量数据`input`进行标准化处理。通过对各通道数据分别减去均值再除以标准差，将每个通道中的数据转换为均值为0、标准差为1的分布，这样就可以实现对输入数据的标准化。

```cpp
std::vector<sftensor> inputs;
const std::string &path("model_file/car.jpg");
for (uint32_t i = 0; i < batch_size; ++i) {
    cv::Mat image = cv::imread(path);
    // 图像预处理
    sftensor input = PreProcessImage(image);
    inputs.push_back(input);
}
```

首先，我们使用`OpenCV`的`imread`函数读入输入图像`car.jpg`。`car.jpg`的内容如下所示，然后对读入的图像应用前面介绍过的预处理方法。

<img src="https://i.imgur.com/kIP1Ja0.jpg" style="zoom: 67%;" align="left" />

### 执行推理

```cpp
// 推理
const std::vector<sftensor> outputs = graph.Forward(inputs, true);
```

以上的代码不多解释了，就是调用graph的Forward方法对输入图像进行推理，并将推理的结果放到outputs张量数组中。在`Python`代码中，还有使用`softmax`来对输出向量进行后处理的步骤，所以我们在C++中也要有对应的后处理过程。

```python
probabilities = torch.nn.functional.softmax(output[0], dim=0)
```

### 后处理过程

```cpp
std::vector<sftensor> outputs_softmax(batch_size);
SoftmaxLayer softmax_layer(0);
softmax_layer.Forward(outputs, outputs_softmax);
```

我们先对上一步【执行推理】中的输出张量计算它的softmax.

```cpp
for (int i = 0; i < outputs_softmax.size(); ++i) {
    const sftensor& output_tensor = outputs_softmax.at(i);
    assert(output_tensor->size() == 1 * 1000);
    // 找到类别概率最大的种类
    float max_prob = -1;
    int max_index = -1;
    for (int j = 0; j < output_tensor->size(); ++j) {
        float prob = output_tensor->index(j);
        if (max_prob <= prob) {
            max_prob = prob;
            max_index = j;
        }
    }
    printf("class with max prob is %f index %d\n", max_prob, max_index);
}
```

随后，我们对softmax的输出张量求最大值，并找到其中概率最大的一种作为分类网络的预测结果。

### 结果分析

```sh
class with max prob is 0.663738 index 817
```

我们通过查阅ImageNet的类别表，可以知道第816+1个类别是运动型跑车，和本图能对应上。至此，我们完成了KuiperInfer对Resnet网络的推理。

## 课堂作业







