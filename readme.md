# Adaptive PLE Transformer

English | [中文](#中文版)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/stellan-project/tqr_guided_learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arxiv badge](https://img.shields.io/badge/arXiv-2408.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2408.XXXXX)

This repository contains the official implementation of the **Adaptive Promoted Latent Embeddings (PLE) Transformer**, a novel architecture designed to enhance context handling and knowledge representation in sequence-to-sequence tasks.

The model introduces a dynamic, internal "concept space" and a **Dual-Stream Context** mechanism, allowing it to process both fine-grained details and high-level abstractions simultaneously. This makes it particularly powerful for complex generation and translation tasks where a deep understanding of the source material is crucial.

## ✨ Core Features

- **🧠 Dynamic Concept Space**: A learnable set of "concepts" (`ple_concepts`) within each layer that represent high-level ideas.
- **🌊 Dual-Stream Context**: The decoder receives two streams of information: a standard **Detail Stream** from the encoder's tokens and a unique **Abstract Stream** from the encoder's learned concepts.
- **🧬 Adaptive Mechanism**: The model can dynamically `grow` its concept space to learn new information and `prune` it to remove redundancies, creating an efficient, evolving knowledge structure.
- **⚖️ Diversity Regularization**: A specialized loss function that encourages the concepts to be distinct and non-redundant, leading to a more efficient and powerful internal representation.
- **🚀 Superior Performance**: Outperforms standard Transformer baselines in both learning efficiency and final task performance.

## 🏛️ Architecture Overview

The `EncoderDecoderPLEModel` is the core of the architecture.

1.  **Encoding**: The `PLEEncoder` processes the input sequence. In each layer, it uses standard self-attention and then performs **PLE Resonance**, where the sequence attends to the internal `ple_concepts` to retrieve relevant high-level context.
2.  **Concept Consolidation**: The learned `ple_concepts` from all encoder layers are collected and concatenated into a single `ple_memory` tensor.
3.  **Decoding with Dual Streams**: The `PLEDecoder` generates the output. In each layer, it performs three attention steps:
    1.  **Causal Self-Attention** on the generated output.
    2.  **Cross-Attention (Detail Stream)** to the encoder's final hidden states.
    3.  **PLE Cross-Attention (Abstract Stream)** to the consolidated `ple_memory`.

This dual-stream approach gives the decoder unprecedented access to both the "what" (details) and the "why" (abstractions) of the source text.

## 🚀 Best Practices for Training

We recommend a **"Phased Adaptation"** strategy for large-scale training to ensure stability and efficiency:

1.  **Phase 1: Growth**: Allow the model to grow its concept space (`grow_concepts()`) to match data complexity.
2.  **Phase 2: Stable Training**: Disable both growth and pruning for the majority of training to ensure stable convergence.
3.  **Phase 3: Pruning/Distillation**: Enable `prune_concepts()` to distill the learned knowledge into a more efficient, compact representation.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/stellan-project/tqr_guided_learning.git
cd tqr_guided_learning

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from ple_transformer import EncoderDecoderPLEModel, PLEConfig

# 1. Define the model configuration
config = PLEConfig(
    vocab_size=32000,
    n_layers=6,
    n_heads=8,
    d_model=512,
    ple_concepts_per_layer=128,
    # ... other parameters
)

# 2. Initialize the model
model = EncoderDecoderPLEModel(config)

# 3. Forward pass (example)
# (Assuming source_ids and target_ids are tokenized tensors)
outputs = model(
    input_ids=source_ids,
    decoder_input_ids=target_ids
)

logits = outputs.logits
```

---

## <a name="中文版"></a> Adaptive PLE Transformer (中文版)

[English](#) | 中文

[![构建状态](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/stellan-project/tqr_guided_learning)
[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arxiv badge](https://img.shields.io/badge/arXiv-2408.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2408.XXXXX)

本代码库是 **自适应提升潜在嵌入 (Adaptive Promoted Latent Embeddings, PLE) Transformer** 的官方实现。这是一种旨在增强序列到序列任务中上下文处理和知识表示能力的新型架构。

该模型引入了一个动态的、内在的“概念空间”，以及一种**双流上下文（Dual-Stream Context）**机制，使其能够同时处理细粒度的细节和高层次的抽象信息。这使得它在需要深度理解源材料的复杂生成和翻译任务中尤为强大。

## ✨ 核心特性

- **🧠 动态概念空间**：每个层级内部都有一组可学习的“概念”（`ple_concepts`），用于表示高阶思想。
- **🌊 双流上下文**：解码器接收两个信息流：一个来自编码器词元（Token）的标准**细节流**，和一个来自编码器所学概念的独特**抽象流**。
- **🧬 自适应机制**：模型可以动态地`增长`其概念空间以学习新信息，并`修剪`它以消除冗余，从而创建一个高效、持续演化的知识结构。
- **⚖️ 多样性正则化**：一种特制的损失函数，鼓励概念之间保持独特性和非冗余性，从而引导模型形成更高效、更强大的内部表示。
- **🚀 卓越性能**：在学习效率和最终任务性能上均优于标准的 Transformer 基线模型。

## 🏛️ 架构概览

`EncoderDecoderPLEModel` 是该架构的核心。

1.  **编码阶段**：`PLEEncoder` 处理输入序列。在每一层，它首先使用标准的自注意力，然后执行 **PLE 共鸣**，即序列向内部的 `ple_concepts` 进行注意力计算，以检索相关的高阶上下文。
2.  **概念整合**：从所有编码器层学习到的 `ple_concepts` 被收集并拼接成一个单一的 `ple_memory` 张量。
3.  **双流解码**：`PLEDecoder` 负责生成输出。在每一层，它执行三个注意力步骤：
    1.  对已生成内容的**因果自注意力**。
    2.  对编码器最终隐藏状态的**交叉注意力（细节流）**。
    3.  对整合后的 `ple_memory` 的 **PLE 交叉注意力（抽象流）**。

这种双流方法为解码器提供了前所未有的能力，使其能够同时理解源文本的“内容细节”（what）和“核心主旨”（why）。

## 🚀 训练最佳实践

对于大规模训练，我们推荐采用一种**“分阶段自适应”**策略，以确保稳定性和效率：

1.  **第一阶段：增长期**：允许模型增长其概念空间（调用 `grow_concepts()`），以匹配数据的复杂性。
2.  **第二阶段：稳定训练期**：在大部分训练时间内禁用增长和修剪，以确保稳定的收敛。
3.  **第三阶段：修剪/蒸馏期**：启用 `prune_concepts()`，将学到的知识蒸馏成一个更高效、更紧凑的表示。

## 📦 安装

```bash
# 克隆代码库
git clone https://github.com/stellan-project/tqr_guided_learning.git
cd tqr_guided_learning

# 安装依赖
pip install -r requirements.txt
```

## 快速上手

```python
from ple_transformer import EncoderDecoderPLEModel, PLEConfig

# 1. 定义模型配置
config = PLEConfig(
    vocab_size=32000,
    n_layers=6,
    n_heads=8,
    d_model=512,
    ple_concepts_per_layer=128,
    # ... 其他参数
)

# 2. 初始化模型
model = EncoderDecoderPLEModel(config)

# 3. 前向传播 (示例)
# (假设 source_ids 和 target_ids 是已分词的张量)
outputs = model(
    input_ids=source_ids,
    decoder_input_ids=target_ids
)

logits = outputs.logits
```
