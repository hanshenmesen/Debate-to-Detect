# Debate-to-Detect

> 一个基于AI多角色辩论的新闻真伪检测系统, 论文地址: [[2505.18596] Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models (arxiv.org)](https://arxiv.org/abs/2505.18596)

## 🎯 项目介绍

Debate-to-Detect是一个新闻真伪检测系统，通过模拟多个AI角色进行结构化辩论来分析新闻的真实性。系统采用正反双方辩论的形式，由多个专业评判员从不同维度进行评估，最终给出REAL/FAKE的判决结果。

## ✨ 主要特性

- **多角色辩论系统**：正方/反方各4个角色（开场/反驳/自由辩论/结辞）
- **多维度评判**：从准确性、可靠性、逻辑性、清晰度、伦理性5个维度评估
- **智能领域检测**：自动识别新闻领域并为每个角色生成专业背景
- **记忆管理**：支持对话历史摘要，避免token溢出
- **灵活配置**：支持多种OpenAI模型，可调节温度和重试策略
- **结果保存**：支持JSON/TXT格式保存完整辩论过程

## 🏗️ 系统架构

```
├── main.py           # 主程序入口
├── engine.py         # 辩论引擎核心逻辑
├── agent.py          # AI代理和OpenAI接口
├── config.py         # 配置文件和角色定义
├── openai_utils.py   # OpenAI工具函数
└── example.txt       # 示例新闻文本
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenAI API密钥

### 安装依赖

```bash
pip install openai tiktoken backoff
```

### 配置设置

1. **设置API密钥**：

   ```python
   # 方式1：环境变量
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_API_BASE="https://api.openai.com/v1"

   # 方式2：直接修改config.py
   OPENAI_API_KEY = "your-api-key"
   OPENAI_API_BASE = "https://api.openai.com/v1"
   ```
2. **准备新闻文本**：
   创建一个文本文件（如 `example.txt`），内容为待检测的新闻文章。

### 运行示例

```bash
python main.py
```

## 📋 辩论流程

### 1. 角色分配

- **正方（Affirmative）**：认为新闻是真实的

  - `Affirmative_Opening`: 开场陈述
  - `Affirmative_Rebuttal`: 反驳对方
  - `Affirmative_Free`: 自由辩论
  - `Affirmative_Closing`: 结辞总结
- **反方（Negative）**：认为新闻是虚假的

  - `Negative_Opening`: 开场陈述
  - `Negative_Rebuttal`: 反驳对方
  - `Negative_Free`: 自由辩论
  - `Negative_Closing`: 结辞总结

### 2. 评判维度

- **准确性（Accuracy）**：事实准确性评估
- **可靠性（SourceReliability）**：信息源可靠性
- **逻辑性（Reasoning）**：推理和内在一致性
- **清晰度（Clarity）**：语言清晰度和中性程度
- **伦理性（Ethics）**：伦理责任和潜在危害

### 3. 评分机制

- 每个评判员给正反双方整数打分，总分为7分（1:6 or 2:5 or 3:4），从机制上避免了判别平局的产生。
- 最终汇总所有维度得分

## ⚙️ 配置说明

### 模型配置

```python
SUPPORT_MODELS = [
    "gpt-4o-mini",
    "gpt-4o", 
    "gpt-4",
    "gpt-3.5-turbo"
]
```

### 系统参数

```python
MAX_COMPLETION_TOKENS = 1024        # 最大生成token数
MEMORY_SUMMARIZE_THRESHOLD = 20     # 记忆摘要阈值
MEMORY_KEEP_RECENT = 10            # 保留最近对话数
FREE_ROUNDS = 1                    # 自由辩论轮次
```

### 添加新的评判维度

在 `config.py`中修改 `DIMENSIONS`字典：

```python
DIMENSIONS = {
    "Accuracy": "factual accuracy",
    "YourDimension": "your custom dimension description"
}
```

### 调整辩论流程

修改 `PHASES`配置来自定义辩论阶段：

```python
PHASES = [
    ("YourPhase", ["Speaker1", "Speaker2"], "Your prompt template"),
    # ...
]
```

## 🛠️ 技术特性

- **重试机制**：使用指数退避策略处理API限流
- **Token管理**：自动计算和限制token使用量
- **记忆优化**：对话历史过长时自动摘要
- **错误处理**：完善的异常处理和日志记录
- **并发支持**：支持设置请求间隔避免频率限制

## 📄 许可证

本项目基于MIT许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱：hanshenmesen@163.com
