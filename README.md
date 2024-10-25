# 简单语音识别系统

## 数据集

在 `speech-commands` 数据集上训练，能识别数据集内的 35 个单词：

### 数字类
- zero, one, two, three, four, five, six, seven, eight, nine

### 方向类
- left, right, forward, backward, up, down

### 命令类
- go, stop, yes, no, on, off, follow

### 动物类
- bird, cat, dog

### 其他
- bed, house, happy, tree, wow, Sheila, learn, visual, Marvin

---

## 模型文件: `speech_commands_model_epoch_20_9621--64mel.pth`

这是已经训练好的模型，模型在 `speech-commands` 测试集上的准确率为 **96.05%**。

---

## 训练代码: `train.py`

该脚本使用 `speech-commands` 数据集训练语音识别模型。

---

## 推理代码: `Inference.ipynb`

`Inference.ipynb` 文件加载已经训练好的模型，提供语音识别功能，支持以下功能：

1. **识别单个 `.wav` 音频文件** 对应的单词。
2. **识别文件夹内所有 `.wav` 音频文件** 对应的单词。
3. **录音 2 秒**，识别所说的单词。
4. **连续录音**，识别所说的一系列单词，并给出每个单词的 **(开始时间, 结束时间)**。
