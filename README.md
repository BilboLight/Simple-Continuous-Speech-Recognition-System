# 简单语音识别系统

### 1）数据集

在 `speech-commands` 数据集上训练，能识别数据集内的 35 个单词：

#### （1）数字类
- zero, one, two, three, four, five, six, seven, eight, nine

#### （2）方向类
- left, right, forward, backward, up, down

#### （3）命令类
- go, stop, yes, no, on, off, follow

#### （4）动物类
- bird, cat, dog

#### （5）其他
- bed, house, happy, tree, wow, learn, visual, sheila, marvin


### 2）模型文件: `speech_commands_model_epoch_20_9621--64mel.pth`

已经训练好的模型，模型在 `speech-commands` 测试集上的准确率为 **96.05%**。


### 3）训练代码: `train.py`

使用 `speech-commands` 数据集训练语音识别模型。


### 4）推理代码: `Inference.ipynb`

加载已经训练好的模型，进行语音识别，支持以下功能：

1. **识别单个 `.wav` 音频文件** 对应的单词。
2. **识别文件夹内所有 `.wav` 音频文件** 对应的单词。
3. **录音 2 秒**，识别所说的单词。
4. **连续录音**，识别所说的一系列单词，并给出每个单词的 **(开始时间, 结束时间)**。
