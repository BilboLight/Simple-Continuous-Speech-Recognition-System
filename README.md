# 简单的连续语音识别系统<br>

这是一个简单的语音识别系统，能进行连续的语音识别，能识别 35 个词汇。<br><br>

#### 1、数据集

在 `speech-commands` 数据集上训练，能识别数据集内的 35 个单词：

一、数字类：zero, one, two, three, four, five, six, seven, eight, nine

二、方向类：left, right, forward, backward, up, down

三、命令类：go, stop, yes, no, on, off, follow

四、动物类：bird, cat, dog

五、其他：bed, house, happy, tree, wow, learn, visual, sheila, marvin<br><br>


#### 2、模型文件: `speech_commands_model_epoch_20_9621--64mel.pth`

已经训练好的模型，模型在 `speech-commands` 测试集上的准确率为 **96.05%**。<br><br>


#### 3、训练代码: `train.py`

使用 `speech-commands` 数据集训练语音识别模型。<br><br>


#### 4、推理代码: `Inference.ipynb`

加载已经训练好的模型，进行语音识别，支持以下功能：

1. 识别单个 `.wav` 音频文件对应的单词。
2. 识别文件夹内所有 `.wav` 音频文件对应的单词。
3. 录音 2 秒，识别所说的单词。
4. 连续录音，识别所说的一系列单词，并给出每个单词的 (开始时间, 结束时间)。<br>

安装了依赖库就可以使用，无需其他配置。<br><br>

#### 5、如果觉得不错，给个 star 吧，谢谢~~
<br><br><br><br>

---
# Simple Continuous Speech Recognition System<br>
This is a simple speech recognition system capable of continuous speech recognition, with a vocabulary of 35 words.<br><br>

#### 1. Dataset

The model is trained on the `speech-commands` dataset, which can recognize 35 words from the dataset:

1. **Numbers**: zero, one, two, three, four, five, six, seven, eight, nine
2. **Directions**: left, right, forward, backward, up, down
3. **Commands**: go, stop, yes, no, on, off, follow
4. **Animals**: bird, cat, dog
5. **Others**: bed, house, happy, tree, wow, learn, visual, sheila, marvin

<br>

#### 2. Model File: `speech_commands_model_epoch_20_9621--64mel.pth`

This is a pre-trained model. The accuracy on the `speech-commands` test set is **96.05%**.

<br>

#### 3. Training Code: `train.py`

This code trains the speech recognition model using the `speech-commands` dataset.

<br>

#### 4. Inference Code: `Inference.ipynb`

The inference script can load the pre-trained model and perform speech recognition. It supports the following functionalities:

1. Recognize the word corresponding to a single `.wav` audio file.
2. Recognize the words corresponding to all `.wav` audio files in a folder.
3. Record a 2-second audio clip and recognize the spoken word.
4. Continuously record audio, recognize a series of spoken words, and provide the (start time, end time) for each word.<br>

Once the dependencies are installed, you can use it without any additional configuration.<br><br>

#### 5、If you like it, please give it a star. Thank you very much.
<br><br><br><br>
