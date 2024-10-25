import os
import pathlib
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob

# 安装必要的库（如果尚未安装）
# 请取消以下行的注释以安装 huggingface_hub
# !pip install huggingface_hub
# 请取消以下行的注释以安装 speechbrain（如果需要）
# !pip install speechbrain

from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub import notebook_login


# 设定调试信息函数
def debug_print(message, level="INFO"):
    print(f"[{level}] {message}")


# 1. 设置数据目录并确保它存在
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
debug_print(f"数据目录设置为: {data_dir}")

# 2. 定义数据集类
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, subset, labels_to_index, augment_pipeline=None, concat_augment=True):
        super().__init__("./data", subset=subset, download=True)
        self.subset = subset
        if subset == 'validation':
            self._walker = self._load_list('validation_list.txt')
        elif subset == 'testing':
            self._walker = self._load_list('testing_list.txt')
        elif subset == 'training':
            excludes = self._load_list('validation_list.txt') + self._load_list('testing_list.txt')
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

        # 标签映射
        self.labels_to_index = labels_to_index

        # 数据增强管道
        self.augment_pipeline = augment_pipeline
        self.concat_augment = concat_augment

        debug_print(f"{subset.capitalize()} 集合样本数量: {len(self)}")

    def _load_list(self, filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as file:
            lines = [os.path.normpath(os.path.join(self._path, line.strip())) for line in file]
        debug_print(f"加载文件 {filename}，共 {len(lines)} 条记录", level="DEBUG")
        return lines

    def _get_label(self, filepath):
        # 从文件路径中提取标签
        return os.path.basename(os.path.dirname(filepath))

    def __getitem__(self, n):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        target = self.labels_to_index[label]

        waveforms = [waveform]
        targets = [target]

        if self.augment_pipeline:
            for augment in self.augment_pipeline:
                augmented_waveform = augment(waveform, sample_rate)
                if augmented_waveform is not None:
                    waveforms.append(augmented_waveform)
                    targets.append(target)

        if self.concat_augment:
            return waveforms, targets
        else:
            # 随机选择一个增强方式
            idx = random.randint(0, len(waveforms) - 1)
            return waveforms[idx], targets[idx]

    def __len__(self):
        return len(self._walker)


# 3. 动态提取所有唯一标签并创建映射
# 首先创建一个临时的训练集实例以提取标签
temp_train_set = SpeechCommandsDataset('training', labels_to_index={})
all_labels = sorted(list(set([temp_train_set._get_label(w) for w in temp_train_set._walker])))
labels_to_index = {label: idx for idx, label in enumerate(all_labels)}
index_to_labels = {idx: label for label, idx in labels_to_index.items()}
num_classes = len(all_labels)
debug_print(f"Number of classes: {num_classes}")
debug_print(f"Classes: {all_labels}", level="DEBUG")

# 检查标签映射
debug_print("标签到索引映射:")
for label, idx in labels_to_index.items():
    debug_print(f"    {label}: {idx}", level="DEBUG")


# 4. 定义数据增强模块

class TimeDomainSpecAugment:
    """
    实现音频速度变化的增强类。

    参数:
    - sample_rate (int): 原始音频的采样率
    - speeds (list): 可选的速度变化百分比列表，默认为[100]（即不变）
    """

    def __init__(self, sample_rate, speeds=[100]):
        self.sample_rate = sample_rate
        self.speeds = speeds

    def __call__(self, waveform, sample_rate):
        """
        对输入的波形进行速度变化。

        参数:
        - waveform (Tensor): 输入的音频波形
        - sample_rate (int): 输入波形的采样率

        返回:
        - Tensor: 速度变化后的波形
        """
        speed = random.choice(self.speeds)
        if speed == 100:
            return waveform  # 速度不变，直接返回原波形
        # 速度变化实现：通过重采样改变速度
        new_rate = int(self.sample_rate * speed / 100)
        resampler = T.Resample(orig_freq=self.sample_rate, new_freq=new_rate)
        waveform = resampler(waveform)
        # 为了保持原始长度，重新采样回原频率
        resampler_back = T.Resample(orig_freq=new_rate, new_freq=self.sample_rate)
        waveform = resampler_back(waveform)
        return waveform


class Wavedrop:
    """
    实现随机将音频片段置零的增强类。

    参数:
    - drop_prob (float): 执行wavedrop的概率，默认为0.1
    - drop_length (float): 置零部分占总长度的比例，默认为0.1
    - sample_rate (int): 音频的采样率，默认为16000
    """

    def __init__(self, drop_prob=0.1, drop_length=0.1, sample_rate=16000):
        self.drop_prob = drop_prob
        self.drop_length = drop_length
        self.sample_rate = sample_rate

    def __call__(self, waveform, sample_rate):
        """
        对输入的波形进行wavedrop操作。

        参数:
        - waveform (Tensor): 输入的音频波形
        - sample_rate (int): 输入波形的采样率

        返回:
        - Tensor: 经过wavedrop处理后的波形
        """
        if random.random() < self.drop_prob:
            drop_samples = int(self.drop_length * sample_rate)
            if waveform.size(1) <= drop_samples:
                start = 0
            else:
                start = random.randint(0, waveform.size(1) - drop_samples)
            waveform[:, start:start + drop_samples] = 0
        return waveform


class SpectralDropout(nn.Module):
    def __init__(self, p=0.1):
        """
        初始化 SpectralDropout 模块。

        参数:
        - p (float): 每个频率通道被屏蔽的概率。
        """
        super(SpectralDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        前向传播。

        参数:
        - x (Tensor): 输入的谱图，形状为 (batch, n_mels, time)

        返回:
        - Tensor: 应用 Spectral Dropout 后的谱图
        """
        if not self.training or self.p == 0:
            return x
        # 生成屏蔽的掩码，形状为 (batch, n_mels, 1)
        mask = torch.ones_like(x)
        drop_mask = torch.bernoulli(torch.ones(x.size(0), x.size(1), 1) * (1 - self.p)).to(x.device)
        mask = mask * drop_mask
        return x * mask


class AddNoise:
    """
    实现向音频添加白噪声的增强类。

    参数:
    - noise_prob (float): 添加噪声的概率，默认为1.0
    - snr_low (int): 信噪比范围的下限，默认为0
    - snr_high (int): 信噪比范围的上限，默认为15
    - sample_rate (int): 音频的采样率，默认为16000
    """

    def __init__(self, noise_prob=1.0, snr_low=0, snr_high=15, sample_rate=16000):
        self.noise_prob = noise_prob
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.sample_rate = sample_rate
        # 生成白噪声
        self.noise = self.generate_noise()

    def generate_noise(self):
        """
        生成1秒长的白噪声。

        返回:
        - Tensor: 生成的白噪声
        """
        # 生成白噪声，持续时间为1秒
        noise = torch.randn(1, self.sample_rate)  # 1秒的噪声
        return noise

    def __call__(self, waveform, sample_rate):
        """
        对输入的波形添加白噪声。

        参数:
        - waveform (Tensor): 输入的音频波形
        - sample_rate (int): 输入波形的采样率

        返回:
        - Tensor: 添加噪声后的波形
        """
        if random.random() < self.noise_prob:
            snr = random.uniform(self.snr_low, self.snr_high)
            # 计算信号功率
            sig_power = waveform.pow(2).mean()
            # 生成与信号长度相同的噪声
            if self.noise.size(1) < waveform.size(1):
                # 循环噪声以匹配波形长度
                repeats = (waveform.size(1) // self.noise.size(1)) + 1
                noise = self.noise.repeat(1, repeats)[:, :waveform.size(1)]
            else:
                noise = self.noise[:, :waveform.size(1)]
            noise = noise.to(waveform.device)
            noise_power = noise.pow(2).mean()
            # 计算所需的噪声幅度
            desired_noise_power = sig_power / (10 ** (snr / 10))
            noise = noise * torch.sqrt(desired_noise_power / noise_power)
            # 添加噪声
            waveform = waveform + noise
        return waveform


def get_augmentations():
    """
    创建数据增强管道。

    返回:
    - list: 包含各种数据增强方法的列表
    """
    augment_wavedrop = Wavedrop(drop_prob=0.8, drop_length=0.1, sample_rate=16000)
    augment_speed = TimeDomainSpecAugment(sample_rate=16000, speeds=[90, 95, 100, 105, 110])
    add_noise = AddNoise(noise_prob=1.0, snr_low=0, snr_high=15, sample_rate=16000)
    # 组成增强管道
    augment_pipeline = [augment_wavedrop, augment_speed, add_noise]
    return augment_pipeline


# 5. 创建训练、验证和测试数据集
train_set = SpeechCommandsDataset(
    'training',
    labels_to_index=labels_to_index,
    augment_pipeline=get_augmentations(),
    concat_augment=True
)
valid_set = SpeechCommandsDataset(
    'validation',
    labels_to_index=labels_to_index,
    augment_pipeline=None,
    concat_augment=False  # 验证集不进行数据增强
)
test_set = SpeechCommandsDataset(
    'testing',
    labels_to_index=labels_to_index,
    augment_pipeline=None,
    concat_augment=False  # 测试集不进行数据增强
)

# 6. 定义数据加载器的 collate_fn
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    tensors = []
    targets = []
    for item in batch:
        waveforms, labels = item
        if isinstance(waveforms, list) or isinstance(waveforms, tuple):
            # 训练集返回的是多个增强后的波形和标签
            for waveform, label in zip(waveforms, labels):
                tensors.append(waveform.squeeze(0))
                targets.append(torch.tensor(label))
        else:
            # 验证集和测试集返回的是单个波形和标签
            waveform, label = item
            tensors.append(waveform.squeeze(0))
            targets.append(torch.tensor(label))
    tensors = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return tensors, targets


batch_size = 256

# 创建 DataLoader
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)
valid_loader = DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# 7. 定义特征提取器
n_mels = 64
sample_rate = 16000
mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
debug_print("定义 MelSpectrogram 作为特征提取器")


# 8. 定义 TDNN 层
class TDNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, context_size, dilation=1):
        super(TDNNLayer, self).__init__()
        self.context_size = context_size
        self.dilation = dilation
        self.tdnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=context_size,
            stride=1,
            dilation=dilation
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.tdnn(x)
        out = self.activation(out)
        return out


debug_print("定义 TDNNLayer")


# 9. 定义统计池化层
class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling, self).__init__()

    def forward(self, x):
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        stats = torch.cat((mean, std), dim=1)
        return stats


debug_print("定义 StatsPooling")


# 10. 定义 XVector 模型
class XVector(nn.Module):
    def __init__(self):
        super(XVector, self).__init__()
        self.tdnn1 = TDNNLayer(input_dim=64, output_dim=512, context_size=5, dilation=1)
        self.tdnn2 = TDNNLayer(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.tdnn3 = TDNNLayer(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.tdnn4 = TDNNLayer(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.tdnn5 = TDNNLayer(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.stats_pooling = StatsPooling()
        self.embedding1 = nn.Linear(3000, 512)
        self.embedding1_act = nn.LeakyReLU()

    def forward(self, x):
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats_pooling(x)
        x = self.embedding1(x)
        x = self.embedding1_act(x)
        return x


debug_print("定义 XVector 模型")


# 11. 定义分类器
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512, 512)  # 输入维度设为512（XVector的输出）
        self.fc1_act = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)
        return x


debug_print("定义 Classifier")


# 12. 定义整个模型
class SpeechCommandModel(nn.Module):
    def __init__(self, num_classes, spectral_dropout_p=0.1):
        super(SpeechCommandModel, self).__init__()
        self.feature_extractor = mel_spectrogram
        self.spectral_dropout = SpectralDropout(p=spectral_dropout_p)  # 添加 SpectralDropout
        self.xvector = XVector()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)  # 计算 Mel Spectrogram
        features = features.clamp(min=1e-9).log()  # 对数变换
        features = self.spectral_dropout(features)  # 应用 Spectral Dropout
        features = features.squeeze(1)  # 移除不必要的维度
        embedding = self.xvector(features)  # 提取 XVector
        outputs = self.classifier(embedding)  # 分类
        return outputs


debug_print("定义整个 SpeechCommandModel")

# 13. 创建模型、优化器、损失函数
model = SpeechCommandModel(num_classes=num_classes, spectral_dropout_p=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
debug_print(f"将模型移动到设备: {device}")

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000002)
criterion = nn.CrossEntropyLoss()
debug_print("定义优化器 (Adam) 和损失函数 (CrossEntropyLoss)")

# 打印模型结构和参数数量
debug_print("模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
debug_print(f"可训练参数总数: {total_params}")

# 14. 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
debug_print("设置随机种子为 42")

# 15. 训练模型
num_epochs = 20

# 创建用于保存最佳模型的变量
best_valid_acc = 0.0

for epoch in range(num_epochs):
    # 学习率调整（线性调度）
    lr_final = 0.0001
    lr_start = 0.001
    lr = lr_start - (lr_start - lr_final) * (epoch + 1) / num_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    debug_print(f"Epoch {epoch + 1}/{num_epochs}: 设置学习率为 {lr:.6f}")

    # 训练阶段
    model.train()
    running_loss = 0.0
    train_corrects = 0
    train_total = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for waveforms, labels in train_loader_tqdm:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)
        train_total += labels.size(0)

        current_loss = running_loss / train_total
        current_acc = train_corrects.double() / train_total
        train_loader_tqdm.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.4f}")

    epoch_loss = running_loss / train_total
    epoch_acc = train_corrects.double() / train_total
    debug_print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # 验证阶段
    model.eval()
    valid_corrects = 0
    valid_total = 0
    valid_loss = 0.0

    valid_loader_tqdm = tqdm(valid_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch")
    with torch.no_grad():
        for waveforms, labels in valid_loader_tqdm:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * waveforms.size(0)
            _, preds = torch.max(outputs, 1)
            valid_corrects += torch.sum(preds == labels.data)
            valid_total += labels.size(0)

            current_valid_loss = valid_loss / valid_total
            current_valid_acc = valid_corrects.double() / valid_total
            valid_loader_tqdm.set_postfix(loss=f"{current_valid_loss:.4f}", accuracy=f"{current_valid_acc:.4f}")

    valid_epoch_loss = valid_loss / valid_total
    valid_epoch_acc = valid_corrects.double() / valid_total
    debug_print(f"Validation - Loss: {valid_epoch_loss:.4f} - Accuracy: {valid_epoch_acc:.4f}")

    # 每个 epoch 保存模型
    model_save_path = f'speech_commands_model_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    debug_print(f"模型已保存到 {model_save_path}")

    # 如果当前验证准确率是最好的，则保存为最佳模型
    if valid_epoch_acc > best_valid_acc:
        best_valid_acc = valid_epoch_acc
        best_model_path = 'speech_commands_model_best.pth'
        torch.save(model.state_dict(), best_model_path)
        debug_print(f"最佳模型已更新并保存到 {best_model_path}")

# 16. 测试模型并收集预测结果
model.eval()

test_corrects = 0
test_total = 0
all_preds = []
all_labels = []

test_loader_tqdm = tqdm(test_loader, desc='Testing', unit='batch')
with torch.no_grad():
    for waveforms, labels in test_loader_tqdm:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        outputs = model(waveforms)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)
        test_total += labels.size(0)

        # 收集预测和真实标签
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条后面的准确率
        test_acc = test_corrects.double() / test_total
        test_loader_tqdm.set_postfix(accuracy=f'{test_acc.item():.4f}')

test_acc = test_corrects.double() / test_total
debug_print(f'Test Accuracy: {test_acc:.4f}')

# 17. 绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(24, 20))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=index_to_labels.values(), yticklabels=index_to_labels.values())
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('normalized_confusion_matrix.png')  # 保存混淆矩阵图
plt.show()
debug_print("混淆矩阵已保存为 'normalized_confusion_matrix.png'")