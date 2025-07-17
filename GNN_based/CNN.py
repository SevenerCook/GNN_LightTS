import torch.nn as nn

#source 64,96,7
#target 64,96,7
class Basic1DCNN(nn.Module):
    def __init__(self, input_dim=96, out_dim=10):
        super().__init__()
        self.features = nn.Sequential(
            # 卷积核覆盖局部节点关系
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 压缩序列长度

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(7)  # 全局池化 → 定长输出
        )
        self.classifier = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] → 转置为 [batch, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.features(x)  # 输出 [batch, 128, 1]
        x = x.view(x.size(0), -1)   #输出[batch,128]
        return x