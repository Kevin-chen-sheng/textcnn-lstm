import torch.nn as nn
import torch.nn.functional as F
import torch


# ======================================================================================================================
# 定义网络模型
# ======================================================================================================================
class cnn(nn.Module):

    def __init__(self, weight1, weight2, config):
        super(cnn, self).__init__()

        self.config = config
        vocab_size, dim = weight1.shape
        self.vovab_size = vocab_size
        self.dim = dim

        # embedding
        if config.static:
            config.freeze = True
        else:
            config.freeze = False
        self.embedding = nn.Embedding(vocab_size, dim).from_pretrained(embeddings=weight1, freeze=config.freeze)

        if config.multichannel:
            self.embedding2 = nn.Embedding(vocab_size, dim).from_pretrained(embeddings=weight2, freeze=config.freeze)
            config.chanel_num += 1
        else:
            self.embedding2 = None

        filter_sizes = [3, 3, 3]
        self.convs = nn.ModuleList(
            [nn.Conv2d(config.chanel_num, config.filter_num, (size, dim)) for size in filter_sizes])

        # dropout layer
        self.dropout = nn.Dropout(config.dropout)

        self.l1 = nn.Linear(len(filter_sizes) * config.filter_num, config.num_classes)

    def forward(self, x):

        x = x.long()

        if self.embedding2:
            out = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            out = self.embedding(x)
            out = out.unsqueeze(1)

        out = [F.relu(conv(out)).squeeze(3) for conv in self.convs]

        out = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in out]

        out = torch.cat(out, 1)
        out = self.dropout(out)

        out = self.l1(out)

        return out
