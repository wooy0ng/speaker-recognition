import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    def __init__(self, in_features):
        super(FeedForward, self).__init__()
        self.in_features = in_features

        self.feed_forward = nn.Sequential(
            nn.Linear(self.in_features, self.in_features*4, bias=True),
            nn.Tanh(),
            nn.Linear(self.in_features*4, self.in_features, bias=True),
        )
    def forward(self, x) -> torch.Tensor:
        out = self.feed_forward(x)
        return out


class MemorizedModel(nn.Module):
    def __init__(self, embed_size, output_size):
        super(MemorizedModel, self).__init__()

        self.embed_size = embed_size # 256
        self.output_size = output_size
        self.feed_forward = FeedForward(self.embed_size)
        self.to_bit = nn.Sequential(
            nn.Linear(self.embed_size, self.output_size, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        norm = F.layer_norm(x, [self.embed_size])
        out = self.feed_forward(norm)
        out = out + norm
        norm = F.layer_norm(out, [self.embed_size])
        out = self.feed_forward(norm)
        out = self.to_bit(out)
        return out

    
    