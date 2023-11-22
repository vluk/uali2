import torch
import torch.nn as nn

NUM_CHANNELS = 16
BOARD_SIZE = 20 * 10

class CNBlock(nn.Module):
    """Implementation of ConvNext block"""
    def __init__(self, dim, stride=1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 5, padding=2, stride=stride, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)

    def forward(self, x):
        out = self.dwconv(x)
        out = torch.permute(out, (0, 2, 3, 1))
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = torch.permute(out, (0, 3, 1, 2))
        out = out + x
        return out

class Representation(nn.Module):
    def __init__(self, input_dims, dim, blocks):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dims[0], dim, 3, stride=1, padding=1),
        )
        for _ in range(blocks):
            self.layers.append(CNBlock(dim))

    def forward(self, x):
        x = self.layers(x)
        return x

class Policy(nn.Module):
    def __init__(self, input_dims, dim, action_dim):
        super().__init__()
        self.pwconv = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(dim * input_dims[1] * input_dims[2], action_dim)
    
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.flat(x)
        x = self.fc(x)
        x = torch.reshape(x, (-1, 8, 20, 10))
        return x

class Value(nn.Module):
    def __init__(self, input_dims, dim):
        super().__init__()
        self.pwconv = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(dim * input_dims[1] * input_dims[2], dim)
        self.fc2 = nn.Linear(dim, 1)
    
    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class NNet(nn.Module):
    def __init__(self, input_dims = (2, 20, 10), dim=16, blocks=5, action_dim=8*20*10):
        super().__init__()
        self.embed = nn.Linear(49, 20 * 10)
        self.rep = Representation(input_dims, dim, blocks)
        self.p = Policy(input_dims, dim, action_dim)
        self.v = Value(input_dims, dim)
    
    def forward(self, obs):
        b, q = obs
        board, queue = torch.FloatTensor(b).cuda(), torch.FloatTensor(q).cuda()

        queue = torch.reshape(queue, (-1, 49))

        queue_embedding = self.embed(queue)
        x = torch.stack([board, torch.reshape(queue_embedding, (-1, 20, 10))], dim=1)

        x = self.rep(x)
        p = self.p(x)
        v = self.v(x)

        return p, v
