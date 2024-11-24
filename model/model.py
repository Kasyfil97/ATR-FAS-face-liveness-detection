import numpy as np
from torch import nn
from model.res_u_net import ResUNet
from model.easy_res_u_net import EasyResUNet
from model.base_layer import DownConvNormAct, ConvNormAct, Reshape, L2Normalize, Mean
import torch

# type gating nerwork
class TyepGatingNetwork(nn.Module):
    def __init__(self, number_attacks=3):
        super(TyepGatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            DownConvNormAct(3, 32), # [1, 3, 6, 256, 256] -> [1, 32, 6, 128, 128]
            DownConvNormAct(32, 64), # [1, 32, 6, 128, 128] -> [1, 64, 6, 64, 64]
            DownConvNormAct(64, 64, kernel_size=7),  # [1, 64, 6, 64, 64] -> [1, 64, 6, 32, 32]
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),  # [1, 64, 6, 32, 32] -> [1, 64, 6, 1, 1]
            Reshape(64), # [1, 64, 1, 1, 1] -> [1, 64]
            nn.Linear(64, 32), # [1, 32]
            nn.Linear(32, number_attacks), # [1, 3]
        )
    def forward(self, x):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())
        assert x.shape[2]==6, "input tensor must have 6 frames, but got {}".format(x)
        assert x.shape[1]==3, "input tensor must have 3 channels, but got {}".format(x) 

        x = self.gate(x)
        x = torch.softmax(x, dim=-1)
        return x
    
# MEMM
class MEMM(nn.Module):
    def __init__(self, num_batches: int, num_frames: int=6):
        super(MEMM, self).__init__()
        self.head_stem = nn.Sequential(
            DownConvNormAct(3, 32), # [6, 32, 128, 128]
            DownConvNormAct(32, 64), # [6, 64, 64, 64]
        )
        self.positional_embedding = torch.nn.Parameter(torch.randn(num_batches, 64, num_frames, 64, 64)) # [B, 64, N, 64, 64]
        self.depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, 1, -1]).astype(np.float32)
        self.resunet = ResUNet()

    @staticmethod
    def pixel_wise_softmax(x):
        """
        Applies a pixel-wise softmax operation to the input tensor.

        The function moves the channel dimension to the last position, computes the 
        exponential of each element subtracted by the maximum value in its channel 
        (for numerical stability), and normalizes by the sum of exponentials along 
        the channel dimension.
        """
        # Move the channel dimension to the last
        x = x.permute(0, 2, 3, 4, 1)
        channel_max, _ = torch.max(x, dim=4, keepdim=True)
        exponential_map = torch.exp(x - channel_max)
        normalize = torch.sum(exponential_map, dim=4, keepdims=True)
        return exponential_map / (normalize + 1e-5)
    
    def forward(self, x: torch.tensor, type_gating: torch.tensor):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())
        assert x.shape[2]==6, "input tensor must have 6 frames, but got {}".format(x)
        assert x.shape[1]==3, "input tensor must have 3 channels, but got {}".format(x)

        # input embedding
        x = self.head_stem(x) # [B, 3, N, 256, 256] -> [B, 64, N, 64, 64]
        x = x + self.positional_embedding
        # number of types
        M = type_gating.shape[1]

        # result from 
        x_bar = [self.resunet(x) for _ in range(M)]
        type_gating = torch.reshape(type_gating, [-1, 3, 1, 1, 1]) # [B, M] -> [M, B, 1, 1, 1]
        
        x_prime = sum(x_bar[i] * type_gating[:, i:i+1, :, :, :] for i in range(M))
        depth_softmax = MEMM.pixel_wise_softmax(x_prime)

        depth_map_cof = torch.from_numpy(self.depth_map_cor)
        depth_map = torch.sum(depth_softmax * depth_map_cof, axis=-1)
        
        return depth_map
    
# attention gating network
class AttentionGatingNetwork(nn.Module):
    def __init__(self):
        super(AttentionGatingNetwork, self).__init__()
        self.attention_stem = nn.Sequential(
            DownConvNormAct(3, 16),
            DownConvNormAct(16, 32),
        )
        self.easyunet = EasyResUNet()
        self.depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, 1, -1]).astype(np.float32)

    @staticmethod
    def pixel_wise_softmax(x):
        """
        Applies a pixel-wise softmax operation to the input tensor.

        The function moves the channel dimension to the last position, computes the 
        exponential of each element subtracted by the maximum value in its channel 
        (for numerical stability), and normalizes by the sum of exponentials along 
        the channel dimension.
        """
        # Move the channel dimension to the last
        x = x.permute(0, 2, 3, 4, 1)
        channel_max, _ = torch.max(x, dim=3, keepdim=True)
        exponential_map = torch.exp(x - channel_max)
        normalize = torch.sum(exponential_map, dim=3, keepdims=True)
        return exponential_map / (normalize + 1e-5)

    def forward(self, x: torch.tensor):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())
        assert x.shape[2]==6, "input tensor must have 6 frames, but got {}".format(x)
        assert x.shape[1]==3, "input tensor must have 3 channels, but got {}".format(x)

        atten_x = self.attention_stem(x) # convert from (N, C, H, W) to (N, 32, 64, 64)
        attention_x = self.easyunet(atten_x)

        attention_soft_max = AttentionGatingNetwork.pixel_wise_softmax(attention_x)
        
        depth_map_cof = torch.from_numpy(self.depth_map_cor)

        dot = depth_map_cof * attention_soft_max
        summation = torch.sum(dot, dim=-1)
        attention_map = torch.unsqueeze(summation, dim=1)
        attention_map = torch.softmax(torch.reshape(attention_map, [attention_map.shape[0],6, 64, 64]), dim=1)
        return attention_map

# classification head
class ClassificationHead(nn.Module):
    def __init__(self, return_proba = False):
        super(ClassificationHead, self).__init__()
        self.return_proba = return_proba
        self.f_net = nn.Sequential(
            DownConvNormAct(1, 16),  # 32*32*16
            ConvNormAct(16, 8, 3),  # 32*32*8
            ConvNormAct(8, 4, 3),  # 32*32*4
            Reshape(32 * 32 * 4),
            L2Normalize(1),
            nn.Linear(32 * 32 * 4, 2),
        )
    
    def forward(self, x: torch.tensor):
        assert x.dim() == 3, "input tensor must be 3D, but got {}D".format(x.dim())

        x_final = x.unsqueeze(1).unsqueeze(2)
        pred_logits = self.f_net(x_final)
        if self.return_proba:
            return torch.softmax(pred_logits, dim=-1)
        return pred_logits
    
class ATRFAS(nn.Module):
    def __init__(self, num_batches: int, num_frames: int=6, return_proba: bool=True, infer_type: str='inference'):
        super(ATRFAS, self).__init__()
        self.type_gating_network = TyepGatingNetwork()
        self.memm = MEMM(num_batches, num_frames)
        self.attention_gating_network = AttentionGatingNetwork()
        self.classification_head = ClassificationHead(return_proba)

        self.infer_type = infer_type
    
    def forward(self, x: torch.tensor):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())

        x = x.permute(0, 2, 1, 3, 4) # [B, N, C, H, W] -> [B, C, N, H, W]

        type_gating = self.type_gating_network(x)
        frame_depth_map = self.memm(x, type_gating)
        frame_attention_map = self.attention_gating_network(x)
        depth_map = (frame_depth_map * frame_attention_map).sum(dim=1)
        pred = self.classification_head(depth_map)
        if self.infer_type == 'inference':
            return pred
        return pred, type_gating, frame_depth_map, frame_attention_map, depth_map
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        
        for name, param in state_dict.items():
            if name in own_state:
                print(f"Copying value to {name}")
                param_data = param.data if isinstance(param, nn.Parameter) else param
                try:
                    own_state[name].copy_(param_data)
                except Exception:
                    if 'tail' not in name:
                        raise RuntimeError(
                            f"Error copying parameter '{name}'. Model dimensions: {own_state[name].size()}, "
                            f"Checkpoint dimensions: {param_data.size()}"
                        )
            elif strict and 'tail' not in name:
                raise KeyError(f"Unexpected key '{name}' in state_dict")
