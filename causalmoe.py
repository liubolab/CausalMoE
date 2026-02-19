import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
# import plotly.express as px
from torch.amp import autocast
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau,OneCycleLR,CosineAnnealingWarmRestarts,CosineAnnealingLR
import seaborn as sns
import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from utils.logger import Logger
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import faulthandler
from transformers import CLIPProcessor, CLIPModel
import einops
from layers.Transformer_EncDec import EncoderLayer, Encoder, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from models.MultiscaleModel import MS_Model
from models.TimeFrequencyModel import TF_Model
from layers.Embed import PatchEmbedding
from layers.PatchTST_MoE_backbone_cluster import PatchTST_MoE_cluster_patch
from layers.Patch_EncDec import Flatten_Head
from layers.RevIN import RevIN
from models.TimeTextModel import TimeTextModel
from layers.SparseMoE import SparseMoE
from layers.Cluster import EDESC
from layers.PatchTST_backbone import PatchTST_backbone
from layers.Causality_Augmenter import Causal_Aware_Attention
from models.TriModalityModel import TriModalityModel


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 定义Dataset
class get_dataset(Dataset):
    def __init__(self,llm_model_name,data_path, seq_len,pred_length,
                 features, train_split, mode,start_time=None,frequency_minute=1):

        self.mode = mode
        self.llm_model_name = llm_model_name
        self.features = features
        self.seq_len = seq_len
        self.pred_length = pred_length
        self.data_path = data_path

        self.start_time=start_time
        self.frequency_minute = frequency_minute

        # self.data,self.data_stamp,self.embedding = self.get_data()
        self.data,self.embedding=self.get_data()
        train_num = int(len(self.data) * train_split)
        if self.mode == 'train':
            self.data = self.data[:train_num]
        else:
            self.data = self.data[train_num:]

        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :self.seq_len, :], \
                self.embedding[index,:,:],\
                self.data[index, self.seq_len:, :]
                
    
    def get_data(self):
        data=np.load(self.data_path)
        embedding=torch.load(f'/data2/2shared/liubo/CausalMoE/causalmoe/Embeddings/lorenz/p10_t1000_f10/gpt2/train_emb.pt')
        # data_max = np.max(data, axis=0)
        # data_min = np.min(data, axis=0)
 
        # data = (data - data_min) / (data_max - data_min)
        num_sample = len(data) - self.seq_len-self.pred_length  + 1
        seq_data = torch.zeros(num_sample,
                               self.seq_len+self.pred_length,
                               self.features)
        seq_emb=[]
        # time_stamp = torch.zeros(num_sample, 5)
 
        for i in range(num_sample):
            seq_data[i] = torch.tensor(data[i:i + self.seq_len+self.pred_length ,
                                       :])
            # time = calculate_end_time(self.start_time, i, self.frequency_minute)
            # time_stamp[i][0] = int(time[:2]) # MM
            # time_stamp[i][1] = int(time[3:5]) # dd
            # time_stamp[i][2] = int(time[6:10]) # yyyy
            # time_stamp[i][3] = int(time[11:13]) # hh
            # time_stamp[i][4] = int(time[14:16]) # mm
            
        
 
        # return seq_data, time_stamp,embedding
        return seq_data,embedding

# args={
#     'data_path': '/data2/2shared/liubo/CausalMoE/causalmoe/dataset/lorenz/x_10_1000_10.npy',
#     'batch_size': 64,
#     'seq_len': 96,
#     'input_size': 10,
#     'pred_len':1,
#     'ts_dmodel': 64,
#     'dropout': 0.1,
#     'd_llm': 768,
#     'e_layer': 1,
#     'd_layer': 1,
#     'd_ff': 32,
#     'nhead': 2,
#     'lrate': 5e-4,
#     'weight_decay': 1e-6,
#     'device': 'cuda:6',
#     'epochs': 100,
#     'ts_target': 0,
#     'train_split': 0.8,
#     'start_time': '07/01/2025 00:00',
#     'frequency_minute': 1,
#     'batch_size': 64,
#     'train_split': 0.8,
#     'dataset_name': 'lorenz_p10_t1000_f10',
# }




parser = argparse.ArgumentParser(description='TimeVLM')
parser.add_argument('--data_path', type=str, default='/data2/2shared/liubo/CausalMoE/causalmoe/dataset/lorenz/x_10_1000_10.npy', help='data path')

parser.add_argument('--dataset_name', type=str, default='lorenz_p10_t1000_f10', help='dataset name')


parser.add_argument('--vlm_name', type=str, default='openai/clip-vit-base-patch32', help='VLM model type, e.g. CLIP, BLIP2, etc.')
parser.add_argument('--learnable_image', type=str2bool, default=True, help='learnable image')
parser.add_argument('--image_size', type=int, default=56, help='image size for time series to image')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--periodicity', type=int, default=24)
parser.add_argument('--input_size', type=int, default=10, help='input size')
parser.add_argument('--train_split', type=float, default=0.8, help='train set split ratio')

parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')

# vlm parameters
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--patch_len', type=int, default=48, help='patch length')
parser.add_argument('--stride', type=int, default=1, help='stride')
parser.add_argument('--patch_memory_size', type=int, default=50, help='patch memory bank size')
# parser.add_argument('--vlm_hidden_size', type=int, default=512, help='vlm hidden size')
parser.add_argument('--vlm_dmodel', type=int, default=512, help='vlm model dimension')  

# temporal textual retrieval
parser.add_argument('--tllm_llm_dmodel', type=int, default=256, help='temporal llm model dimension')
parser.add_argument('--tllm_head', type=int, default=4, help='temporal llm head')
parser.add_argument('--tllm_ff', type=int, default=512, help='temporal llm feedforward dimension')
parser.add_argument('--tllm_prompt_layer', type=int, default=2, help='temporal llm prompt encoder layers')
parser.add_argument('--tllm_ts_dmodel', type=int, default=256, help='temporal llm time series encoder dimension')
parser.add_argument('--tllm_ts_head', type=int, default=4, help='temporal llm time series encoder head')
parser.add_argument('--tllm_ts_ff', type=int, default=512, help='temporal llm time series encoder feedforward dimension')
parser.add_argument('--tllm_ts_layer', type=int, default=2, help='temporal llm time series encoder layers')


# tempral memeory model
parser.add_argument('--padding', type=int, default=8, help='padding')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')


parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')
parser.add_argument('--use_mem_gate', type=str2bool, default=True, help='use memory gate')
parser.add_argument('--three_channel_image', type=str2bool, default=True, help='use three channel image')
parser.add_argument('--norm_const', type=float, default=0.4)
parser.add_argument('--save_images', type=str2bool, default=True, help='save images')
parser.add_argument('--vlm_max_input_text_length', type=int, default=77, help='vlm max input text length')
parser.add_argument('--device', type=str, default='cuda:7', help='gpu')
args = parser.parse_args(args=[])

# optimizer parameters
args.weight_decay=1e-5
args.batch_size=48
args.learning_rate=1e-3

args.prox_lam=15
args.ridge_lam=1e-2
args.log_frequency = 10
args.train_epochs=200

# multiscale model parameters
args.use_gpu = True if torch.cuda.is_available() else False
args.content='This is a synthetic nonlinear time series dataset.'
args.ms_down_sampling_window=2
args.ms_down_sampling_layers=3
args.ms_channel_independence=1
args.ms_e_layers=2
args.ms_moving_avg=25
args.ms_use_future_temporal_feature=0
args.ms_d_model=96
args.ms_embed='timeF'
args.ms_enc_in=args.input_size
args.ms_use_norm=1
args.ms_down_sampling_method='avg'
args.ms_decomp_method='moving_avg'
args.ms_d_ff=32


# temporal frequency model parameters
args.tf_c_out=args.input_size
args.tf_enc_in=args.input_size
args.tf_d_model=96
args.tf_use_norm=1
args.tf_filter_type='all'
args.tf_quantile=0.9
args.tf_bandwidth=1
args.tf_embedding='fourier_interpolate'
args.tf_top_K_static_freqs=10
args.tf_e_layers=2

# Sparse MoE parameters
args.affine=0
args.subtract_last=0
args.expert_embedding=96
args.patch_num_out=2
args.expert_out_dmodel=96
args.ms_usage=1
args.tf_usage=1
args.llm_usage=0
args.vlm_usage=1

# causal augmenter parameters
args.ca_n_heads=4
args.target=0


dataset_train = get_dataset('gpt2', args.data_path, seq_len=args.seq_len, pred_length=args.pred_len, features=args.input_size, train_split=args.train_split, mode='train')
dataset_test = get_dataset('gpt2', args.data_path, seq_len=args.seq_len, pred_length=args.pred_len, features=args.input_size, train_split=1-args.train_split, mode='test')

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

# set gpu id
# if args.use_gpu and args.use_multi_gpu:
#     args.devices = args.devices.replace(' ', '')
#     device_ids = args.devices.split(',')
#     args.device_ids = [int(id_) for id_ in device_ids]
#     args.gpu = args.device_ids[0]


def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # Adjust padding to make context_len a multiple of periodicity
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # Pad the time series
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # Reshape to [B * nvars, 1, f, p]
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # Resize the time series data
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Convert to 3-channel image
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)  # [B * nvars, 3, H, W]

    # Reshape back to [B, nvars, 3, H, W] and average over nvars
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images


class LearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]
        
        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodic
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D Convolution processing
        x_enc = x_enc.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(self.conv2d_2(x_enc))
        
        # Resize to target image size
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x_enc  # [B, output_channels, H, W]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



class PatchMemoryBank:
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        """
        Initialize the patch memory bank.
        
        Args:
            max_size (int): Maximum number of patches to store.
            patch_size (int): Size of each patch.
            feature_dim (int): Dimensionality of each patch feature.
            device (torch.device): Device to store memory bank on (CPU/GPU).
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)  # [100, d_model]
        self.ptr = 0

    def update(self, new_patches):
        """
        Update the patch memory bank with new patches using circular buffer strategy.
        
        Args:
            new_patches (Tensor): New patches to add to the memory bank.
        """
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]
        
        if self.ptr + n > self.max_size:
            # Wrap around if the memory bank is full
            remaining_space = self.max_size - self.ptr
            self.patches[self.ptr:] = new_patches_flat[:remaining_space]        
            remaining_patches = n - remaining_space
            if remaining_patches >= self.max_size:
                self.patches[:] = new_patches_flat[-self.max_size:]
                self.ptr = 0
            else:
                self.patches[:remaining_patches] = new_patches_flat[remaining_space:]
                self.ptr = remaining_patches
        else:
            self.patches[self.ptr:self.ptr + n] = new_patches_flat
            self.ptr += n

    def retrieve(self, query_patches, top_k=5):
        """
        Retrieve the top-k most similar patches from the memory bank.
        
        Args:
            query_patches (Tensor): Query patches for retrieval.
            top_k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            retrieved_patches (Tensor): Retrieved patches from the memory bank.
            indices (Tensor): Indices of the retrieved patches.
        """
        query_flat = query_patches.mean(dim=1)  # [224, d_model]
        memory_flat = self.patches  # [100, d_model]
        
        similarity = torch.matmul(query_flat, memory_flat.T)  # [224, 100]
        _, indices = similarity.topk(top_k, dim=-1)
        
        retrieved_patches = self.patches[indices]
        return retrieved_patches, indices



class CausalMoe(nn.Module):
    """
    Time-VLM model with image and text modalities for enhanced time series forecasting.
    """
    def __init__(self, config, **kwargs):
        super(CausalMoe, self).__init__()
        self.config = config
        # self.vlm_manager = VLMManager(config)
        # self.device = torch.device('cuda:{}'.format(self.config.device))
        self.device=config.device
        self.use_mem_gate = config.use_mem_gate
        
        self.patch_len=config.patch_len
        self.stride=config.stride
        
        self.patch_num_in = math.ceil((config.seq_len - config.patch_len) / config.stride) + 1
        self.patch_num_out = config.patch_num_out 
        
        self.causal_augmenter=Causal_Aware_Attention(
            FullAttention(False, attention_dropout=config.dropout,output_attention=True),
            n_feature=config.input_size,
            n_heads=config.ca_n_heads
        )
                
        # Initialize patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=config.patch_memory_size,  # e.g., 100 patches
            patch_size=config.patch_len,
            feature_dim=config.vlm_dmodel,
            device=self.device
        )
        
        
        
        # Temporal-Textual retrieval model
        # self._init_temporal_textual_model(config)
        self.time_text_model= TimeTextModel(config)

        # vlm model
        # self._init_vlm_model(config)
        self.trimodal_model=TriModalityModel(config)
        
        # Multi-scale model        
        self.ms_model= MS_Model(config)
        
        # Temporal Frequency Model
        self.tf_model= TF_Model(config)
        
        # temporal model
        self._init_temporal_memory_model(config)
        
        
        experts = []
        if getattr(config, "ms_usage", 1) != 0:
            experts.append(self.ms_model)
        if getattr(config, "tf_usage", 1) != 0:
            experts.append(self.tf_model)
        if getattr(config, "llm_usage", 1) != 0:
            experts.append(self.time_text_model)
        if getattr(config, "vlm_usage", 1) != 0:
            experts.append(self.trimodal_model)

        if len(experts) == 0:
            raise ValueError("No experts enabled. Set at least one of ms_usage/tf_usage/llm_usage/vlm_usage to 1.")

        self.experts_repo = nn.ModuleList(experts)

        
        self.foundation_model=SparseMoE(
            top_k=1,
            Experts_Repo=self.experts_repo,
            d_model=config.expert_out_dmodel,
            causal_augmenter=self.causal_augmenter
        )
        
        # Patch Pattern Learning
        self._init_patch_pattern_learning(config)
        
        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = config.expert_out_dmodel, nhead = config.tllm_ts_head, batch_first=True, norm_first = True, dropout = config.dropout,dim_feedforward=config.tllm_ff)
        self.pred_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = config.tllm_ts_layer)
        
        # Projection       
        self.projection = Flatten_Head(True, config.input_size, config.expert_out_dmodel*self.patch_num_out, config.pred_len,head_dropout=config.dropout)

        self._init_weights()
    
        
    def _init_patch_pattern_learning(self, config):
        
        
        self.revin_layer = RevIN(config.input_size, affine=config.affine, subtract_last=config.subtract_last)
        
        self.patch_PL = PatchTST_MoE_cluster_patch(c_in=config.input_size, c_out=config.input_size, target_window=config.pred_len, d_model=config.expert_embedding,
                                                    patch_len=config.patch_len, stride=config.stride, patch_num_in=self.patch_num_in, patch_num_out=self.patch_num_out, T_num_expert=len(self.experts_repo))
    
    def patch_pattern_learning_forward(self, x_enc):
        
        # x_enc: (B,L,N)
        x = self.revin_layer(x_enc, 'norm')
        x = x.permute(0, 2, 1)  # [B,N,L]
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [B,N,patch_num_in,patch_len]
        x = x.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num_in]
        
        # model
        time_z = self.patch_PL.project(x)  # z: [bs x nvars x patch_len x patch_num_out]
        
        time_z = self.patch_PL.backbone(time_z)  # z: [bs x nvars x d_model x patch_num_out]

        

        time_cluster_input = time_z
        time_cluster_input = torch.reshape(time_cluster_input, (time_cluster_input.shape[0] * time_cluster_input.shape[3], time_cluster_input.shape[1] * time_cluster_input.shape[2]))  # z: [bs * patch_num_out x nvars * d_model]
        s, h = self.patch_PL.cluster(time_cluster_input) # s: [B*patch_num_out,expert_num]

        bs, nvars, d_model, patch_num_out = time_z.shape
        time_z = time_z.permute(0, 3, 1, 2) # [B,patch_num_out,N,d_model]
        time_z = torch.reshape(time_z, (time_z.shape[0], time_z.shape[1], time_z.shape[2] * time_z.shape[3]))   # z: [bs x patch_num_out x nvars * d_model]
        time_z = self.SaprseMoE(time_z, s)                                                                      # z: [bs x patch_num_out x nvars * d_model]
        time_z = torch.reshape(time_z, (time_z.shape[0], time_z.shape[1], nvars, -1))                           # z: [bs x patch_num_out x nvars x d_model]
        time_z = time_z.permute(0, 2, 3, 1)                                                                     # z: [bs x nvars x d_model x patch_num_out]

        # time_z = self.head(time_z)  # z: [bs x nvars x patch_num_out x patch_len]

        return s, h, time_z
     
    def _init_temporal_textual_model(self, config):
        
        self.normalize_layers = Normalize(config.input_size, affine=False).to(self.device)
        # self.length_to_feature = nn.Linear(self.n_feature, self.ts_dmodel).to(self.device)
        self.length_to_feature = nn.Linear(config.seq_len, config.tllm_ts_dmodel).to(self.device)
        
        self.tllm_ts_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=config.dropout,
                                      output_attention=True), config.input_size, config.tllm_ts_dmodel, config.tllm_ts_head),
                    config.tllm_ts_dmodel,
                    config.tllm_ts_ff,
                    dropout=config.dropout,
                    activation="gelu"
                ) for l in range(config.tllm_ts_layer)
            ],
            # conv_layers=[
            #     ConvLayer(self.ts_dmodel) for _ in range(self.e_layer)
            #     ],
            norm_layer=torch.nn.LayerNorm(config.tllm_ts_dmodel)
        )
        
        # Prompt Encoder
        self.tllm_prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = config.tllm_llm_dmodel, nhead = config.tllm_ts_head, batch_first=True, dim_feedforward=config.tllm_ff,
                                                               norm_first = True, dropout = config.dropout)
        self.tllm_prompt_encoder = nn.TransformerEncoder(
            self.tllm_prompt_encoder_layer,
            num_layers=config.tllm_prompt_layer
        )
        self.tllm_ts_retrieval = CrossModal( d_model=config.input_size, n_heads= 1, d_ff=config.tllm_ff,norm_feature=config.input_size, norm='LayerNorm', attn_dropout=config.dropout, 
                                dropout=config.dropout, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False)
        self.tllm_prompt_retrieval = CrossModal( d_model=config.input_size, n_heads= 1, d_ff=config.tllm_ff,norm_feature=config.input_size, norm='LayerNorm', attn_dropout=config.dropout, 
                                dropout=config.dropout, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False)

        # Transformer decoder
        self.tllm_decoder_layer = nn.TransformerDecoderLayer(d_model = config.tllm_ts_dmodel+config.tllm_llm_dmodel, nhead = config.tllm_ts_head, batch_first=True, norm_first = True, dropout = config.dropout,dim_feedforward=config.tllm_ff).to(self.device)
        self.tllm_decoder = nn.TransformerDecoder(self.tllm_decoder_layer, num_layers = config.tllm_ts_layer).to(self.device)
        
        # Projection
        self.tllm_projection = nn.Linear(config.tllm_ts_dmodel+config.tllm_llm_dmodel, config.pred_len, bias=True)
        
        self.tllm_model=nn.ModuleList([self.tllm_ts_encoder,self.tllm_prompt_encoder,self.tllm_ts_retrieval,self.tllm_prompt_retrieval,self.tllm_decoder,self.tllm_projection])
        
        self._init_weights(self.tllm_model)

    
    def tllm_forward(self, x_enc, text_embeddings):
        input_data = x_enc.float()
        embeddings = text_embeddings.float()
        
        # input_data = self.normalize_layers(input_data, 'norm')
        input_data = input_data.permute(0,2,1) # [B, N, L]
        input_data = self.length_to_feature(input_data) # [B, N, d], linear
        
        # Encoder
        enc_out, attn_scores = self.tllm_model[0](input_data) # [B, N, d], transformer encoder
        enc_out = enc_out.permute(0,2,1) # [B, d, N]
        enc_embeddings = self.tllm_model[1](embeddings) # [B, N, E] transformer encoder
        enc_embeddings = enc_embeddings.permute(0,2,1) # [B, E, N]
        
        ts_retrieval = self.tllm_model[2](enc_out, enc_embeddings, enc_embeddings) # Q X KV  [B, d, N]X[B, E, N] =
        prompt_retrieval = self.tllm_model[3](enc_embeddings, enc_out, enc_out) # [B, E, N]X[B, d, N] = [B, E

        cross_out = torch.cat([ts_retrieval, prompt_retrieval], dim=1) # [B, d+E, N]
        cross_out = cross_out.permute(0, 2, 1) # [B, N, d_model+d_llm]

        # Decoder  self-attention, dec_out (B,T,ts_dmodel)  
        dec_out = self.tllm_model[4](cross_out, cross_out,) # [B, N, d]
        # dec_out = dec_out.view(dec_out.shape[0], -1)
        # Projection
        dec_out = self.tllm_model[5](dec_out) # [B,N, S]
        dec_out = dec_out.permute(0,2,1) # [B, S, N]

        # denorm
        # dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    
    def _init_temporal_memory_model(self, config):
        
        self.head_nf = config.vlm_dmodel * (math.ceil((config.seq_len - config.patch_len) / config.stride) + 1)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.vlm_dmodel),
            nn.Dropout(config.dropout)
        )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.vlm_dmodel, config.vlm_dmodel * 2),
            nn.GELU(),
            nn.Linear(config.vlm_dmodel * 2, config.vlm_dmodel)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.vlm_dmodel,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        
        # Memory fusion gate
        if self.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.vlm_dmodel * 2, config.vlm_dmodel),
                nn.GELU(),
                nn.Linear(config.vlm_dmodel, 2),
                nn.Softmax(dim=-1)
            )
            
        self.flatten = nn.Flatten(start_dim=-2)

        self.PL_embedding = PatchEmbedding(
            config.vlm_dmodel, 
            config.patch_len, 
            config.stride, 
        )
    
    def _init_vlm_model(self, config):
        # self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        # model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto")
        self.processor = CLIPProcessor.from_pretrained(config.vlm_name, local_files_only=True)
        self.vlm_model = CLIPModel.from_pretrained(config.vlm_name, output_hidden_states=True, local_files_only=True)
        
        self.PL_embedding = PatchEmbedding(
            config.vlm_dmodel, 
            config.patch_len, 
            config.stride, 
            # config.padding, 
            config.dropout
        )
        self.head_nf = config.vlm_dmodel * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.vlm_dmodel),
            nn.Dropout(config.dropout)
        )
        
        self.multimodal_head = nn.Sequential(
            nn.Linear(config.vlm_dmodel, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Multimodal enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(config.vlm_hidden_size * 2, config.vlm_dmodel),  # Combine vision and text
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Cross-modal attention for feature enhancement
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.vlm_dmodel,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Memory fusion gate
        if self.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.vlm_dmodel * 2, config.vlm_dmodel),
                nn.GELU(),
                nn.Linear(config.vlm_dmodel, 2),
                nn.Softmax(dim=-1)
            )

        # Prediction fusion gate
        self.gate = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Linear(config.pred_len, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.vlm_dmodel, config.vlm_dmodel * 2),
            nn.GELU(),
            nn.Linear(config.vlm_dmodel * 2, config.vlm_dmodel)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.vlm_dmodel,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
        self.layer_norm = nn.LayerNorm(config.vlm_dmodel)

    def _compute_local_memory(self, patches):
        """Compute local memory by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank
        retrieved_patches, _ = self.patch_memory_bank.retrieve(patches, top_k=self.config.top_k)
        
        # Process retrieved patches with local MLP
        local_memory = self.local_memory_mlp(retrieved_patches)
        
        # Average over retrieved patches
        local_memory = local_memory.mean(dim=1, keepdim=True)
        
        # Residual connection with original patches
        local_memory = local_memory + patches
        
        return local_memory

    def _compute_global_memory(self, patches):
        """Compute global memory by aggregating information across all patches"""
        # Self-attention to capture global dependencies
        attn_output, _ = self.memory_attention(
            query=patches,
            key=patches,
            value=patches
        )
        
        # Update patch memory bank with current patches
        self.patch_memory_bank.update(patches.detach())
        
        if self.use_mem_gate:
            return attn_output  # Return full attention output for advanced gating
        else:
            # Return global context for simple gating (original behavior)
            return attn_output.mean(dim=1, keepdim=True)

    
    def temporal_memory_forward(self, x_enc):
        B, L, n_vars = x_enc.shape
        
        # 1. Process temporal features
        patches, _ = self.PL_embedding(x_enc.transpose(1, 2))  # [B * n_vars, n_patches, d_model]
        
        # 2. Compute local and global memory
        local_memory = self._compute_local_memory(patches)  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(patches)  # [B * n_vars, n_patches, d_model] or [B * n_vars, 1, d_model]
        
        # 3. Combine local and global memory
        if self.use_mem_gate:
            # Advanced memory fusion with gating
            combined_features = torch.cat([local_memory, global_memory], dim=-1)  # [B * n_vars, n_patches, d_model*2]
            gate_weights = self.memory_fusion_gate(combined_features)  # [B * n_vars, n_patches, 2]
            
            # Weighted fusion
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory +
                gate_weights[:, :, 1:2] * global_memory
            )  # [B * n_vars, n_patches, d_model]
        else:
            # Simple addition (original behavior)
            memory_features = local_memory + global_memory  # [B * n_vars, n_patches, d_model]

        # 4. Get temporal predictions
        memory_features = self.flatten(memory_features)  # [B * n_vars, head_nf]
        memory_features = self.memory_head(memory_features)  # [B * n_vars, pred_len]
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, pred_len]
        
        return memory_features.permute(0, 2, 1)  # [B, pred_len, n_vars]
    
    def forward_prediction(self, x_enc, vision_embeddings, text_embeddings):
        B, L, n_vars = x_enc.shape
        
        # 1. Process temporal features
        patches, _ = self.PL_embedding(x_enc.transpose(1, 2))  # [B * n_vars, n_patches, d_model]
        
        # 2. Compute local and global memory
        local_memory = self._compute_local_memory(patches)  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(patches)  # [B * n_vars, n_patches, d_model] or [B * n_vars, 1, d_model]
        
        # 3. Combine local and global memory
        if self.use_mem_gate:
            # Advanced memory fusion with gating
            combined_features = torch.cat([local_memory, global_memory], dim=-1)  # [B * n_vars, n_patches, d_model*2]
            gate_weights = self.memory_fusion_gate(combined_features)  # [B * n_vars, n_patches, 2]
            
            # Weighted fusion
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory +
                gate_weights[:, :, 1:2] * global_memory
            )  # [B * n_vars, n_patches, d_model]
        else:
            # Simple addition (original behavior)
            memory_features = local_memory + global_memory  # [B * n_vars, n_patches, d_model]

        # 4. Get temporal predictions
        memory_features = self.flatten(memory_features)  # [B * n_vars, head_nf]
        temporal_features = self.temporal_head(memory_features)  # [B, n_vars, d_model]
        memory_features = self.memory_head(memory_features)  # [B * n_vars, pred_len]
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, pred_len]
        
        # 5. Process multimodal features
        multimodal_features = torch.cat([vision_embeddings, text_embeddings], dim=-1)  # [B, hidden_size * 2]
        multimodal_features = self.multimodal_enhancement(multimodal_features)  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        
        # 6. Cross-modal attention enhancement
        temporal_features = temporal_features / torch.norm(temporal_features, dim=-1, keepdim=True)
        multimodal_features = multimodal_features / torch.norm(multimodal_features, dim=-1, keepdim=True)
        multimodal_features, _ = self.cross_attention(
            query=temporal_features,
            key=multimodal_features,
            value=multimodal_features
        )  # [B, n_vars, d_model]
        
        # 7. Normalize cross attention output
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        multimodal_features = self.multimodal_head(multimodal_features)  # [B, n_vars, pred_len]
        
        # 8. Compute gating weights
        combined_features = torch.cat([memory_features, multimodal_features], dim=-1)  # [B, n_vars, pred_len * 2]
        gate_weights = self.gate(combined_features)  # [B, n_vars, 2]
        
        # 9. Weighted fusion
        fused_features = (
            gate_weights[:, :, 0:1] * memory_features +
            gate_weights[:, :, 1:2] * multimodal_features
        ) # [B, n_vars, pred_len]
        
        # 10. Final fusion
        predictions = self.fusion_layer(
            torch.cat([memory_features, fused_features], dim=-1)
        ) + memory_features  # [B, n_vars, pred_len]
        
        return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    def _process_clip_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
        outputs = self.vlm_model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        return image_features, text_features  # Both shape: [B, hidden_size]
    
    def vlm_forward(self, x_enc):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)  # torch.Size([32, 512, 7])
        prompt_emb=prompt_emb.float().to(self.device)  
        
        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)
        
        # Convert time series data to images and generate text prompts 
        images = self.vision_augmented_learner(x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity)
        prompts = self.text_augmented_learner(x_enc, self.config.content, self.config.pred_len, self.config.seq_len)
        # images: torch.Size([32, 3, 56, 56]) prompts: list of 32 strings
        
        # Process inputs with the VLM
        vision_embeddings, text_embeddings = self._process_clip_inputs(B, images, prompts)
        
        # Main prediction branch  predictions: torch.Size([32, 96, 7])
        predictions = self.forward_prediction(x_enc, vision_embeddings, text_embeddings)
        
        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y
        
    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y
    
    def forward(self, x_enc,prompt_emb, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        
        
        # temporal frequency model self.tf_model输出 [B, pred_len, n_vars]
        # out=self.tf_model(x_enc)
        
        # multiscale model self.ms_model输出 [B, pred_len, n_vars]
        # out=self.ms_model(x_enc)
        
        # temporal memory model 输出 [B, pred_len, n_vars]
        # temporal_memory_forward_out=self.temporal_memory_forward(x_enc)
        
        # patch pattern learning model 输出 [B, n_vars, d_model, pred_len]
        # s, h, patch_pattern_learning_out=self.patch_pattern_learning_forward(x_enc)
        
        
        # x_enc: (B,L,N)
        B, T, N = x_enc.shape
        # x = self.revin_layer(x_enc, 'norm')
        
        # Normalize input
        x, means, stdev = self._normalize_input(x_enc)
        x = x.permute(0, 2, 1)  # [B,N,L]
        
        patch_x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [B,N,patch_num_in,patch_len]
        patch_x = patch_x.permute(0, 1, 3, 2)  # patch_x: [B,N,patch_len,patch_num_in]
        
        
        # model
        patch_x = self.patch_PL.project(patch_x)  # [B,N,patch_len,patch_num_out]
        _,_,_,patch_num_out = patch_x.shape
        PL_embedding = self.patch_PL.backbone(patch_x)  # [B,N,d_model,patch_num_out]

        time_cluster_input = PL_embedding
        time_cluster_input = torch.reshape(time_cluster_input, (time_cluster_input.shape[0] * time_cluster_input.shape[3], time_cluster_input.shape[1] * time_cluster_input.shape[2]))  # z: [bs * patch_num_out, nvars * d_model]
        
        # time_cluster_input: [B*patch_num_out, nvars * d_model]
        affinity = self.patch_PL.cluster(time_cluster_input) # s: [B*patch_num_out,expert_num]

        # PL_embedding: [B*patch_num_out,N,d_model]
        PL_embedding=PL_embedding.reshape(PL_embedding.shape[0]*PL_embedding.shape[3], PL_embedding.shape[1], PL_embedding.shape[2])  
        patch_x=patch_x.reshape(patch_x.shape[0]*patch_x.shape[3],patch_x.shape[1],patch_x.shape[2])  # [B*patch_num_out,N,patch_len]
        
        moe_output = self.foundation_model(patch_x, PL_embedding, affinity)  # [B*patch_num_out,N,expert_out_dmodel]
        moe_output = self.pred_decoder(moe_output,moe_output)  # [B*patch_num_out,N,expert_out_dmodel]
        
        moe_output = moe_output.reshape(B, patch_num_out, N, -1)  # [B,patch_num_out,N,expert_out_dmodel]
        moe_output = moe_output.permute(0, 2, 3, 1)  # [B,N,expert_out_dmodel,patch_num_out]

        predictions = self.projection(moe_output)  # [B,N,pred_len]
        
        # Denormalize output
        predictions = self._denormalize_output(predictions, means, stdev)
        
        return predictions.permute(0, 2, 1)  # [B, pred_len, N]

    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    def text_augmented_learner(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.
        Each variable in the time series will have its own prompt.
        """
        B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

        # Initialize a list to store prompts for each batch
        prompts = []
    
        # Calculate overall statistics for each batch
        for b in range(B):
            # Calculate statistics for the current batch
            min_value = torch.min(x_enc[b]).item()  # Overall minimum value for the batch
            max_value = torch.max(x_enc[b]).item()  # Overall maximum value for the batch
            median_value = torch.median(x_enc[b]).item()  # Overall median value for the batch
            trend = x_enc[b].diff(dim=0).sum().item()  # Overall trend for the batch

            # Determine the overall trend direction
            trend_direction = "upward" if trend > 0 else "downward"
                
            prompt_parts = [
                "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.config.vlm_max_input_text_length] if len(prompt) > self.config.vlm_max_input_text_length else prompt
            
            prompts.append(prompt)  

        return prompts

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.config.save_images:
            self.save_images(images)

        return images
    
    @staticmethod
    def _normalize_images(images):
        """
        Normalize image tensors to [0, 255] as uint8.
        Assumes images are in [0, 1] or need to be scaled.
        
        Args:
        - images (Tensor): Input images with shape [B, C, H, W]
        
        Returns:
        - Tensor: Normalized images as uint8 with shape [B, C, H, W]
        """
        # Compute min and max per image across all channels and spatial dimensions
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        
        return images

    @torch.no_grad()
    def save_images(self, images):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved with shape [B, C, H, W]
        """
        save_dir = "/data2/2shared/liubo/CausalMoE/causalmoe/ts-images/"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            # Move to CPU and convert to numpy
            img_tensor = img_tensor.cpu().numpy()
            
            # Check channel count and handle accordingly
            if img_tensor.shape[0] == 3:
                # RGB image: Convert from [C, H, W] to [H, W, C]
                img_tensor = np.transpose(img_tensor, (1, 2, 0))
                mode = 'RGB'
            elif img_tensor.shape[0] == 1:
                # Grayscale image: Convert from [C, H, W] to [H, W]
                img_tensor = np.squeeze(img_tensor, 0)
                mode = 'L'
            else:
                print(f"Warning: Unexpected number of channels {img_tensor.shape[0]} for image {i}. Skipping...")
                continue
            
            # Ensure data type is uint8
            if img_tensor.dtype != np.uint8:
                img_tensor = img_tensor.astype(np.uint8)
            
            # Create PIL image and save
            try:
                img = Image.fromarray(img_tensor, mode=mode)
                img.save(os.path.join(save_dir, f"image_{i}.png"))
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue


CausalMoE=CausalMoe(args).to(args.device)
model_optim = optim.Adam(CausalMoE.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=args.train_epochs, eta_min=5e-5, verbose=True  # 学习率从初始值逐渐降低到5e-5
        )
criterion = nn.MSELoss()
CausalMoE.train()

def get_qkv_weight(model):
    W_q=model.causal_augmenter.query_projection.weight
    W_k=model.causal_augmenter.key_projection.weight
    W_v=model.causal_augmenter.value_projection.weight
    
    wq_norm=torch.norm(W_q,dim=0, keepdim=True)
    wk_norm=torch.norm(W_k,dim=0, keepdim=True)
    wv_norm=torch.norm(W_v,dim=0, keepdim=True)
    
    return W_q, W_k, W_v, wq_norm, wk_norm, wv_norm

def prox_regularize(model):
    W_q=model.causal_augmenter.query_projection.weight
    norm= torch.norm(W_q, dim=0, keepdim=True)  # l2 norm
    W_q.data = ((W_q / torch.clamp(norm, min=(args.prox_lam * model_optim.param_groups[0]['lr'])))
        * torch.clamp(norm - (model_optim.param_groups[0]['lr'] * args.prox_lam), min=0.0))
    
    W_k=model.causal_augmenter.key_projection.weight
    norm = torch.norm(W_k, dim=0, keepdim=True)
    W_k.data = ((W_k / torch.clamp(norm, min=(args.prox_lam * model_optim.param_groups[0]['lr'])))
        * torch.clamp(norm - (model_optim.param_groups[0]['lr'] * args.prox_lam), min=0.0))
    
    W_v=model.causal_augmenter.value_projection.weight
    norm = torch.norm(W_v, dim=0, keepdim=True)
    W_v.data = ((W_v / torch.clamp(norm, min=(args.prox_lam * model_optim.param_groups[0]['lr'])))
        * torch.clamp(norm - (model_optim.param_groups[0]['lr'] * args.prox_lam), min=0.0))

def ridge_regularize(model):
    '''Apply ridge penalty at linear layer and hidden-hidden weights.'''
        
    ridge = 0
    
    # time_text_model
    if args.llm_usage:
        for layer in model.time_text_model.ts_encoder.layers:
            ridge += torch.sum(layer.self_attn.in_proj_weight ** 2) + \
                            torch.sum(layer.self_attn.out_proj.weight ** 2) + \
                            torch.sum(layer.linear1.weight ** 2) + \
                            torch.sum(layer.linear2.weight ** 2) + \
                            torch.sum(layer.norm1.weight ** 2) + \
                            torch.sum(layer.norm2.weight ** 2) 
                            
        for layer in model.time_text_model.prompt_encoder.layers:
            ridge += torch.sum(layer.self_attn.in_proj_weight ** 2) + \
                            torch.sum(layer.self_attn.out_proj.weight ** 2) + \
                            torch.sum(layer.linear1.weight ** 2) + \
                            torch.sum(layer.linear2.weight ** 2) + \
                            torch.sum(layer.norm1.weight ** 2) + \
                            torch.sum(layer.norm2.weight ** 2)

        for layer in model.time_text_model.ts_retrieval.layers:
            ridge += torch.sum(layer.self_attn.W_Q.weight ** 2) + \
                        torch.sum(layer.self_attn.W_K.weight ** 2) + \
                        torch.sum(layer.self_attn.W_V.weight ** 2) + \
                        torch.sum(layer.self_attn.to_out[0].weight ** 2) + \
                        torch.sum(layer.norm_attn.weight ** 2) + \
                        torch.sum(layer.ff[0].weight ** 2) + \
                        torch.sum(layer.ff[3].weight ** 2) + \
                        torch.sum(layer.norm_ffn.weight ** 2)
        
        for layer in model.time_text_model.prompt_retrieval.layers:
            ridge += torch.sum(layer.self_attn.W_Q.weight ** 2) + \
                        torch.sum(layer.self_attn.W_K.weight ** 2) + \
                        torch.sum(layer.self_attn.W_V.weight ** 2) + \
                        torch.sum(layer.self_attn.to_out[0].weight ** 2) + \
                        torch.sum(layer.norm_attn.weight ** 2) + \
                        torch.sum(layer.ff[0].weight ** 2) + \
                        torch.sum(layer.ff[3].weight ** 2) + \
                        torch.sum(layer.norm_ffn.weight ** 2)

        for layer in model.time_text_model.decoder.layers:
            ridge += torch.sum(layer.self_attn.in_proj_weight ** 2) + \
                        torch.sum(layer.self_attn.out_proj.weight ** 2) + \
                            torch.sum(layer.multihead_attn.in_proj_weight ** 2) + \
                            torch.sum(layer.multihead_attn.out_proj.weight ** 2) + \
                            torch.sum(layer.linear1.weight ** 2) + \
                            torch.sum(layer.linear2.weight ** 2) + \
                            torch.sum(layer.norm1.weight ** 2) + \
                            torch.sum(layer.norm2.weight ** 2) + \
                            torch.sum(layer.norm3.weight ** 2) 
            
        ridge += torch.sum(model.time_text_model.projection.weight ** 2)
    
    
    # trimodal_model
    if args.vlm_usage:
        ridge+= torch.sum(model.trimodal_model.temporal_head[0].weight ** 2) +\
                torch.sum(model.trimodal_model.memory_head[0].weight ** 2) +\
                torch.sum(model.trimodal_model.multimodal_head[0].weight ** 2)+\
                torch.sum(model.trimodal_model.multimodal_head[1].weight ** 2)+\
                torch.sum(model.trimodal_model.multimodal_enhancement[0].weight ** 2)+\
                torch.sum(model.trimodal_model.memory_fusion_gate[0].weight ** 2)+\
                torch.sum(model.trimodal_model.memory_fusion_gate[2].weight ** 2)+\
                torch.sum(model.trimodal_model.local_memory_mlp[0].weight ** 2)+\
                torch.sum(model.trimodal_model.local_memory_mlp[2].weight ** 2)+\
                torch.sum(model.trimodal_model.memory_attention.in_proj_weight ** 2)+\
                torch.sum(model.trimodal_model.memory_attention.out_proj.weight ** 2)+\
                torch.sum(model.trimodal_model.layer_norm.weight ** 2)
        
        ridge+= torch.sum(model.memory_head[0].weight ** 2)+\
                    torch.sum(model.temporal_head[0].weight ** 2)+\
                    torch.sum(model.local_memory_mlp[0].weight ** 2)+\
                    torch.sum(model.local_memory_mlp[2].weight ** 2)+\
                    torch.sum(model.memory_attention.in_proj_weight ** 2)+\
                    torch.sum(model.memory_attention.out_proj.weight ** 2)+\
                    torch.sum(model.PL_embedding.value_embedding.weight ** 2)+\
                    torch.sum(model.patch_PL.backbone.W_P.weight ** 2)
        
        for layer in model.patch_PL.backbone.encoder.layers:
            ridge += torch.sum(layer.self_attn.W_Q.weight ** 2) + \
                        torch.sum(layer.self_attn.W_K.weight ** 2) + \
                        torch.sum(layer.self_attn.W_V.weight ** 2) + \
                        torch.sum(layer.self_attn.to_out[0].weight ** 2)+\
                        torch.sum(layer.norm_attn[1].weight ** 2)+\
                        torch.sum(layer.ff[0].weight ** 2)+\
                        torch.sum(layer.ff[3].weight ** 2)+\
                        torch.sum(layer.norm_ffn[1].weight ** 2)
    
    # ms_model
    if args.ms_usage:
        for layer in model.ms_model.pdm_blocks:
            ridge += torch.sum(layer.layer_norm.weight ** 2) 
            for down_sample in layer.mixing_multi_scale_season.down_sampling_layers:
                ridge += torch.sum(down_sample[0].weight ** 2) + \
                            torch.sum(down_sample[2].weight ** 2)
            for up_sample in layer.mixing_multi_scale_trend.up_sampling_layers:
                ridge += torch.sum(up_sample[0].weight ** 2) + \
                            torch.sum(up_sample[2].weight ** 2)
            ridge+= torch.sum(layer.out_cross_layer[0].weight ** 2)+\
                    torch.sum(layer.out_cross_layer[2].weight ** 2)
        
        for layer in model.ms_model.predict_layers:
            ridge += torch.sum(layer[0].weight ** 2) + \
                        torch.sum(layer[2].weight ** 2)
        
        ridge+= torch.sum(model.ms_model.projection_layer.weight ** 2)
    
    
    # tf_model
    # for m in model.tf_model.model:
    #     ridge+= torch.sum(m.static_filter.weight ** 2)+\
    #             torch.sum(m.dynamic_cross_filter.weight ** 2)
                
    # ridge+= torch.sum(model.tf_model.layer_norm.layer_norm_real.weight ** 2)+\
    #     torch.sum(model.tf_model.projection.linear_real.weight ** 2)+\
    #         torch.sum(model.tf_model.projection.linear_imag.weight ** 2)+\
    #             torch.sum(model.tf_model.projection.linear_out.weight ** 2)
    
    if args.tf_usage:
        for m in model.tf_model.model:
            # static_filter may be PredefinedStaticFilter (no parameters)
            if hasattr(m, "static_filter"):
                for p in m.static_filter.parameters(recurse=True):
                    ridge += torch.sum((p.abs() ** 2) if torch.is_complex(p) else (p ** 2))


            # dynamic_cross_filter is usually learnable (but still guard it)
            if hasattr(m, "dynamic_cross_filter"):
                for p in m.dynamic_cross_filter.parameters(recurse=True):
                    ridge += torch.sum((p.abs() ** 2) if torch.is_complex(p) else (p ** 2))

        # Keep the rest if these modules exist in your TF implementation
        if hasattr(model.tf_model, "layer_norm") and hasattr(model.tf_model.layer_norm, "layer_norm_real"):
            ridge += torch.sum(model.tf_model.layer_norm.layer_norm_real.weight ** 2)

        if hasattr(model.tf_model, "projection"):
            if hasattr(model.tf_model.projection, "linear_real"):
                ridge += torch.sum(model.tf_model.projection.linear_real.weight ** 2)
            if hasattr(model.tf_model.projection, "linear_imag"):
                ridge += torch.sum(model.tf_model.projection.linear_imag.weight ** 2)
            if hasattr(model.tf_model.projection, "linear_out"):
                ridge += torch.sum(model.tf_model.projection.linear_out.weight ** 2)

    
    #       
    for layer in model.pred_decoder.layers:
        ridge += torch.sum(layer.self_attn.in_proj_weight ** 2) + \
                torch.sum(layer.self_attn.out_proj.weight ** 2) + \
                torch.sum(layer.multihead_attn.in_proj_weight ** 2) + \
                torch.sum(layer.multihead_attn.out_proj.weight ** 2) + \
                torch.sum(layer.linear1.weight ** 2) + \
                torch.sum(layer.linear2.weight ** 2) + \
                torch.sum(layer.norm1.weight ** 2) + \
                torch.sum(layer.norm2.weight ** 2) + \
                torch.sum(layer.norm3.weight ** 2)
    
    for layer in model.projection.linears:
        ridge += torch.sum(layer[0].weight ** 2)+\
                    torch.sum(layer[2].weight ** 2)            
    
    return args.ridge_lam * ridge

log_name=f'Expert:time_ms{args.ms_usage}_tf{args.tf_usage}_llm{args.llm_usage}_vlm{args.vlm_usage}_proxlam{args.prox_lam}_ridgelam{args.ridge_lam}_lr{args.learning_rate}_bs{args.batch_size}'
gc_logger = Logger(log_name, args.dataset_name, args.target)

save_path=f'/data2/2shared/liubo/CausalMoE/causalmoe/logs/{args.dataset_name}/target{args.target}/{log_name}'
model_save_path=save_path+'/best_model.pth'

best_train_loss = float('inf')
best_val_loss = float('inf')
early_stop_patience=50

# for name, param in CausalMoE.named_parameters():
#     print(f"Parameter name: {name}, Shape: {param.shape}")

print(f"Training started with parameters: {args}")
with tqdm(initial=0, total=args.train_epochs) as pbar:
    for epoch in range(args.train_epochs):
        # epoch_loss = 0
        train_loss = []
        total_loss=[]
        for i, (batch_x, prompt_emb, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(CausalMoE.device).float() # batch_x: (B,L,N)
            batch_y = batch_y.to(CausalMoE.device).float() # batch_y: (B,S,N)
            train_embedding = torch.Tensor(prompt_emb).to(CausalMoE.device) # (B,N,E)
            
            model_optim.zero_grad()
            outputs = CausalMoE(batch_x, train_embedding)
            l2_loss = criterion(outputs[:,:,args.target].unsqueeze(2), batch_y[:,:,args.target].unsqueeze(2))
            
            loss = l2_loss + ridge_regularize(CausalMoE)
            loss.backward()
            model_optim.step()
            
            # epoch_loss += loss.item()
            train_loss.append(l2_loss.item())
            total_loss.append(loss.item())
        
        mtrain_loss = np.mean(train_loss)
        mtotal_loss= np.mean(total_loss)
        scheduler.step()
        prox_regularize(CausalMoE)
        
        pbar.set_description(f'Epoch:{epoch}|L_train: {mtrain_loss:.4f}|L_total:{mtotal_loss:.4f}  |lr:{scheduler.get_last_lr()[0]:.6f}')
        pbar.update(1)

        _,_,_,wq_norm,wk_norm,wv_norm = get_qkv_weight(CausalMoE)
        if (epoch+1) % args.log_frequency == 0:
            
            info = f'{args.dataset_name} Train: {args.target}'
            info = info + ': Epoch {}/{}'.format(epoch, args.train_epochs)
            info += '| Train Loss: {:.6f}'.format(mtrain_loss)
            info += '| Total Loss: {:.6f}'.format(mtotal_loss)
            info += '\n'
            info += '| Learning Rate: {:.6f}'.format(model_optim.param_groups[0]['lr'])
            info += '| Prox lambda: {:.4f}'.format(args.prox_lam)
            info += '| Ridge lambda: {:.4f}'.format(args.ridge_lam)
            info += '| Batch size: {:.4f}'.format(args.batch_size)
            
            info += f'| Wq_norm: {wq_norm}'
            info += f'| Wk_norm: {wk_norm}'
            info += f'| Wv_norm: {wv_norm}'
            
            gc_logger.log_info(info)
            
        if (epoch+1) % 10 == 0:
            print(f'Epoch:{epoch}|train loss: {mtrain_loss:.4f}|lr:{model_optim.param_groups[0]["lr"]:.6f}')
            print(f'W_q norm: {wq_norm}')
            print(f'W_k norm: {wk_norm}')
            print(f'W_v norm: {wv_norm}')
            
        if mtrain_loss < best_train_loss:
            # best_val_loss = mvalid_loss
            best_train_loss = mtrain_loss
            best_epoch = epoch
            early_stop_counter = 0
            # 保存最佳模型
            torch.save({
                'lr': args.learning_rate,
                'batch_size': args.batch_size,  # Corrected to use args['batch_size']
                'seq_len': args.seq_len,
                'patch_len': args.patch_len,
                'target_length': args.pred_len,
                'epoch': epoch,
                'prox_lam': args.prox_lam,
                'ridge_lam': args.ridge_lam,
                'wq_norm': wq_norm,
                'wk_norm': wk_norm,
                'wv_norm': wv_norm,
                'model_state_dict': CausalMoE.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
            }, model_save_path)
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}, best val loss: {best_train_loss:.6f} at epoch {best_epoch+1}")
            stop_epoch=epoch
            break
        
        
        