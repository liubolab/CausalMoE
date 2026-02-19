import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Causal_Aware_Attention(nn.Module):
    def __init__(self, attention, n_feature, n_heads):
        super(Causal_Aware_Attention, self).__init__()
        self.n_heads=n_heads
              
        self.inner_attention = attention
        
        # 使用shared weight，ts和prompt用同一组weights提取feature
        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.query_projection=nn.Linear(n_feature,n_feature)
        # self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection=nn.Linear(n_feature,n_feature)
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.value_projection=nn.Linear(n_feature,n_feature)

        
        self.n_heads = n_heads
        
    def forward(self, emb): # emb: [B, N, d]
        B, N, _ = emb.shape
        H = self.n_heads
        emb = emb.permute(0, 2, 1)  # [B, d, N]
        queries = self.query_projection(emb).view(B, H, -1, N)
        keys = self.key_projection(emb).view(B, H, -1, N)
        values = self.value_projection(emb).view(B, H, -1, N)
        emb, attn = self.inner_attention(
            queries,
            keys,
            values,
            None
        )
        emb = emb.view(B, -1, N)
        return emb.permute(0, 2, 1)  # [B, N, d]

    def forward1(self, ts, prompt_emb=None, attn_mask=None):
        
        
        
        B, _, N = ts.shape  # [B, L, N]
        H = self.n_heads
        ts = ts.permute(0, 2, 1)  # [B, N, L]
        ts_emb = self.length_to_feature(ts) # [B, N, d_model]
        ts_emb = ts_emb.permute(0, 2, 1)  # [B, d_model, N]
        ts_queries = self.query_projection(ts_emb).view(B, H, -1, N)
        ts_keys = self.key_projection(ts_emb).view(B, H, -1, N)
        ts_values = self.value_projection(ts_emb).view(B, H, -1, N)

        ts_emb, ts_attn = self.inner_attention(
            ts_queries,
            ts_keys,
            ts_values,
            attn_mask
        )
        ts_emb = ts_emb.view(B, -1, N)
        
        prompt_emb=self.llm_proj(prompt_emb).permute(0, 2, 1)
        prompt_queries = self.query_projection(prompt_emb).view(B, H, -1, N)
        prompt_keys = self.key_projection(prompt_emb).view(B, H, -1, N)
        prompt_values = self.value_projection(prompt_emb).view(B, H, -1, N)
        
        prompt_emb, promp_attn = self.inner_attention(
            prompt_queries,
            prompt_keys,
            prompt_values,
            attn_mask
        )
        prompt_emb=prompt_emb.view(B,-1,N)
        
        return ts_emb.permute(0,2,1), prompt_emb.permute(0,2,1)
