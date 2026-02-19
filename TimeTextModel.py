import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import EncoderLayer, Encoder, ConvLayer
from layers.StandardNorm import Normalize
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Cross_Modal_Align import CrossModal
from layers.Causality_Augmenter import Causal_Aware_Attention
from layers.SelfAttention_Family import FullAttention


class TimeTextModel(nn.Module):
    def __init__(self,configs):
        super(TimeTextModel, self).__init__()
        self.model_name = 'TimeTextModel'

        self.n_feature= configs.input_size
        self.ts_dmodel = configs.tllm_ts_dmodel
        # self.cross_dmodel = cross_dmodel
        self.dropout_n= configs.dropout
        self.d_llm = configs.tllm_llm_dmodel
        self.e_layer = configs.tllm_ts_layer
        self.d_layer = configs.tllm_prompt_layer
        self.d_ff = configs.tllm_ff
        self.head = configs.tllm_ts_head
        self.seq_len=configs.seq_len
        
        # self.length_to_feature = nn.Linear(self.n_feature, self.ts_dmodel)
        # self.length_to_feature = nn.Linear(seq_len, tllm_ts_dmodel)
        
        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.ts_dmodel, nhead = self.head, batch_first=True, dim_feedforward=self.d_ff,
                                                           norm_first = True,dropout = self.dropout_n)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer)
        
        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.ts_dmodel, nhead = self.head, batch_first=True, dim_feedforward=self.d_ff,
                                                               norm_first = True,dropout = self.dropout_n)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer)

        self.ts_retrieval = CrossModal( d_model=self.n_feature, n_heads= 1, d_ff=self.d_ff,norm_feature=self.n_feature, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=self.e_layer, store_attn=False)
        self.prompt_retrieval = CrossModal( d_model=self.n_feature, n_heads= 1, d_ff=self.d_ff,norm_feature=self.n_feature, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=self.e_layer, store_attn=False)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.ts_dmodel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n,dim_feedforward=self.d_ff)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer)
        
        # Projection
        self.projection = nn.Linear(configs.tllm_ts_dmodel+configs.tllm_llm_dmodel, configs.expert_out_dmodel, bias=True)
        
        # self.tllm_model=nn.ModuleList([self.tllm_ts_encoder,self.tllm_prompt_encoder,self.tllm_ts_retrieval,self.tllm_prompt_retrieval,self.tllm_decoder,self.tllm_projection])
        
        self._init_weights()
        
        # self.causal_augmenter=Causal_Aware_Attention(
        #     FullAttention(False, attention_dropout=configs.dropout,output_attention=True),
        #     n_feature=configs.input_size,
        #     n_heads=configs.ca_n_heads
        # )
        
        self.causal_augmenter=None
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x_enc, text_embeddings, causal_augmenter):
        
        self.causal_augmenter=causal_augmenter
        
        ts_emb = x_enc.float()
        prompt_emb = text_embeddings.float()
        
        # input_data = self.normalize_layers(input_data, 'norm')
        # input_data = input_data.permute(0,2,1) # [B, N, L]
        # input_data = self.length_to_feature(input_data) # [B, N, d], linear
        
        ts_emb=self.causal_augmenter(ts_emb)
        prompt_emb=self.causal_augmenter(prompt_emb)
        
        # Encoder
        enc_out = self.ts_encoder(ts_emb) # [B, N, d], transformer encoder
        enc_out = enc_out.permute(0,2,1) # [B, d, N]
        
        enc_embeddings = self.prompt_encoder(prompt_emb) # [B, N, E] transformer encoder
        enc_embeddings = enc_embeddings.permute(0,2,1) # [B, E, N]
        
        ts_retrieval = self.ts_retrieval(enc_out, enc_embeddings, enc_embeddings) # Q X KV  [B, d, N]X[B, E, N] =
        prompt_retrieval = self.prompt_retrieval(enc_embeddings, enc_out, enc_out) # [B, E, N]X[B, d, N] = [B, E

        # cross_out = torch.cat([ts_retrieval, prompt_retrieval], dim=1) # [B, d+E, N]
        cross_out=ts_retrieval+prompt_retrieval
        cross_out = cross_out.permute(0, 2, 1) # [B, N, d_model+d_llm]

        # Decoder  self-attention, dec_out (B,T,ts_dmodel)  
        dec_out = self.decoder(cross_out, cross_out) # [B, N, d]
        # dec_out = dec_out.view(dec_out.shape[0], -1)
        # Projection
        dec_out = self.projection(dec_out) # [B,N, expert_emb_dmodel]
        # dec_out = dec_out.permute(0,2,1) # [B, S, N]

        # denorm
        # dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out # [B,N, expert_emb_dmodel]
