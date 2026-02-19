import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_Freq_FourierInterpolate, DataEmbedding_FreqInterpolate, DataEmbedding_FreqComplex
from layers.FilterLayer import FrequencyDomainFilterLayer
from layers.Complex_Func import ComplexLayerNorm
from layers.Causality_Augmenter import Causal_Aware_Attention
from layers.SelfAttention_Family import FullAttention

class ComplexProjection(nn.Module):
    def __init__(self, d_model, freq_len):
        super(ComplexProjection, self).__init__()
        self.linear_real = nn.Linear(d_model, d_model)
        self.linear_imag = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model * 2, freq_len)

    def forward(self, x):
        real_part = self.linear_real(x.real) - self.linear_imag(x.imag)
        imag_part = self.linear_imag(x.real) + self.linear_real(x.imag)
        x = torch.cat((real_part, imag_part), dim=-1)
        x = self.linear_out(x)
        return x

class TF_Model(nn.Module):
    def __init__(self, configs):
        super(TF_Model, self).__init__()
        self.model_name = 'TF_Model'
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_out = configs.tf_c_out
        self.c_in = configs.tf_enc_in
        self.d_model = configs.tf_d_model
        self.use_norm = configs.tf_use_norm
        self.filter_type = configs.tf_filter_type
        self.quantile = configs.tf_quantile
        self.bandwidth = configs.tf_bandwidth
        self.embedding = configs.tf_embedding # "fourier_interpolate" or "interpolate" or "complex"
        self.top_K_static_freqs = configs.tf_top_K_static_freqs
        self.model = nn.ModuleList([FrequencyDomainFilterLayer(
            self.seq_len, self.d_model, self.c_in,
            filter_type=configs.tf_filter_type,
            bandwidth=self.bandwidth,
            top_K_static_freqs=self.top_K_static_freqs,
            quantile=self.quantile)
            for _ in range(configs.tf_e_layers)])

        if self.embedding == "fourier_interpolate":
            self.enc_embedding = DataEmbedding_Freq_FourierInterpolate(self.seq_len, self.d_model, self.c_in)
        elif self.embedding == "interpolate":
            self.enc_embedding = DataEmbedding_FreqInterpolate(self.seq_len, self.d_model)
        else:
            self.enc_embedding = DataEmbedding_FreqComplex(self.seq_len, self.d_model)
        self.layer = configs.tf_e_layers
        self.layer_norm = ComplexLayerNorm(self.d_model)
        self.projection = ComplexProjection(self.d_model, configs.expert_out_dmodel)
        
        # self.causal_augmenter=Causal_Aware_Attention(
        #     FullAttention(False, attention_dropout=configs.dropout,output_attention=True),
        #     n_feature=configs.input_size,
        #     n_heads=configs.ca_n_heads
        # )
        
        self.causal_augmenter=None
        
    def forward(self, x_enc,causal_augmenter):
        # Normalization from Non-stationary Transformer.
        # if self.use_norm:
        #     means = x_enc.mean(1, keepdim=True).detach()
        #     x_enc = x_enc - means
        #     stdev = torch.sqrt(
        #         torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x_enc /= stdev
        self.causal_augmenter=causal_augmenter
        enc_emb=self.enc_embedding(x_enc, None)
        enc_emb=self.causal_augmenter(enc_emb)
        enc_in = self.layer_norm(enc_emb) #enc_in: [B, N, d_model]
        for i in range(self.layer):
            enc_out = self.model[i](enc_in)
            enc_out = self.layer_norm(enc_out)
        #out (B, N, d_model)
        dec_in = enc_out + enc_in
        dec_out = self.projection(dec_in)
        # dec_out = dec_out.transpose(2,1)[:, :, :self.c_out]

        # De-Normalization from Non-stationary Transformer
        # if self.use_norm:
        #     dec_out = dec_out * \
        #               (stdev[:, 0, :].unsqueeze(1).repeat(
        #                   1, self.pred_len, 1))
        #     dec_out = dec_out + \
        #               (means[:, 0, :].unsqueeze(1).repeat(
        #                   1, self.pred_len, 1))

        # return dec_out[:, -self.pred_len:, :]  # dec_out: torch.Size([32, 96, 7])
        return dec_out

    def set_static_freqs_idx(self, static_freqs_idx):
        for i in range(self.layer):
            try:
                self.model[i].static_filter.set_static_freqs_idx(static_freqs_idx)
            except:
                pass