import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos,DataEmbedding_wo_temp
from layers.StandardNorm import Normalize
from layers.Causality_Augmenter import Causal_Aware_Attention
from layers.SelfAttention_Family import FullAttention

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.patch_len // (configs.ms_down_sampling_window ** i),
                        configs.patch_len // (configs.ms_down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.patch_len // (configs.ms_down_sampling_window ** (i + 1)),
                        configs.patch_len // (configs.ms_down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.ms_down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high)

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.patch_len // (configs.ms_down_sampling_window ** (i + 1)),
                        configs.patch_len // (configs.ms_down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.patch_len // (configs.ms_down_sampling_window ** i),
                        configs.patch_len // (configs.ms_down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.ms_down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.patch_len = configs.patch_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.ms_down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.ms_d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.ms_channel_independence

        if configs.ms_decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.ms_moving_avg,time_dim=2)
        elif configs.ms_decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.ms_top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.ms_channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.ms_d_model, out_features=configs.ms_d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.ms_d_ff, out_features=configs.ms_d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.patch_len, out_features=configs.ms_d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.ms_d_ff, out_features=configs.ms_d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:  # x:[B,N,T]
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season)
            trend_list.append(trend)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            # if self.channel_independence:
            #     out = ori + self.out_cross_layer(out)
            # out_list.append(out[:, :length, :])
            out_list.append(out)
        return out_list


class MS_Model(nn.Module):

    def __init__(self, configs):
        super(MS_Model, self).__init__()
        
        self.model_name='MS_Model'
        
        self.causal_augmenter=None
        
        self.configs = configs
        self.task_name = 'long_term_forecast'
        self.patch_len = configs.patch_len
        # self.label_len = configs.ms_label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.ms_down_sampling_window
        self.channel_independence = configs.ms_channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.ms_e_layers)])

        self.preprocess = series_decomp(configs.ms_moving_avg)
        # self.enc_in = configs.ms_enc_in
        self.use_future_temporal_feature = configs.ms_use_future_temporal_feature

        if self.channel_independence == 1:
            # self.enc_embedding = DataEmbedding_wo_temp(1, configs.ms_d_model, configs.ms_embed,
            #                                           configs.dropout)
            self.enc_embedding = DataEmbedding_wo_temp(configs.ms_enc_in, configs.ms_d_model, configs.ms_embed, 
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_temp(configs.ms_enc_in, configs.ms_d_model, configs.ms_embed, 
                                                      configs.dropout)
        
        self.enc_embedding_layers = torch.nn.ModuleList(
            [
                DataEmbedding_wo_temp(
                    configs.patch_len // (configs.ms_down_sampling_window ** i),configs.ms_d_model, configs.ms_embed, configs.dropout
                )
                for i in range(configs.ms_down_sampling_layers + 1) 
            ]
        )
        

        self.layer = configs.ms_e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.ms_enc_in, affine=True, non_norm=True if configs.ms_use_norm == 0 else False)
                for i in range(configs.ms_down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [   
                    nn.Sequential( 
                        nn.Linear(
                            configs.patch_len // (configs.ms_down_sampling_window ** i),
                            configs.ms_d_ff
                        ),
                        nn.GELU(),
                        nn.Linear(in_features=configs.ms_d_ff, out_features=configs.ms_d_model)
                    )
                    for i in range(configs.ms_down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.ms_d_model, configs.expert_out_dmodel, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.ms_d_model, configs.expert_out_dmodel, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.patch_len // (configs.ms_down_sampling_window ** i),
                        configs.patch_len // (configs.ms_down_sampling_window ** i),
                    )
                    for i in range(configs.ms_down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.patch_len // (configs.ms_down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.ms_down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.ms_d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.ms_d_model, configs.ms_c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.ms_d_model * configs.patch_len, configs.ms_num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc=None):
        if self.configs.ms_down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.ms_down_sampling_window, return_indices=False)
        elif self.configs.ms_down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.ms_down_sampling_window)
        elif self.configs.ms_down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.ms_enc_in, out_channels=self.configs.ms_enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.ms_down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.ms_down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.ms_down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.ms_down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc,causal_augmenter):

        # if self.use_future_temporal_feature:
        #     if self.channel_independence == 1:
        #         B, T, N = x_enc.size()
        #         x_mark_dec = x_mark_dec.repeat(N, 1, 1)
        #         self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        #     else:
        #         self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
        
        self.causal_augmenter=causal_augmenter
        # x_enc: [B, N, T]
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc.permute(0, 2, 1), None)

        x_list = []  # [B,T,N] -> [B,N,T]
        for x in x_enc:
            x_list.append(x.permute(0, 2, 1))
        
        # x_mark_list = []
        # if x_mark_enc is not None:
        #     for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
        #         B, T, N = x.size()
        #         x = self.normalize_layers[i](x, 'norm')
        #         if self.channel_independence == 1:
        #             x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        #             x_mark = x_mark.repeat(N, 1, 1)
        #         x_list.append(x)
        #         x_mark_list.append(x_mark)
        # else:
        #     for i, x in zip(range(len(x_enc)), x_enc, ):
        #         B, T, N = x.size()
        #         x = self.normalize_layers[i](x, 'norm')
        #         if self.channel_independence == 1:
        #             x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        #         x_list.append(x)

        # embedding
        # enc_out_list = []
        # x_list = self.pre_enc(x_list)
        # if x_mark_enc is not None:
        #     for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
        #         enc_out = self.enc_embedding(x, x_mark)  
        #         enc_out_list.append(enc_out)
        # else:
        #     for i, x in zip(range(len(x_list[0])), x_list[0]):
        #         # enc_out = self.enc_embedding(x, None)  # [B,N,d_model]
                
        #         enc_out = self.enc_embedding_layers[i](x, None)  # [B,N,d_model]
        #         enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            # enc_out_list = self.pdm_blocks[i](enc_out_list)
            enc_out_list = self.pdm_blocks[i](x_list)  # [B,N,T]

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing( enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out  # [B,N,expert_out_dmodel]

    def future_multi_mixing(self, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out)  # enc_out: [B,N,T], dec_out: [B,N,d_model]
                dec_out = self.causal_augmenter(dec_out) # dec_out: [B,N,d_model]
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out) # dec_out: [B,N,expert_out_dmodel]
                # dec_out = dec_out.reshape(B, self.configs.ms_c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    

    def forward(self, x_enc,causal_augmenter):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc,causal_augmenter)
            return dec_out
        
