# The code is based on the PyTorch implementation:
# https://zhuanlan.zhihu.com/p/701777558
# https://github.com/mst272/LLM-Dojo/blob/main/llm_tricks/moe/make_moe_step_by_step.ipynb


import torch
from torch import nn
import torch.nn.functional as F

#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        self.dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class NoisyTopkRouter_Cluster(nn.Module):
    def __init__(self, top_k):
        super(NoisyTopkRouter_Cluster, self).__init__()
        self.top_k = top_k

    def forward(self, logits):
        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, top_k, Experts_Repo, d_model, causal_augmenter):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter_Cluster(top_k)
        # self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.Experts_Repo = Experts_Repo
        self.expert_out_dmodel = d_model
        self.causal_augmenter = causal_augmenter
        
    def forward(self, patch_x, patch_embedding, affinity):  
        # patch_x: [B*patch_num_out,N,patch_len], patch_embedding: [B*patch_num_out, N, d_model], affinity: [B*patch_num_out, num_experts]
        
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(affinity)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros((
            patch_embedding.shape[0], patch_embedding.shape[1], self.expert_out_dmodel),
            device=patch_embedding.device,
            dtype=patch_embedding.dtype
            )  ## 

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        # flat_x = x.reshape(-1, x.size(-1)) # [B*patch_num_out, N*d_model]

        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # [B*patch_num_out, num_experts]


        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.Experts_Repo):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)  # expert_mask: [B*patch_num_out]
            # 5. 展平操作
            # flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if expert_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input_embedding = patch_embedding[expert_mask]  #expert_input: [token,N,d_model]
                expert_input_time_series= patch_x[expert_mask]  #expert_input: [token,N,patch_len]
                # token_num,N,_=expert_input_embedding.shape
                
                # # 7. 将token输入expert得到输出
                # if expert.model_name=='MS_Model':
                #     expert_output = expert(expert_input_time_series).reshape(token_num,-1)  #expert_output: [token,N*export的输出维度]
                # else:
                #     expert_output = expert(expert_input_embedding) #expert_output: [token,export的输出维度]

                
                # # 8. 计算当前专家对于有作用的token的权重分数
                # gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)  # [token,1]
                # # 9. 将expert输出乘上权重分数
                # weighted_output = expert_output * gating_scores
                # # weighted_output=weighted_output.view(token_num,N,d_model)

                # # 10. 循环进行做种的结果叠加
                # flat_final_output = final_output.reshape(final_output.shape[0], -1)  # [B*N*patch_num_out, N*d_model]
                # flat_final_output[expert_mask] += weighted_output.squeeze(1)
                # final_output = flat_final_output.reshape(x.shape[0], x.shape[1], x.shape[2])
                # print(final_output.shape)

                if expert.model_name=='TimeTextModel':
                    expert_output = expert(expert_input_time_series,self.causal_augmenter)    # 期望: [token_i, N, out_d]
                else:
                    expert_output = expert(expert_input_time_series,self.causal_augmenter)    # 期望: [token_i, N, out_d]
                
                # if getattr(expert, "model_name", None) == "MS_Model" or getattr(expert, "model_name", None) == "TF_Model":
                #     expert_output,expert_causal_augmenter = expert(expert_input_time_series,self.causal_augmenter)    # 期望: [token_i, N, out_d]
                # else:
                #     expert_output = expert(expert_input_embedding)      # 期望: [token_i, N, out_d]

                # gating_output: [tokens, num_experts] -> [token_i,1,1] 用于广播到 [token_i,N,out_d]
                gating_scores = gating_output[expert_mask, i].view(-1, 1, 1)

                final_output[expert_mask] += expert_output * gating_scores
                
        return final_output
