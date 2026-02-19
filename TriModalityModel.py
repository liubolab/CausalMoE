import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from layers.Embed import PatchEmbedding
import torch.nn as nn
import torch
from layers.Causality_Augmenter import Causal_Aware_Attention
from layers.SelfAttention_Family import FullAttention
import einops


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


class TriModalityModel(nn.Module):
    def __init__(self, configs):
        super(TriModalityModel, self).__init__()
        
        self.model_name = 'TriModalityModel'
        self.processor = CLIPProcessor.from_pretrained(configs.vlm_name, local_files_only=True)
        self.vlm_model = CLIPModel.from_pretrained(configs.vlm_name, output_hidden_states=True, local_files_only=True)
        self.device=configs.device
        self.use_mem_gate=True
        
        # Initialize patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=configs.patch_memory_size,  # e.g., 100 patches
            patch_size=configs.patch_len,
            feature_dim=configs.vlm_dmodel,
            device=self.device
        )
        
        self.PL_embedding = PatchEmbedding(
            configs.vlm_dmodel,
            configs.patch_len,
            configs.stride,
            # configs.padding,
            configs.dropout
        )
        # self.head_nf = configs.vlm_dmodel * int((configs.seq_len - configs.patch_len) / configs.stride + 2)
        # self.head_nf=vlm_dmodel
        self.flatten = nn.Flatten(start_dim=-2)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(configs.vlm_dmodel, configs.expert_out_dmodel),
            nn.Dropout(configs.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(configs.vlm_dmodel, configs.vlm_dmodel),
            nn.Dropout(configs.dropout)
        )
        
        self.multimodal_head = nn.Sequential(
            nn.Linear(configs.vlm_dmodel, configs.expert_out_dmodel),
            nn.LayerNorm(configs.expert_out_dmodel),
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )
        
        # Multimodal enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(configs.vlm_dmodel * 2, configs.vlm_dmodel),  # Combine vision and text
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )
        
        # Cross-modal attention for feature enhancement
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=configs.vlm_dmodel,
            num_heads=4,
            dropout=configs.dropout,
            batch_first=True
        )
        
        # Memory fusion gate
        self.memory_fusion_gate = nn.Sequential(
            nn.Linear(configs.vlm_dmodel * 2, configs.vlm_dmodel),
            nn.GELU(),
            nn.Linear(configs.vlm_dmodel, 2),
            nn.Softmax(dim=-1)
        )

        # Prediction fusion gate
        self.gate = nn.Sequential(
            nn.Linear(configs.expert_out_dmodel * 2, configs.expert_out_dmodel),
            nn.GELU(),
            nn.Linear(configs.expert_out_dmodel, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(configs.expert_out_dmodel * 2, configs.expert_out_dmodel),
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(configs.vlm_dmodel, configs.vlm_dmodel * 2),
            nn.GELU(),
            nn.Linear(configs.vlm_dmodel * 2, configs.vlm_dmodel)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=configs.vlm_dmodel,
            num_heads=4,
            dropout=configs.dropout,
            batch_first=True
        )
        
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if configs.three_channel_image else 1,
            image_size=configs.image_size, 
            periodicity=configs.periodicity
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
        self.layer_norm = nn.LayerNorm(configs.vlm_dmodel)

        self.image_size= configs.image_size
        self.seq_len= configs.seq_len
        self.periodicity= configs.periodicity
        self.content= configs.content
        self.expert_out_dmodel= configs.expert_out_dmodel

        # self.causal_augmenter=Causal_Aware_Attention(
        #     FullAttention(False, attention_dropout=configs.dropout,output_attention=True),
        #     n_feature=configs.input_size,
        #     n_heads=configs.ca_n_heads
        # )
        
        self.causal_augmenter=None
        
        self.learnable_image=True
        self.save_images=False
        
        self.vlm_max_input_text_length=configs.vlm_max_input_text_length
        
    
    def _compute_local_memory(self, patches):
        """Compute local memory by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank
        retrieved_patches, _ = self.patch_memory_bank.retrieve(patches)
        
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
    
    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        if self.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = self.time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.save_images:
            self.save_images(images)

        return images
    
    def text_augmented_learner(self, x_enc, description, expert_out_dmodel, seq_len, top_k=5):
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
                f"Task: Forecast the next {expert_out_dmodel} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_max_input_text_length] if len(prompt) > self.vlm_max_input_text_length else prompt
            
            prompts.append(prompt)  

        return prompts
    
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
        memory_features = self.memory_head(memory_features)  # [B * n_vars, expert_out_dmodel]
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, expert_out_dmodel]
        
        temporal_features=self.causal_augmenter(temporal_features)
        memory_features=self.causal_augmenter(memory_features)
        
        # 5. Process multimodal features
        multimodal_features = torch.cat([vision_embeddings, text_embeddings], dim=-1)  # [B, hidden_size * 2]
        multimodal_features = self.multimodal_enhancement(multimodal_features)  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        
        multimodal_features=self.causal_augmenter(multimodal_features)
        
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
        multimodal_features = self.multimodal_head(multimodal_features)  # [B, n_vars, expert_out_dmodel]
        
        # 8. Compute gating weights
        combined_features = torch.cat([memory_features, multimodal_features], dim=-1)  # [B, n_vars, expert_out_dmodel * 2]
        gate_weights = self.gate(combined_features)  # [B, n_vars, 2]
        
        # 9. Weighted fusion
        fused_features = (
            gate_weights[:, :, 0:1] * memory_features +
            gate_weights[:, :, 1:2] * multimodal_features
        ) # [B, n_vars, expert_out_dmodel]
        
        # 10. Final fusion
        predictions = self.fusion_layer(
            torch.cat([memory_features, fused_features], dim=-1)
        ) + memory_features  # [B, n_vars, expert_out_dmodel]
        
        return predictions  # [B, n_vars, expert_out_dmodel]
    
    def forward(self, x_enc, causal_augmenter):  # x_enc: [B, N, T]
        
        self.causal_augmenter=causal_augmenter
        
        x_enc = x_enc.permute(0, 2, 1)  # [B, T, N]
        B, L, D = x_enc.shape  
        
        # Convert time series data to images and generate text prompts 
        # images:[B,3,H,W], prompts: list of length B
        images = self.vision_augmented_learner(x_enc, self.image_size, self.seq_len, self.periodicity)
        prompts = self.text_augmented_learner(x_enc, self.content, self.expert_out_dmodel, self.seq_len)
        
        # Process inputs with the VLM, [B, hidden_size]
        vision_embeddings, text_embeddings = self._process_clip_inputs(B, images, prompts)
        
        # Main prediction branch  predictions: torch.Size([32, 96, 7])
        output = self.forward_prediction(x_enc, vision_embeddings, text_embeddings)
        
        return output  # [B, N, expert_out_dmodel]
        
        
    def _process_clip_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
        outputs = self.vlm_model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        return image_features, text_features  # Both shape: [B, hidden_size]

    
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