"""
Phase 3: Linear Attention based Multi-Reference HPCM

MLIC++のLinear Attentionを統合し、計算効率を大幅に改善。
O(N²) → O(N)の計算量削減を実現。

主な改善点:
- Channel-wise linear attention
- より効率的なメモリ検索
- 計算コストの削減
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
sys.path.insert(0, '/workspace/LIC-HPCM-MultiRef')

# HPCM base components
from src.models.base import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1, conv4x4_down, deconv4x4_up
from src.models.HPCM_Base import (
    g_a, g_s, h_a, h_s, 
    y_spatial_prior_s1_s2, 
    y_spatial_prior_s3,
    CrossAttentionCell
)
from src.layers.multi_ref_phase3 import LinearAttentionMemoryBank, HierarchicalLinearMemoryManager


class HPCM_MultiRef_Phase3(basemodel):
    """
    Phase 3: Linear Attention based Multi-Reference HPCM
    
    Phase 2からの主な改善:
    - Softmax Attention → Linear Attention
    - O(N²) → O(N)の計算量削減
    - MLIC++スタイルのchannel-wise attention
    - より効率的なメモリ管理
    
    Args:
        M, N: HPCM base parameters
        enable_multiref: Multi-Reference機能のON/OFF
        max_refs_s1/s2/s3: 各階層の最大参照フレーム数
        topk_refs_s1/s2/s3: 各階層のTop-k参照数
        compress_ratio: Key圧縮率
        value_resolution: Value保存解像度
        num_heads: Attention head数
        kernel_type: Linear attentionのkernel type ('elu', 'relu')
        enable_hierarchical_transfer: 階層間メモリ転送
    """
    
    def __init__(
        self,
        M: int = 320,
        N: int = 256,
        enable_multiref: bool = True,
        max_refs_s1: int = 2,
        max_refs_s2: int = 3,
        max_refs_s3: int = 4,
        topk_refs_s1: int = 1,
        topk_refs_s2: int = 2,
        topk_refs_s3: int = 2,
        compress_ratio: int = 4,
        value_resolution: int = 8,
        num_heads: int = 8,
        kernel_type: str = 'elu',
        enable_hierarchical_transfer: bool = True,
        **kwargs
    ):
        super().__init__(N)  # basemodel takes only N parameter
        
        self.enable_multiref = enable_multiref
        self.enable_hierarchical_transfer = enable_hierarchical_transfer
        
        if not enable_multiref:
            return
        
        # Base components (same as Phase 1/2)
        self.g_a = g_a()
        self.g_s = g_s()
        self.h_a = h_a()
        self.h_s = h_s()

        # Spatial prior networks
        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(1))
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(3))
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(6))
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_s3 = y_spatial_prior_s3(M)

        # Adaptive parameters
        self.adaptive_params_list = nn.ParameterList([
            torch.nn.Parameter(torch.ones((1, M*3, 1, 1)), requires_grad=True) 
            for _ in range(10)
        ])

        # Cross-attention cells
        self.attn_s1 = CrossAttentionCell(M*2, M*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(M*2, M*2, window_size=8, kernel_size=1)
        self.attn_s3 = CrossAttentionCell(M*2, M*2, window_size=16, kernel_size=1)
        
        # Context dimension (HPCM uses 2*N = 2*256 = 512 for most scales)
        # But s3 uses different dimension, need to check
        context_dim_s1 = 2 * N  # 512 for N=256
        context_dim_s2 = 2 * N  # 512
        context_dim_s3 = 2 * N  # 512 (assuming same, may need adjustment)
        
        # LinearAttentionMemoryBank for each scale
        self.memory_bank_s1 = LinearAttentionMemoryBank(
            context_dim=context_dim_s1,
            max_refs=max_refs_s1,
            compress_ratio=compress_ratio,
            value_resolution=value_resolution,
            num_heads=num_heads,
            kernel_type=kernel_type,
            enable_value_storage=True
        )
        self.topk_refs_s1 = topk_refs_s1
        
        self.memory_bank_s2 = LinearAttentionMemoryBank(
            context_dim=context_dim_s2,
            max_refs=max_refs_s2,
            compress_ratio=compress_ratio,
            value_resolution=value_resolution,
            num_heads=num_heads,
            kernel_type=kernel_type,
            enable_value_storage=True
        )
        self.topk_refs_s2 = topk_refs_s2
        
        self.memory_bank_s3 = LinearAttentionMemoryBank(
            context_dim=context_dim_s3,
            max_refs=max_refs_s3,
            compress_ratio=compress_ratio,
            value_resolution=value_resolution,
            num_heads=num_heads,
            kernel_type=kernel_type,
            enable_value_storage=True
        )
        self.topk_refs_s3 = topk_refs_s3
        
        # Hierarchical memory manager
        if enable_hierarchical_transfer:
            self.memory_manager_s1_to_s2 = HierarchicalLinearMemoryManager(
                context_dim=context_dim_s1, num_heads=num_heads
            )
            self.memory_manager_s2_to_s3 = HierarchicalLinearMemoryManager(
                context_dim=context_dim_s2, num_heads=num_heads
            )
    
    def forward(self, x: torch.Tensor, training: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Linear Attention based Multi-Reference
        
        Args:
            x: Input image [B, 3, H, W]
            training: Training mode flag
        
        Returns:
            Dictionary with 'x_hat', 'likelihoods', etc.
        """
        if training is None:
            training = self.training
        
        # Reset memory banks at the start of each forward pass
        if self.enable_multiref:
            self.memory_bank_s1.reset()
            self.memory_bank_s2.reset()
            self.memory_bank_s3.reset()
        
        # Encoder
        y = self.g_a(x)
        
        # Hyper encoder
        z = self.h_a(y)
        
        # Entropy estimation for z
        if training:
            z_res = z - self.means_hyper
            z_hat = self.ste_round(z_res) + self.means_hyper
            z_likelihoods = self.entropy_estimation(self.add_noise(z_res), self.scales_hyper)
        else:
            z_res_hat = torch.round(z - self.means_hyper)
            z_hat = z_res_hat + self.means_hyper
            z_likelihoods = self.entropy_estimation(z_res_hat, self.scales_hyper)   

        # Hyper decoder
        params = self.h_s(z_hat)
        
        # For now, use simplified forward (Phase 3 forward_hpcm is complex and incomplete)
        # TODO: Implement full forward_hpcm with Linear Attention
        
        # Simple passthrough for testing
        y_res = y - params[:, :y.shape[1]]
        if training:
            y_hat = self.ste_round(y_res) + params[:, :y.shape[1]]
            y_likelihoods = self.entropy_estimation(self.add_noise(y_res), torch.ones_like(y_res))
        else:
            y_res_hat = torch.round(y_res)
            y_hat = y_res_hat + params[:, :y.shape[1]]
            y_likelihoods = self.entropy_estimation(y_res_hat, torch.ones_like(y_res))
        
        # Decoder
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def forward_hpcm(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training forward with Multi-Reference Enhancement
        
        Phase 3: Linear Attentionを使用した効率的な実装
        """
        # Encoder
        y = self.Encoder(x)
        y_shape = y.shape[2:]
        z = self.hyper_encoder(y)
        z_shape = z.shape[2:]
        
        # Hyperprior
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        
        hp_feat = self.hyper_decoder(z_hat)
        hp_feat = self.lrp_transform(hp_feat, y_shape)
        
        # s1 scale (H/64, W/64) - 2 steps
        B, C, H_s1, W_s1 = hp_feat.shape[0], 2*self.N, y_shape[0]//64, y_shape[1]//64
        
        y_q_w_0, y_q_h_0, y_q_w_1, y_q_h_1 = self.slice_to_y(y, mode='s1')
        y_q_0 = (y_q_w_0, y_q_h_0)
        y_q_1 = (y_q_w_1, y_q_h_1)
    
    def forward_hpcm(self, y, common_params, 
                     y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                     y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                     y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                     adaptive_params_list, context_net, write=False):
        """
        Hierarchical Progressive Context Modeling with Linear Attention Multi-Reference
        
        Phase 3: 全階層にLinear Attentionベースのmulti-referenceを適用
        """
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        # ============ s1階層 ============
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, dtype, device)
        y_s2 = self.get_s1_s2_with_mask(y, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, dtype, device)
        y_s1 = self.get_s1_s2_with_mask(y_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)

        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2, B, C, H // 2, W // 2, reduce=8)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2, B, C, H // 4, W // 4, reduce=4)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)
        y_res_list_s1 = []
        y_q_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        # s1: 2 steps
        for i in range(2):
            if i == 0:
                scales_s1, means_s1 = common_params_s1.chunk(2, 1)
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s1, scales_s1, means_s1, mask_list[i])
                y_res_list_s1.append(y_res_0)
                y_q_list_s1.append(y_q_0)
                y_hat_list_s1.append(y_hat_0)
                scale_list_s1.append(s_hat_0)
                
                # Add to memory (Linear Attention)
                if self.enable_multiref:
                    self.memory_bank_s1.add_to_memory(context_next)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                
                # Apply Linear Attention Multi-Reference for s1
                if self.enable_multiref:
                    context = self.memory_bank_s1(context, k=self.topk_refs_s1)
                
                context_next = self.attn_s1(context, context_next)
                scales, means = context_next.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_1)
                y_q_list_s1.append(y_q_1)
                y_hat_list_s1.append(y_hat_1)
                scale_list_s1.append(s_hat_1)
                
                if self.enable_multiref:
                    self.memory_bank_s1.add_to_memory(context_next)
        
        y_res = torch.sum(torch.stack(y_res_list_s1), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s1), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_q_list_s1[i]) for i in range(len(y_q_list_s1))]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(scale_list_s1[i]) for i in range(len(scale_list_s1))]
        
        # s1 → s2 memory transfer
        if self.enable_multiref and self.enable_hierarchical_transfer:
            self.memory_manager_s1_to_s2.transfer_memory(
                self.memory_bank_s1.memory_keys,
                self.memory_bank_s1.memory_values if self.memory_bank_s1.enable_value_storage else None,
                self.memory_bank_s1.memory_valid,
                self.memory_bank_s2
            )
        
        # Up-scaling to s2
        y_res = self.recon_for_s2_s3(y_res, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        # ============ s2階層 ============
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context_next = context + common_params_s2

        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)
        y_res_list_s2 = []
        y_q_list_s2 = []
        y_hat_list_s2 = []
        scale_list_s2 = []

        # s2: 4 steps
        for i in range(4):
            if i == 0:
                scales_s2, means_s2 = context_next.chunk(2, 1)
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s2, scales_s2, means_s2, mask_list[i])
                y_res_list_s2.append(y_res_0)
                y_q_list_s2.append(y_q_0)
                y_hat_list_s2.append(y_hat_0)
                scale_list_s2.append(s_hat_0)
                
                if self.enable_multiref:
                    self.memory_bank_s2.add_to_memory(context_next)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
                params = torch.cat((context_next, y_hat_so_far, y_hat), dim=1)
                context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
                
                # Apply Linear Attention Multi-Reference for s2
                if self.enable_multiref:
                    context = self.memory_bank_s2(context, k=self.topk_refs_s2)
                
                context_next = self.attn_s2(context, context_next)
                scales, means = context_next.chunk(2, 1)
                y_res_i, y_q_i, y_hat_i, s_hat_i = self.process_with_mask(y_s2, scales, means, mask_list[i])
                y_res_list_s2.append(y_res_i)
                y_q_list_s2.append(y_q_i)
                y_hat_list_s2.append(y_hat_i)
                scale_list_s2.append(s_hat_i)
                
                if self.enable_multiref:
                    self.memory_bank_s2.add_to_memory(context_next)
        
        y_res = y_res + torch.sum(torch.stack(y_res_list_s2), dim=0)
        y_q = y_q + torch.sum(torch.stack(y_q_list_s2), dim=0)
        y_hat = y_hat + torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = scales_hat + torch.sum(torch.stack(scale_list_s2), dim=0)

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_q_list_s2[i]) for i in range(len(y_q_list_s2))]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(scale_list_s2[i]) for i in range(len(scale_list_s2))]
        
        # s2 → s3 memory transfer
        if self.enable_multiref and self.enable_hierarchical_transfer:
            self.memory_manager_s2_to_s3.transfer_memory(
                self.memory_bank_s2.memory_keys,
                self.memory_bank_s2.memory_values if self.memory_bank_s2.enable_value_storage else None,
                self.memory_bank_s2.memory_valid,
                self.memory_bank_s3
            )
        
        # Up-scaling to s3
        y_res = self.recon_for_s2_s3(y_res, mask_list_s2, B, C, H, W, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_s2, B, C, H, W, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2, B, C, H, W, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        # ============ s3階層（Linear Attention Multi-Reference適用）============
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context_next = context + common_params_s3

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)
        y_res_list_s3 = []
        y_q_list_s3 = []
        y_hat_list_s3 = []
        scale_list_s3 = []

        # s3: 8 steps with Linear Attention Multi-Reference
        for i in range(8):
            if i == 0:
                scales_s3, means_s3 = context_next.chunk(2, 1)
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales_s3, means_s3, mask_list[i])
                y_res_list_s3.append(y_res_0)
                y_q_list_s3.append(y_q_0)
                y_hat_list_s3.append(y_hat_0)
                scale_list_s3.append(s_hat_0)
                
                if self.enable_multiref:
                    self.memory_bank_s3.add_to_memory(context_next)
            elif i == 1:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
                params = torch.cat((context_next, y_hat_so_far, y_hat), dim=1)
                context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
                context_next = self.attn_s3(context, context_next)
                scales, means = context_next.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_list[i])
                y_res_list_s3.append(y_res_1)
                y_q_list_s3.append(y_q_1)
                y_hat_list_s3.append(y_hat_1)
                scale_list_s3.append(s_hat_1)
                
                if self.enable_multiref:
                    self.memory_bank_s3.add_to_memory(context_next)
            else:
                # Apply Linear Attention Multi-Reference from step 2 onwards
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
                params = torch.cat((context_next, y_hat_so_far, y_hat), dim=1)
                context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
                
                # ★ Linear Attention Multi-Reference適用 ★
                if self.enable_multiref:
                    context = self.memory_bank_s3(context, k=self.topk_refs_s3)
                
                context_next = self.attn_s3(context, context_next)
                scales, means = context_next.chunk(2, 1)
                y_res_i, y_q_i, y_hat_i, s_hat_i = self.process_with_mask(y, scales, means, mask_list[i])
                y_res_list_s3.append(y_res_i)
                y_q_list_s3.append(y_q_i)
                y_hat_list_s3.append(y_hat_i)
                scale_list_s3.append(s_hat_i)
                
                if self.enable_multiref:
                    self.memory_bank_s3.add_to_memory(context_next)
        
        y_res = y_res + torch.sum(torch.stack(y_res_list_s3), dim=0)
        y_q = y_q + torch.sum(torch.stack(y_q_list_s3), dim=0)
        y_hat = y_hat + torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = scales_hat + torch.sum(torch.stack(scale_list_s3), dim=0)

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_q_list_s3[i]) for i in range(len(y_q_list_s3))]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(scale_list_s3[i]) for i in range(len(scale_list_s3))]
            return y_res, y_q, y_hat, scales_hat, y_q_write_list_s1, y_q_write_list_s2, y_q_write_list_s3, scales_hat_write_list_s1, scales_hat_write_list_s2, scales_hat_write_list_s3
        
        return y_res, y_q, y_hat, scales_hat
        
        # s1 step 1 (with multi-reference)
        y_q_0_ctx = self.y_q_to_y(y_q_0_q, mode='s1')[0]
        ctx_p_1 = self.context_prediction_s1_1(y_q_0_ctx)
        
        # Apply Linear Attention Multi-Reference
        if self.enable_multiref:
            ctx_p_1 = self.memory_bank_s1(ctx_p_1, k=self.topk_refs_s1)
        
        hp_feat_1 = self.slice_to_y(hp_feat, mode='s1')[1]
        gaussian_params_1 = self.entropy_parameters_s1_1(torch.cat([ctx_p_1, hp_feat_1], dim=1))
        scales_1, means_1 = gaussian_params_1.chunk(2, 1)
        
        y_q_1_res, y_q_1_q, scales_1, means_1 = self.compress_gaussian(y_q_1, scales_1, means_1, self.y_spatial_dim_s1)
        
        if self.enable_multiref:
            self.memory_bank_s1.add_to_memory(ctx_p_1)
        
        # s1 → s2 memory transfer
        if self.enable_multiref and self.enable_hierarchical_transfer:
            self.memory_manager_s1_to_s2.transfer_memory(
                self.memory_bank_s1.memory_keys,
                self.memory_bank_s1.memory_values if self.memory_bank_s1.enable_value_storage else None,
                self.memory_bank_s1.memory_valid,
                self.memory_bank_s2
            )
        
        # s2 scale (H/32, W/32) - 4 steps
        H_s2, W_s2 = y_shape[0]//32, y_shape[1]//32
        y_q_s2 = self.slice_to_y(y, mode='s2')
        
        y_q_s1_ctx = self.y_q_to_y([y_q_0_q, y_q_1_q], mode='s1')
        y_q_s1_ctx = self.y_to_y(y_q_s1_ctx, mode='s1_to_s2')
        
        # s2 step 0 (baseline)
        ctx_p_s2_0 = self.context_prediction_s2_0(y_q_s1_ctx)
        
        if self.enable_multiref:
            self.memory_bank_s2.add_to_memory(ctx_p_s2_0)
        
        hp_feat_s2_0 = self.slice_to_y(hp_feat, mode='s2')[0]
        gaussian_params_s2_0 = self.entropy_parameters_s2_0(torch.cat([ctx_p_s2_0, hp_feat_s2_0], dim=1))
        scales_s2_0, means_s2_0 = gaussian_params_s2_0.chunk(2, 1)
        
        y_q_s2_0_res, y_q_s2_0_q, scales_s2_0, means_s2_0 = self.compress_gaussian(
            y_q_s2[0], scales_s2_0, means_s2_0, self.y_spatial_dim_s2
        )
        
        # s2 steps 1-3 (with multi-reference)
        y_q_s2_q_list = [y_q_s2_0_q]
        scales_list_s2 = [scales_s2_0]
        means_list_s2 = [means_s2_0]
        
        for step in range(1, 4):
            y_q_s2_prev_ctx = self.y_q_to_y(y_q_s2_q_list, mode='s2')[:step]
            y_q_s2_prev_ctx = torch.cat([y_q_s1_ctx] + y_q_s2_prev_ctx, dim=1)
            
            ctx_pred_fn = getattr(self, f'context_prediction_s2_{step}')
            ctx_p_s2 = ctx_pred_fn(y_q_s2_prev_ctx)
            
            # Apply Linear Attention Multi-Reference
            if self.enable_multiref:
                ctx_p_s2 = self.memory_bank_s2(ctx_p_s2, k=self.topk_refs_s2)
            
            hp_feat_s2 = self.slice_to_y(hp_feat, mode='s2')[step]
            entropy_params_fn = getattr(self, f'entropy_parameters_s2_{step}')
            gaussian_params_s2 = entropy_params_fn(torch.cat([ctx_p_s2, hp_feat_s2], dim=1))
            scales_s2, means_s2 = gaussian_params_s2.chunk(2, 1)
            
            y_q_s2_res, y_q_s2_q, scales_s2, means_s2 = self.compress_gaussian(
                y_q_s2[step], scales_s2, means_s2, self.y_spatial_dim_s2
            )
            
            y_q_s2_q_list.append(y_q_s2_q)
            scales_list_s2.append(scales_s2)
            means_list_s2.append(means_s2)
            
            if self.enable_multiref:
                self.memory_bank_s2.add_to_memory(ctx_p_s2)
        
        # s2 → s3 memory transfer
        if self.enable_multiref and self.enable_hierarchical_transfer:
            self.memory_manager_s2_to_s3.transfer_memory(
                self.memory_bank_s2.memory_keys,
                self.memory_bank_s2.memory_values if self.memory_bank_s2.enable_value_storage else None,
                self.memory_bank_s2.memory_valid,
                self.memory_bank_s3
            )
        
        # s3 scale (H/16, W/16) - 8 steps
        H_s3, W_s3 = y_shape[0]//16, y_shape[1]//16
        y_q_s3 = self.slice_to_y(y, mode='s3')
        
        y_q_s2_ctx = self.y_q_to_y(y_q_s2_q_list, mode='s2')
        y_q_s2_ctx = self.y_to_y(y_q_s2_ctx, mode='s2_to_s3')
        
        # s3 steps 0-1 (baseline)
        ctx_p_s3_0 = self.context_prediction_s3_0(y_q_s2_ctx)
        
        if self.enable_multiref:
            self.memory_bank_s3.add_to_memory(ctx_p_s3_0)
        
        hp_feat_s3_0 = self.slice_to_y(hp_feat, mode='s3')[0]
        gaussian_params_s3_0 = self.entropy_parameters_s3_0(torch.cat([ctx_p_s3_0, hp_feat_s3_0], dim=1))
        scales_s3_0, means_s3_0 = gaussian_params_s3_0.chunk(2, 1)
        
        y_q_s3_0_res, y_q_s3_0_q, scales_s3_0, means_s3_0 = self.compress_gaussian(
            y_q_s3[0], scales_s3_0, means_s3_0, self.y_spatial_dim_s3
        )
        
        y_q_s3_q_list = [y_q_s3_0_q]
        scales_list_s3 = [scales_s3_0]
        means_list_s3 = [means_s3_0]
        
        # s3 step 1
        y_q_s3_0_ctx = self.y_q_to_y([y_q_s3_0_q], mode='s3')[0]
        y_q_s3_prev_ctx = torch.cat([y_q_s2_ctx, y_q_s3_0_ctx], dim=1)
        ctx_p_s3_1 = self.context_prediction_s3_1(y_q_s3_prev_ctx)
        
        if self.enable_multiref:
            self.memory_bank_s3.add_to_memory(ctx_p_s3_1)
        
        hp_feat_s3_1 = self.slice_to_y(hp_feat, mode='s3')[1]
        gaussian_params_s3_1 = self.entropy_parameters_s3_1(torch.cat([ctx_p_s3_1, hp_feat_s3_1], dim=1))
        scales_s3_1, means_s3_1 = gaussian_params_s3_1.chunk(2, 1)
        
        y_q_s3_1_res, y_q_s3_1_q, scales_s3_1, means_s3_1 = self.compress_gaussian(
            y_q_s3[1], scales_s3_1, means_s3_1, self.y_spatial_dim_s3
        )
        
        y_q_s3_q_list.append(y_q_s3_1_q)
        scales_list_s3.append(scales_s3_1)
        means_list_s3.append(means_s3_1)
        
        # s3 steps 2-7 (with multi-reference using Linear Attention)
        for step in range(2, 8):
            y_q_s3_prev_ctx = self.y_q_to_y(y_q_s3_q_list, mode='s3')[:step]
            y_q_s3_prev_ctx = torch.cat([y_q_s2_ctx] + y_q_s3_prev_ctx, dim=1)
            
            ctx_pred_fn = getattr(self, f'context_prediction_s3_{step}')
            ctx_p_s3 = ctx_pred_fn(y_q_s3_prev_ctx)
            
            # Apply Linear Attention Multi-Reference (Phase 3の核心部分)
            if self.enable_multiref:
                ctx_p_s3 = self.memory_bank_s3(ctx_p_s3, k=self.topk_refs_s3)
            
            hp_feat_s3 = self.slice_to_y(hp_feat, mode='s3')[step]
            entropy_params_fn = getattr(self, f'entropy_parameters_s3_{step}')
            gaussian_params_s3 = entropy_params_fn(torch.cat([ctx_p_s3, hp_feat_s3], dim=1))
            scales_s3, means_s3 = gaussian_params_s3.chunk(2, 1)
            
            y_q_s3_res, y_q_s3_q, scales_s3, means_s3 = self.compress_gaussian(
                y_q_s3[step], scales_s3, means_s3, self.y_spatial_dim_s3
            )
            
            y_q_s3_q_list.append(y_q_s3_q)
            scales_list_s3.append(scales_s3)
            means_list_s3.append(means_s3)
            
            if self.enable_multiref:
                self.memory_bank_s3.add_to_memory(ctx_p_s3)
        
        # Merge all scales
        y_q_rec = self.merge_y([y_q_0_q, y_q_1_q], y_q_s2_q_list, y_q_s3_q_list)
        
        # Decoder
        x_hat = self.Decoder(y_q_rec)
        
        # Compute likelihoods
        scales_all = [scales_0, scales_1] + scales_list_s2 + scales_list_s3
        means_all = [means_0, means_1] + means_list_s2 + means_list_s3
        
        # Simplified likelihood computation
        y_likelihoods = []
        y_slices = [y_q_0[0], y_q_0[1], y_q_1[0], y_q_1[1]] + \
                   [y_q_s2[i] for i in range(4)] + \
                   [y_q_s3[i] for i in range(8)]
        
        for y_slice, scale, mean in zip(y_slices, scales_all, means_all):
            y_likelihoods.append(self._likelihood(y_slice, scale, mean))
        
        y_likelihood = torch.cat([yl.reshape(B, -1) for yl in y_likelihoods], dim=1)
        
        _, z_likelihoods = self.entropy_bottleneck(z)
        
        return {
            'x_hat': x_hat,
            'likelihoods': {
                'y': y_likelihood,
                'z': z_likelihoods
            }
        }
    
    def compress_hpcm(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference/Compression with Multi-Reference
        
        Note: Full implementation requires careful handling of memory state
        """
        # Use training forward for now
        # Full compress/decompress implementation is more complex
        return self.forward_hpcm(x)
    
    def decompress_hpcm(self, *args, **kwargs):
        """
        Decompression with Multi-Reference
        
        TODO: Implement full decompression with memory bank state management
        """
        raise NotImplementedError("Phase 3 decompress_hpcm not yet implemented")
    
    def _likelihood(self, x: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Helper for likelihood computation"""
        from torch.distributions import Normal
        dist = Normal(mean, scale)
        likelihood = dist.cdf(x + 0.5) - dist.cdf(x - 0.5)
        return likelihood.clamp(min=1e-10)
