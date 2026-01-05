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

from src.models.HPCM_Base import *
from src.layers.multi_ref_phase3 import LinearAttentionMemoryBank, HierarchicalLinearMemoryManager


class HPCM_MultiRef_Phase3(HPCM_Base):
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
        super().__init__(M=M, N=N, **kwargs)
        
        self.enable_multiref = enable_multiref
        self.enable_hierarchical_transfer = enable_hierarchical_transfer
        
        if not enable_multiref:
            return
        
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
        
        # Standard HPCM forward
        if training:
            return self.forward_hpcm(x)
        else:
            return self.compress_hpcm(x)
    
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
        
        # s1 step 0 (baseline)
        ctx_p_0 = torch.zeros(B, 2*self.N, H_s1, W_s1, device=y.device)
        ctx_p_0 = self.context_prediction_s1_0(ctx_p_0)
        
        hp_feat_0 = self.slice_to_y(hp_feat, mode='s1')[0]
        gaussian_params_0 = self.entropy_parameters_s1_0(torch.cat([ctx_p_0, hp_feat_0], dim=1))
        scales_0, means_0 = gaussian_params_0.chunk(2, 1)
        
        y_q_0_res, y_q_0_q, scales_0, means_0 = self.compress_gaussian(y_q_0, scales_0, means_0, self.y_spatial_dim_s1)
        
        # Add to memory s1
        if self.enable_multiref:
            self.memory_bank_s1.add_to_memory(ctx_p_0)
        
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
