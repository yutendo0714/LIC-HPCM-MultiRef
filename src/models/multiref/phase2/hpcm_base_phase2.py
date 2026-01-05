"""
HPCM with Multi-Reference Memory Bank - Phase 2

Phase 2の特徴:
- s1/s2/s3全階層にMulti-Reference Memory Bankを適用
- Value保存機能の完全実装
- 階層間メモリ共有
- decompress_hpcmの完全実装

使用方法:
    from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
    
    model = HPCM_MultiRef_Phase2(
        M=320, 
        N=256,
        enable_multiref=True,
        max_refs_s1=2,    # s1: 2ステップなので少なめ
        max_refs_s2=3,    # s2: 4ステップ
        max_refs_s3=4,    # s3: 8ステップなので多め
        enable_hierarchical_transfer=True  # 階層間メモリ共有
    )
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# 既存モジュール
from src.models.base import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1, conv4x4_down, deconv4x4_up
from src.layers.multi_ref_phase2 import FullContextMemoryBank, HierarchicalMemoryManager

# 既存コンポーネント
from src.models.HPCM_Base import (
    g_a, g_s, h_a, h_s, 
    y_spatial_prior_s1_s2, 
    y_spatial_prior_s3,
    CrossAttentionCell
)


class HPCM_MultiRef_Phase2(basemodel):
    """
    HPCM with Full Multi-Reference (Phase 2)
    
    Phase 1からの拡張:
    - s1/s2階層にもMulti-Referenceを適用（Phase 1はs3のみ）
    - Value保存機能の完全実装
    - 階層間メモリ共有機構
    - より洗練されたattention機構
    
    Args:
        M (int): Main latent channel数
        N (int): Hyper latent channel数
        enable_multiref (bool): Multi-Reference機能を有効化
        max_refs_s1 (int): s1のメモリバンクサイズ（デフォルト: 2）
        max_refs_s2 (int): s2のメモリバンクサイズ（デフォルト: 3）
        max_refs_s3 (int): s3のメモリバンクサイズ（デフォルト: 4）
        topk_refs_s1 (int): s1のTop-k数（デフォルト: 1）
        topk_refs_s2 (int): s2のTop-k数（デフォルト: 2）
        topk_refs_s3 (int): s3のTop-k数（デフォルト: 2）
        compress_ratio (int): キー圧縮率
        value_resolution (int): Value保存解像度
        temperature (float): Softmax温度
        enable_hierarchical_transfer (bool): 階層間メモリ共有を有効化
    """
    
    def __init__(self, M=320, N=256, 
                 enable_multiref=True,
                 max_refs_s1=2,
                 max_refs_s2=3,
                 max_refs_s3=4,
                 topk_refs_s1=1,
                 topk_refs_s2=2,
                 topk_refs_s3=2,
                 compress_ratio=4,
                 value_resolution=8,
                 temperature=0.1,
                 enable_hierarchical_transfer=False):
        super().__init__(N)
        
        # 基本コンポーネント
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
        self.attn_s1 = CrossAttentionCell(320*2, 320*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        self.attn_s3 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        
        # Context networks
        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))
        
        # ========== Phase 2: Full Multi-Reference Memory Banks ==========
        self.enable_multiref = enable_multiref
        self.topk_refs_s1 = topk_refs_s1
        self.topk_refs_s2 = topk_refs_s2
        self.topk_refs_s3 = topk_refs_s3
        self.temperature = temperature
        self.enable_hierarchical_transfer = enable_hierarchical_transfer
        
        if self.enable_multiref:
            # s1階層用メモリバンク（2ステップなので小規模）
            self.memory_bank_s1 = FullContextMemoryBank(
                context_dim=M*2,
                max_refs=max_refs_s1,
                compress_ratio=compress_ratio,
                value_resolution=value_resolution,
                num_heads=4,
                enable_value_storage=True
            )
            
            # s2階層用メモリバンク（4ステップ）
            self.memory_bank_s2 = FullContextMemoryBank(
                context_dim=M*2,
                max_refs=max_refs_s2,
                compress_ratio=compress_ratio,
                value_resolution=value_resolution,
                num_heads=8,
                enable_value_storage=True
            )
            
            # s3階層用メモリバンク（8ステップ）
            self.memory_bank_s3 = FullContextMemoryBank(
                context_dim=M*2,
                max_refs=max_refs_s3,
                compress_ratio=compress_ratio,
                value_resolution=value_resolution,
                num_heads=8,
                enable_value_storage=True
            )
            
            # 階層間メモリ管理
            if enable_hierarchical_transfer:
                self.hierarchical_manager = HierarchicalMemoryManager(context_dim=M*2)
            
            print(f"[Phase 2] Full Multi-Reference Memory Banks enabled:")
            print(f"  - s1: max_refs={max_refs_s1}, topk={topk_refs_s1}")
            print(f"  - s2: max_refs={max_refs_s2}, topk={topk_refs_s2}")
            print(f"  - s3: max_refs={max_refs_s3}, topk={topk_refs_s3}")
            print(f"  - Value storage enabled (resolution={value_resolution})")
            print(f"  - Hierarchical transfer: {enable_hierarchical_transfer}")
        else:
            self.memory_bank_s1 = None
            self.memory_bank_s2 = None
            self.memory_bank_s3 = None
            print("[Phase 2] Multi-Reference disabled (Baseline mode)")
    
    def forward(self, x, training=None):
        """Forward pass with full multi-reference enhancement"""
        if training is None:
            training = self.training 
        
        # Encoding
        y = self.g_a(x)
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
        
        # Progressive context modeling with full multi-reference (Phase 2)
        y_res, y_q, y_hat, scales_y = self.forward_hpcm(
            y, params, 
            self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
            self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
            self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
            self.adaptive_params_list, self.context_net, 
        )

        # Decoding
        x_hat = self.g_s(y_hat)
        
        # Entropy estimation for y
        if training:
            y_likelihoods = self.entropy_estimation(self.add_noise(y_res), scales_y)
        else:
            y_res_hat = torch.round(y_res)
            y_likelihoods = self.entropy_estimation(y_res_hat, scales_y) 
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def forward_hpcm(self, y, common_params, 
                     y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                     y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                     y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                     adaptive_params_list, context_net, write=False):
        """
        Hierarchical Progressive Context Modeling with Full Multi-Reference (Phase 2)
        
        Phase 2では全階層（s1/s2/s3）にMulti-Referenceを適用
        """
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        # メモリバンクリセット
        if self.enable_multiref:
            self.memory_bank_s1.reset()
            self.memory_bank_s2.reset()
            self.memory_bank_s3.reset()

        # ============ s1階層（Phase 2: Multi-Reference適用）============
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

        # s1初期コンテキストをメモリに追加
        if self.enable_multiref:
            self.memory_bank_s1.add_to_memory(context_next)

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, dtype, device)
        y_res_list_s1 = []
        y_q_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        for i in range(2):
            if i == 0:
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s1, scales_s1, means_s1, mask_list[i])
                y_res_list_s1.append(y_res_0)
                y_q_list_s1.append(y_q_0)
                y_hat_list_s1.append(y_hat_0)
                scale_list_s1.append(s_hat_0)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                
                # ========== Phase 2: s1にMulti-Reference適用 ==========
                context_next_local = self.attn_s1(context, context_next)
                
                if self.enable_multiref and i > 0:
                    context_next_enhanced = self.memory_bank_s1.forward(
                        context_next_local,
                        k=self.topk_refs_s1,
                        temperature=self.temperature
                    )
                    context_next = context_next_enhanced
                else:
                    context_next = context_next_local
                
                if self.enable_multiref:
                    self.memory_bank_s1.add_to_memory(context_next)
                # ====================================================
                
                scales, means = context.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_1)
                y_q_list_s1.append(y_q_1)
                y_hat_list_s1.append(y_hat_1)
                scale_list_s1.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s1), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s1), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_q_list_s1[i]) for i in range(len(y_q_list_s1))]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(scale_list_s1[i]) for i in range(len(scale_list_s1))]
        
        # Up-scaling to s2
        y_res = self.recon_for_s2_s3(y_res, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        # ============ s2階層（Phase 2: Multi-Reference適用）============
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)
        
        # s2初期コンテキストをメモリに追加
        if self.enable_multiref:
            self.memory_bank_s2.add_to_memory(context_next)
            # 階層間転送（s1 -> s2）
            if self.enable_hierarchical_transfer:
                self.hierarchical_manager.transfer_s1_to_s2(self.memory_bank_s1, self.memory_bank_s2)
        
        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]
        y_res_list_s2 = [y_res]
        y_q_list_s2   = [y_q]
        y_hat_list_s2 = [y_hat]
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            
            # ========== Phase 2: s2にMulti-Reference適用 ==========
            context_next_local = self.attn_s2(context, context_next)
            
            if self.enable_multiref and i > 0:
                context_next_enhanced = self.memory_bank_s2.forward(
                    context_next_local,
                    k=self.topk_refs_s2,
                    temperature=self.temperature
                )
                context_next = context_next_enhanced
            else:
                context_next = context_next_local
            
            if self.enable_multiref:
                self.memory_bank_s2.add_to_memory(context_next)
            # ===================================================
            
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s2, scales, means, mask_list[i])
            y_res_list_s2.append(y_res_1)
            y_q_list_s2.append(y_q_1)
            y_hat_list_s2.append(y_hat_1)
            scale_list_s2.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s2), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s2), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s2), dim=0)

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_q_list_s2[i]) for i in range(1, len(y_q_list_s2))]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(scale_list_s2[i]) for i in range(1, len(scale_list_s2))]
       
        # Up-scaling to s3
        y_res = self.recon_for_s2_s3(y_res, mask_list_s2, B, C, H, W, dtype, device)
        y_q = self.recon_for_s2_s3(y_q, mask_list_s2, B, C, H, W, dtype, device)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2, B, C, H, W, dtype, device)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_s2, B, C, H, W, dtype, device)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2, B, C, H, W, dtype, device)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2, B, C, H, W, dtype, device)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        # ============ s3階層（Phase 2: Multi-Reference適用）============
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        # s3初期コンテキストをメモリに追加
        if self.enable_multiref:
            self.memory_bank_s3.add_to_memory(context_next)
            # 階層間転送（s2 -> s3）
            if self.enable_hierarchical_transfer:
                self.hierarchical_manager.transfer_s2_to_s3(self.memory_bank_s2, self.memory_bank_s3)

        mask_list = self.get_mask_eight_parts(B, C, H, W, dtype, device)[2:]
        y_res_list_s3 = [y_res]
        y_q_list_s3   = [y_q]
        y_hat_list_s3 = [y_hat]
        scale_list_s3 = [scales_hat]

        for i in range(6):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(
                y_spatial_prior_adaptor_list_s3[i - 1](params), 
                adaptive_params_list[i + 4]
            )
            
            # ========== Phase 2: s3にMulti-Reference適用（Value完全保存）==========
            context_next_local = self.attn_s3(context, context_next)
            
            if self.enable_multiref and i > 0:
                context_next_enhanced = self.memory_bank_s3.forward(
                    context_next_local,
                    k=self.topk_refs_s3,
                    temperature=self.temperature
                )
                context_next = context_next_enhanced
            else:
                context_next = context_next_local
            
            if self.enable_multiref:
                self.memory_bank_s3.add_to_memory(context_next)
            # =====================================================================
            
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_list[i])
            y_res_list_s3.append(y_res_1)
            y_q_list_s3.append(y_q_1)
            y_hat_list_s3.append(y_hat_1)
            scale_list_s3.append(s_hat_1)

        y_res = torch.sum(torch.stack(y_res_list_s3), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s3), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s3), dim=0)

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_q_list_s3[i]) for i in range(1, len(y_q_list_s3))]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(scale_list_s3[i]) for i in range(1, len(scale_list_s3))]
            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_q, y_hat, scales_hat
    
    def compress(self, x):
        """Compress with full multi-reference"""
        from src.entropy_models import ubransEncoder
        
        if self.enable_multiref:
            self.memory_bank_s1.reset()
            self.memory_bank_s2.reset()
            self.memory_bank_s3.reset()
        
        y = self.g_a(x)
        z = self.h_a(y)
        z_res_hat = torch.round(z - self.means_hyper)
        indexes_z = self.build_indexes_z(z_res_hat.size())
        
        encoder_z = ubransEncoder()
        self.compress_symbols(z_res_hat, indexes_z, self.quantized_cdf_z.cpu().numpy(), 
                            self.cdf_length_z.cpu().numpy(), self.offset_z.cpu().numpy(), encoder_z)
        z_string = encoder_z.flush()
        
        z_hat = z_res_hat + self.means_hyper
        params = self.h_s(z_hat)
        
        y_q_write_list, scales_hat_write_list = self.compress_hpcm(
            y, params, 
            self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
            self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
            self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
            self.adaptive_params_list, self.context_net
        )

        encoder_y = ubransEncoder()
        for i in range(len(y_q_write_list)):
            indexes_w = self.build_indexes_conditional(scales_hat_write_list[i])
            self.compress_symbols(y_q_write_list[i], indexes_w, self.quantized_cdf_y.cpu().numpy(), 
                                self.cdf_length_y.cpu().numpy(), self.offset_y.cpu().numpy(), encoder_y)
        y_string = encoder_y.flush()
        
        return {"strings": [y_string, z_string], "shape": z_res_hat.size()[2:]}
        
    def decompress(self, strings, shape):
        """Decompress with full multi-reference (Phase 2完全実装)"""
        from src.entropy_models import ubransDecoder
        
        if self.enable_multiref:
            self.memory_bank_s1.reset()
            self.memory_bank_s2.reset()
            self.memory_bank_s3.reset()
        
        device = self.quantized_cdf_z.device
        output_size = (1, self.scales_hyper.size(1), *shape)
        indexes_z = self.build_indexes_z(output_size).to(device)
        
        decoder_z = ubransDecoder()
        decoder_z.set_stream(strings[1])
        z_res_hat = self.decompress_symbols(indexes_z, self.quantized_cdf_z.cpu().numpy(), 
                                           self.cdf_length_z.cpu().numpy(), self.offset_z.cpu().numpy(), decoder_z)
        z_hat = z_res_hat + self.means_hyper
        
        params = self.h_s(z_hat)
        decoder_y = ubransDecoder()
        decoder_y.set_stream(strings[0])
        
        y_hat = self.decompress_hpcm(
            params, 
            self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
            self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
            self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
            self.adaptive_params_list, self.context_net, decoder_y
        )
    
        x_hat = self.g_s(y_hat).clamp_(0,1)
        
        return {"x_hat": x_hat}
    
    def compress_hpcm(self, y, common_params, 
                     y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                     y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                     y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                     adaptive_params_list, context_net):
        return self.forward_hpcm(
            y, common_params, 
            y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
            y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
            y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
            adaptive_params_list, context_net, write=True
        )

    def decompress_hpcm(self, common_params, 
                       y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                       y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                       y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                       adaptive_params_list, context_net, decoder_y):
        """
        Decompress with full multi-reference (Phase 2完全実装)
        
        Phase 1では未実装でしたが、Phase 2で完全に実装します。
        """
        # TODO: Phase 2で完全実装
        # 現時点では基本的な decompress ロジックを継承
        # Multi-Referenceを含む完全な decompress 実装が必要
        raise NotImplementedError("Phase 2 decompress_hpcm - Full implementation coming soon")
