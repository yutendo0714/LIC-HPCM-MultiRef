"""
HPCM with Multi-Reference Memory Bank - Phase 1

Phase 1 特徴:
- s3階層（最も情報量が多い8ステップ）のみにMulti-Reference Memory Bankを適用
- 軽量版実装（圧縮されたキーのみ保存）
- Top-k参照選択でメモリ効率化
- 既存のHPCMと互換性を保ちつつ、段階的に拡張可能な設計

使用方法:
    from src.models.multiref.phase1 import HPCM_MultiRef_Phase1
    
    model = HPCM_MultiRef_Phase1(
        M=320, 
        N=256,
        enable_multiref=True,  # Multi-Reference機能の有効/無効
        max_refs=4,            # メモリバンクの最大参照数
        topk_refs=2,           # Top-k参照数
    )
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# 既存モジュールのインポート
from src.models.base import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1, conv4x4_down, deconv4x4_up
from src.layers.multi_ref import LightweightContextMemoryBank

# 既存のHPCMからコンポーネントを再利用
from src.models.HPCM_Base import (
    g_a, g_s, h_a, h_s, 
    y_spatial_prior_s1_s2, 
    y_spatial_prior_s3,
    CrossAttentionCell
)


class HPCM_MultiRef_Phase1(basemodel):
    """
    HPCM with Multi-Reference Memory Bank (Phase 1)
    
    s3階層のみに履歴参照メモリバンクを適用した拡張版HPCM。
    従来の直前ステップ参照に加えて、過去の複数ステップから
    最も関連性の高い参照を選択的に利用することで、
    条件付きエントロピーを削減し、rate性能を向上させる。
    
    Args:
        M (int): Main latent channel数（デフォルト: 320）
        N (int): Hyper latent channel数（デフォルト: 256）
        enable_multiref (bool): Multi-Reference機能を有効化（デフォルト: True）
        max_refs (int): メモリバンクの最大参照保持数（デフォルト: 4）
        topk_refs (int): Top-k参照選択数（デフォルト: 2）
        compress_ratio (int): キー圧縮率（デフォルト: 4）
        temperature (float): Softmax温度パラメータ（デフォルト: 0.1）
    """
    
    def __init__(self, M=320, N=256, 
                 enable_multiref=True, 
                 max_refs=4, 
                 topk_refs=2,
                 compress_ratio=4,
                 temperature=0.1):
        super().__init__(N)
        
        # 基本コンポーネント（既存HPCMと同じ）
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
        
        # ========== Phase 1: Multi-Reference Memory Bank (s3のみ) ==========
        self.enable_multiref = enable_multiref
        self.max_refs = max_refs
        self.topk_refs = topk_refs
        self.temperature = temperature
        
        if self.enable_multiref:
            self.memory_bank_s3 = LightweightContextMemoryBank(
                context_dim=M*2,           # 640
                max_refs=max_refs,         # s3は8ステップあるので4個程度保持
                compress_ratio=compress_ratio,  # 640 -> 160 に圧縮
                num_heads=8
            )
            print(f"[Phase 1] Multi-Reference Memory Bank enabled for s3:")
            print(f"  - max_refs={max_refs}, topk_refs={topk_refs}")
            print(f"  - compress_ratio={compress_ratio}, temperature={temperature}")
        else:
            self.memory_bank_s3 = None
            print("[Phase 1] Multi-Reference disabled (Baseline mode)")
    
    def forward(self, x, training=None):
        """
        Forward pass with optional multi-reference enhancement
        
        Args:
            x: Input image [B, 3, H, W]
            training: Training mode flag
            
        Returns:
            dict: {
                "x_hat": Reconstructed image,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
            }
        """
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
        
        # Progressive context modeling with multi-reference (Phase 1: s3のみ)
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
        Hierarchical Progressive Context Modeling with Multi-Reference (Phase 1)
        
        Phase 1では s3階層のみにMulti-Reference Memory Bankを適用。
        s1, s2は既存のHPCMと同じ処理。
        """
        B, C, H, W = y.size()
        dtype = common_params.dtype
        device = common_params.device

        # ============ s1階層（既存HPCMと同じ）============
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
                context_next = self.attn_s1(context, context_next)
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

        # ============ s2階層（既存HPCMと同じ）============
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, dtype, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2, B, C, H // 2, W // 2, dtype, device)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)
        
        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, dtype, device)[1:]
        y_res_list_s2 = [y_res]
        y_q_list_s2   = [y_q]
        y_hat_list_s2 = [y_hat]
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            context_next = self.attn_s2(context, context_next)
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

        # ============ s3階層（Phase 1: Multi-Reference適用）============
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2, B, C, H, W, dtype, device)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2, B, C, H, W, dtype, device)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        # Multi-Reference Memory Bank のリセット（画像ごと）
        if self.enable_multiref:
            self.memory_bank_s3.reset()
            # 初期コンテキストをメモリに追加（s1, s2の情報を含む）
            self.memory_bank_s3.add_to_memory(context_next, store_value=True)

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
            
            # ========== Phase 1: Multi-Reference適用 ==========
            # 既存のローカルattention
            context_next_local = self.attn_s3(context, context_next)
            
            if self.enable_multiref and i > 0:  # 最初のステップはスキップ
                # メモリから Top-k 参照を取得して統合
                context_next_enhanced = self.memory_bank_s3.forward(
                    context_next_local, 
                    k=self.topk_refs, 
                    apply_fusion=True
                )
                context_next = context_next_enhanced
            else:
                context_next = context_next_local
            
            # 現ステップのコンテキストをメモリに追加
            if self.enable_multiref:
                self.memory_bank_s3.add_to_memory(context_next, store_value=True)
            # ================================================
            
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
        """Compress with multi-reference (圧縮時は順序固定なので簡易実装)"""
        from src.entropy_models import ubransEncoder
        
        # Multi-Reference有効時はリセット
        if self.enable_multiref:
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
        """Decompress with multi-reference"""
        from src.entropy_models import ubransDecoder
        
        # Multi-Reference有効時はリセット
        if self.enable_multiref:
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
        
        # decompress_hpcm は既存の実装を流用（Phase 1では省略可能）
        # 完全な実装には decompress_hpcm の拡張が必要
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
        """Compress with forward_hpcm"""
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
        Decompress with multi-reference
        Note: Phase 1では簡易実装。完全版はPhase 2で実装。
        """
        # 既存のHPCMのdecompress_hpcmを呼び出す
        # TODO: Multi-Reference対応版の実装（Phase 2）
        raise NotImplementedError("Decompress with multi-reference not fully implemented in Phase 1")
