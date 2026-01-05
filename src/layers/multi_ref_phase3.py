"""
Phase 3: Linear Attention based Multi-Reference Memory Bank

MLIC++のLinearGlobalInterContextを参考に、計算効率の高いLinear Attentionを実装。

主な改善点:
- Channel-wise linear attention (O(N²) → O(N))
- Kernel feature map (ELU-based)
- より効率的なmemory retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LinearAttentionMemoryBank(nn.Module):
    """
    Linear Attention based Multi-Reference Memory Bank
    
    MLIC++のLinearGlobalInterContextを参考にした効率的な実装。
    Channel-wise attentionにより空間次元を統合し、計算量をO(N)に削減。
    
    Args:
        context_dim: コンテキストの次元数（例: 640）
        max_refs: 最大参照フレーム数
        compress_ratio: キー圧縮率（メモリ節約用）
        value_resolution: Value保存時の解像度
        num_heads: Attention head数
        kernel_type: Kernel feature map type ('elu', 'relu', 'softmax')
        enable_value_storage: Value保存を有効化
    """
    
    def __init__(
        self,
        context_dim: int = 640,
        max_refs: int = 4,
        compress_ratio: int = 4,
        value_resolution: int = 8,
        num_heads: int = 8,
        kernel_type: str = 'elu',
        enable_value_storage: bool = True
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.max_refs = max_refs
        self.compress_ratio = compress_ratio
        self.value_resolution = value_resolution
        self.num_heads = num_heads
        self.kernel_type = kernel_type
        self.enable_value_storage = enable_value_storage
        
        self.compressed_dim = context_dim // compress_ratio
        self.head_dim = context_dim // num_heads
        
        # Linear Attention用のprojection layers
        # MLIC++スタイル: Channel-wise attention
        self.query_proj = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 1),
            nn.GroupNorm(num_heads, context_dim),
        )
        
        self.key_proj = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 1),
            nn.GroupNorm(num_heads, context_dim),
        )
        
        if enable_value_storage:
            self.value_proj = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 1),
                nn.GroupNorm(num_heads, context_dim),
            )
            
            # Value encoder (低解像度保存用)
            self.value_encoder = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 3, 1, 1, groups=context_dim),
                nn.Conv2d(context_dim, context_dim, 1),
                nn.GELU(),
                nn.Conv2d(context_dim, context_dim, 1),
            )
            
            # Value decoder (復元用)
            self.value_decoder = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 1),
                nn.GELU(),
                nn.Conv2d(context_dim, context_dim, 3, 1, 1, groups=context_dim),
                nn.Conv2d(context_dim, context_dim, 1),
            )
        
        # Fusion network (改良版)
        self.fusion_net = nn.Sequential(
            nn.Conv2d(context_dim * 2, context_dim, 1),
            nn.GroupNorm(8, context_dim),
            nn.GELU(),
            nn.Conv2d(context_dim, context_dim, 3, 1, 1),
            nn.GroupNorm(8, context_dim),
            nn.GELU(),
            nn.Conv2d(context_dim, context_dim, 1),
        )
        
        self.gate_net = nn.Sequential(
            nn.Conv2d(context_dim * 2, context_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(context_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Temperature parameter (学習可能)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Memory buffers
        self.register_buffer('memory_keys', torch.zeros(1, max_refs, context_dim))
        self.register_buffer('memory_valid', torch.zeros(1, max_refs, dtype=torch.bool))
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))
        
        if enable_value_storage:
            self.register_buffer(
                'memory_values',
                torch.zeros(1, max_refs, context_dim, value_resolution, value_resolution)
            )
    
    def reset(self):
        """メモリバンクをリセット"""
        self.memory_keys.zero_()
        self.memory_valid.zero_()
        self.memory_count.zero_()
        if self.enable_value_storage:
            self.memory_values.zero_()
    
    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kernel feature map φ(x)
        
        MLIC++スタイルのfeature map。
        ELUベースでnon-negativityを保証。
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Feature map [B, C, H, W]
        """
        if self.kernel_type == 'elu':
            return F.elu(x) + 1.0  # ELU+1で非負を保証
        elif self.kernel_type == 'relu':
            return F.relu(x) + 1e-6
        elif self.kernel_type == 'softmax':
            # Softmax attention (fallback)
            return x
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def linear_attention_similarity(
        self,
        query: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear Attention による類似度計算
        
        φ(Q)φ(K)^T の形式で計算量を削減。
        
        Args:
            query: [B, C, H, W]
            keys: [B, num_refs, C]
        
        Returns:
            similarity: [B, num_refs]
        """
        B, C, H, W = query.shape
        _, num_refs, _ = keys.shape
        
        # Query projection & feature map
        q = self.query_proj(query)  # [B, C, H, W]
        q = self.kernel_feature_map(q)  # φ(Q)
        
        # Global average pooling for channel-wise attention
        # MLIC++スタイル: HW次元を統合
        q_global = q.mean(dim=[2, 3])  # [B, C]
        
        # Keys: already stored as [B, num_refs, C]
        k = keys  # [B, num_refs, C]
        
        # Linear attention: φ(Q) · φ(K)^T
        # [B, C] x [B, C, num_refs] = [B, num_refs]
        similarity = torch.matmul(q_global.unsqueeze(1), k.transpose(1, 2)).squeeze(1)
        
        # Temperature scaling
        similarity = similarity / (self.temperature + 1e-8)
        
        return similarity  # [B, num_refs]
    
    def add_to_memory(self, context: torch.Tensor):
        """
        現在のコンテキストをメモリに追加
        
        Args:
            context: [B, C, H, W]
        """
        B = context.shape[0]
        
        # Ensure batch size matches
        if self.memory_keys.shape[0] != B:
            self.memory_keys = self.memory_keys.repeat(B, 1, 1)
            self.memory_valid = self.memory_valid.repeat(B, 1)
            self.memory_count = self.memory_count.repeat(B)
            if self.enable_value_storage:
                self.memory_values = self.memory_values.repeat(B, 1, 1, 1, 1)
        
        # Key encoding: Channel-wise global feature
        key_feat = self.key_proj(context)  # [B, C, H, W]
        key_feat = self.kernel_feature_map(key_feat)
        key_global = key_feat.mean(dim=[2, 3])  # [B, C]
        
        # Value encoding
        if self.enable_value_storage:
            value_feat = self.value_encoder(context)  # [B, C, H, W]
            value_downsampled = F.adaptive_avg_pool2d(
                value_feat, (self.value_resolution, self.value_resolution)
            )  # [B, C, res, res]
        
        # Add to memory (circular buffer)
        for b in range(B):
            idx = self.memory_count[b] % self.max_refs
            self.memory_keys[b, idx] = key_global[b]
            self.memory_valid[b, idx] = True
            if self.enable_value_storage:
                self.memory_values[b, idx] = value_downsampled[b]
            self.memory_count[b] += 1
    
    def query_memory(
        self,
        current_context: torch.Tensor,
        k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        メモリからTop-k参照を検索
        
        Args:
            current_context: [B, C, H, W]
            k: Top-k数
        
        Returns:
            attn_weights: [B, k] - Attention weights
            topk_indices: [B, k] - Top-k indices
            valid: [B] - 有効なメモリが存在するか
        """
        B = current_context.shape[0]
        
        # Check if memory has valid entries
        has_valid = self.memory_valid.any(dim=1)  # [B]
        
        if not has_valid.any():
            # No valid memory
            dummy_weights = torch.zeros(B, k, device=current_context.device)
            dummy_indices = torch.zeros(B, k, dtype=torch.long, device=current_context.device)
            return dummy_weights, dummy_indices, has_valid
        
        # Linear attention similarity
        similarity = self.linear_attention_similarity(
            current_context, self.memory_keys
        )  # [B, num_refs]
        
        # Mask invalid entries
        similarity = similarity.masked_fill(~self.memory_valid, float('-inf'))
        
        # Top-k selection
        k_actual = min(k, self.memory_valid.sum(dim=1).max().item())
        topk_sim, topk_indices = torch.topk(similarity, k_actual, dim=1)
        
        # Softmax normalization
        attn_weights = F.softmax(topk_sim, dim=1)  # [B, k]
        
        # Pad if necessary
        if k_actual < k:
            pad_size = k - k_actual
            attn_weights = F.pad(attn_weights, (0, pad_size), value=0)
            topk_indices = F.pad(topk_indices, (0, pad_size), value=0)
        
        return attn_weights, topk_indices, has_valid
    
    def retrieve_values(
        self,
        topk_indices: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        """
        Top-k indicesに対応するValueを取得
        
        Args:
            topk_indices: [B, k]
            target_size: (H, W) - 復元先サイズ
        
        Returns:
            values: [B, k, C, H, W] or None
        """
        if not self.enable_value_storage:
            return None
        
        B, k = topk_indices.shape
        C = self.context_dim
        H, W = target_size
        
        # Gather values from memory
        # [B, k, C, res, res]
        indices_expanded = topk_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        indices_expanded = indices_expanded.expand(
            B, k, C, self.value_resolution, self.value_resolution
        )
        
        gathered_values = torch.gather(
            self.memory_values.unsqueeze(1).expand(-1, k, -1, -1, -1, -1).reshape(
                B, k, self.max_refs, C, self.value_resolution, self.value_resolution
            ).gather(2, indices_expanded.unsqueeze(2)).squeeze(2),
            1,
            indices_expanded
        )  # [B, k, C, res, res]
        
        # Simpler gathering
        gathered_values = torch.stack([
            self.memory_values[b, topk_indices[b]]
            for b in range(B)
        ], dim=0)  # [B, k, C, res, res]
        
        # Upsample to target resolution
        gathered_values = gathered_values.view(B * k, C, self.value_resolution, self.value_resolution)
        upsampled = F.interpolate(gathered_values, size=(H, W), mode='bilinear', align_corners=False)
        upsampled = upsampled.view(B, k, C, H, W)
        
        # Decode
        upsampled_flat = upsampled.view(B * k, C, H, W)
        decoded = self.value_decoder(upsampled_flat)
        decoded = decoded.view(B, k, C, H, W)
        
        return decoded
    
    def fuse_references(
        self,
        current_context: torch.Tensor,
        ref_contexts: torch.Tensor,
        attn_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        参照コンテキストとの融合
        
        Args:
            current_context: [B, C, H, W]
            ref_contexts: [B, k, C, H, W]
            attn_weights: [B, k]
        
        Returns:
            fused: [B, C, H, W]
        """
        B, C, H, W = current_context.shape
        k = ref_contexts.shape[1]
        
        # Weighted sum of references
        # [B, k, 1, 1, 1] * [B, k, C, H, W] -> [B, k, C, H, W]
        weighted_refs = ref_contexts * attn_weights.view(B, k, 1, 1, 1)
        aggregated_ref = weighted_refs.sum(dim=1)  # [B, C, H, W]
        
        # Concatenate and fuse
        concat = torch.cat([current_context, aggregated_ref], dim=1)  # [B, 2C, H, W]
        
        fused = self.fusion_net(concat)  # [B, C, H, W]
        gate = self.gate_net(concat)  # [B, 1, H, W]
        
        # Gated residual
        output = gate * fused + (1 - gate) * current_context
        
        return output
    
    def forward(
        self,
        current_context: torch.Tensor,
        k: int = 2
    ) -> torch.Tensor:
        """
        Multi-Reference Enhancement
        
        Args:
            current_context: [B, C, H, W]
            k: Top-k参照数
        
        Returns:
            enhanced_context: [B, C, H, W]
        """
        B, C, H, W = current_context.shape
        
        # Query memory
        attn_weights, topk_indices, valid = self.query_memory(current_context, k)
        
        if not valid.any():
            return current_context
        
        # Retrieve values
        ref_contexts = self.retrieve_values(topk_indices, (H, W))
        
        if ref_contexts is None:
            return current_context
        
        # Fuse
        enhanced = self.fuse_references(current_context, ref_contexts, attn_weights)
        
        return enhanced


class HierarchicalLinearMemoryManager(nn.Module):
    """
    階層間メモリ転送（Linear Attention版）
    
    下位階層のメモリを上位階層に効率的に伝達。
    """
    
    def __init__(self, context_dim: int = 640, num_heads: int = 8):
        super().__init__()
        
        self.context_dim = context_dim
        self.num_heads = num_heads
        
        # Cross-layer projection
        self.cross_proj = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 1),
            nn.GroupNorm(num_heads, context_dim),
            nn.GELU(),
        )
    
    def transfer_memory(
        self,
        source_memory_keys: torch.Tensor,
        source_memory_values: torch.Tensor,
        source_valid: torch.Tensor,
        target_memory: 'LinearAttentionMemoryBank'
    ):
        """
        下位階層から上位階層へメモリ転送
        
        Args:
            source_memory_keys: [B, num_refs, C]
            source_memory_values: [B, num_refs, C, res, res]
            source_valid: [B, num_refs]
            target_memory: 上位階層のMemoryBank
        """
        B, num_refs, C = source_memory_keys.shape
        
        # Valid entries only
        for b in range(B):
            valid_mask = source_valid[b]
            if not valid_mask.any():
                continue
            
            valid_keys = source_memory_keys[b, valid_mask]  # [num_valid, C]
            
            # Transfer keys (already in global feature form)
            for key_vec in valid_keys:
                idx = target_memory.memory_count[b] % target_memory.max_refs
                target_memory.memory_keys[b, idx] = key_vec
                target_memory.memory_valid[b, idx] = True
                
                # Transfer values if available
                if target_memory.enable_value_storage and source_memory_values is not None:
                    # Note: May need resolution adjustment
                    target_memory.memory_values[b, idx] = source_memory_values[b, valid_mask][0]
                
                target_memory.memory_count[b] += 1
