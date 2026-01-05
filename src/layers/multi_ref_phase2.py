"""
Full Context Memory Bank with Value Storage - Phase 2

Phase 2の拡張機能:
- Value（実際のコンテキスト特徴）の保存と復元
- より効率的なメモリ管理
- 階層間でのメモリ共有サポート
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullContextMemoryBank(nn.Module):
    """
    完全版Context Memory Bank（Phase 2）
    
    Phase 1からの拡張:
    - Value保存機能の完全実装
    - 低解像度での保存→復元による効率化
    - より洗練されたattention機構
    - 階層間メモリ共有のサポート
    
    Args:
        context_dim (int): コンテキスト次元
        max_refs (int): 最大参照保持数
        compress_ratio (int): キー圧縮率
        value_resolution (int): Value保存時の空間解像度（8x8など）
        num_heads (int): Multi-head attention のヘッド数
        enable_value_storage (bool): Value保存を有効化
    """
    
    def __init__(self, 
                 context_dim=640, 
                 max_refs=4, 
                 compress_ratio=4,
                 value_resolution=8,
                 num_heads=8,
                 enable_value_storage=True):
        super().__init__()
        self.max_refs = max_refs
        self.key_dim = context_dim // compress_ratio
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.value_resolution = value_resolution
        self.enable_value_storage = enable_value_storage
        
        # キー圧縮（Phase 1と同じ）
        self.key_encoder = nn.Sequential(
            nn.Conv2d(context_dim, self.key_dim, 1, bias=False),
            nn.GroupNorm(min(8, self.key_dim), self.key_dim),
        )
        
        # クエリ生成（改良版）
        self.query_proj = nn.Sequential(
            nn.Conv2d(context_dim, self.key_dim, 1),
            nn.GELU(),
        )
        
        # Value投影・圧縮（Phase 2の新機能）
        if enable_value_storage:
            self.value_encoder = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 3, 1, 1, groups=context_dim),  # Depthwise
                nn.Conv2d(context_dim, context_dim, 1),
                nn.GELU(),
            )
            
            # Value復元用
            self.value_decoder = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 1),
                nn.GELU(),
                nn.Conv2d(context_dim, context_dim, 3, 1, 1, groups=context_dim),
            )
        
        # Multi-head attention用のprojection
        self.mha_query = nn.Linear(self.key_dim, self.key_dim)
        self.mha_key = nn.Linear(self.key_dim, self.key_dim)
        self.mha_value = nn.Linear(self.key_dim, self.key_dim)
        
        # Fusion network（改良版）
        self.fusion_net = nn.Sequential(
            nn.Conv2d(context_dim * 2, context_dim, 1),
            nn.GroupNorm(32, context_dim),
            nn.GELU(),
            nn.Conv2d(context_dim, context_dim, 3, 1, 1, groups=context_dim),
            nn.Conv2d(context_dim, context_dim, 1),
        )
        
        # Gated fusion（学習可能な重み）
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(context_dim * 2, context_dim, 1),
            nn.Sigmoid()
        )
        
        # メモリバッファ
        self.register_buffer('memory_keys', torch.zeros(1, max_refs, self.key_dim, 1, 1))
        self.register_buffer('memory_initialized', torch.zeros(max_refs, dtype=torch.bool))
        
        # Value保存用バッファ（固定サイズ）
        if enable_value_storage:
            self.register_buffer('memory_values', 
                               torch.zeros(1, max_refs, context_dim, value_resolution, value_resolution))
        
        self.current_step = 0
        
    def reset(self):
        """メモリリセット"""
        self.current_step = 0
        self.memory_keys.zero_()
        self.memory_initialized.zero_()
        if self.enable_value_storage:
            self.memory_values.zero_()
        
    def add_to_memory(self, context):
        """
        コンテキストをメモリに追加（Phase 2: Value完全保存）
        
        Args:
            context: [B, C, H, W] 現在のコンテキスト
        """
        B, C, H, W = context.shape
        
        # Key圧縮（Phase 1と同じ）
        key = self.key_encoder(context).mean(dim=[2, 3], keepdim=True)
        
        # Circular buffer index
        idx = self.current_step % self.max_refs
        
        # Key保存
        self.memory_keys[:, idx] = key.detach()
        self.memory_initialized[idx] = True
        
        # Value保存（Phase 2の新機能）
        if self.enable_value_storage:
            # 低解像度に圧縮して保存
            value_encoded = self.value_encoder(context)
            value_compressed = F.adaptive_avg_pool2d(value_encoded, (self.value_resolution, self.value_resolution))
            self.memory_values[:, idx] = value_compressed.detach()
        
        self.current_step += 1
        
    def query_memory(self, current_context, k=2, temperature=0.1):
        """
        Top-k参照を取得（Phase 2: Multi-head attention）
        
        Args:
            current_context: [B, C, H, W]
            k: Top-k数
            temperature: Softmax温度
            
        Returns:
            attn_weights: [B, k]
            topk_indices: [B, k]
            valid: bool
        """
        valid_refs = self.memory_initialized.sum().item()
        
        if valid_refs == 0 or self.current_step == 0:
            B = current_context.size(0)
            dummy_weights = torch.zeros(B, 1, device=current_context.device)
            dummy_indices = torch.zeros(B, 1, dtype=torch.long, device=current_context.device)
            return dummy_weights, dummy_indices, False
        
        B, C, H, W = current_context.shape
        
        # Query生成
        query = self.query_proj(current_context).mean(dim=[2, 3], keepdim=True)
        
        # 有効なキー取得
        valid_keys = self.memory_keys[:, self.memory_initialized]
        
        # Multi-head attention style similarity
        query_norm = F.normalize(query, dim=1)
        keys_norm = F.normalize(valid_keys, dim=2)
        
        # Cosine similarity
        similarities = (query_norm.unsqueeze(1) * keys_norm).sum(dim=2).squeeze(-1).squeeze(-1)
        
        # Top-k選択
        k = min(k, valid_refs)
        topk_values, topk_indices = torch.topk(similarities, k, dim=1)
        
        # Temperature scaling
        attn_weights = F.softmax(topk_values / temperature, dim=1)
        
        return attn_weights, topk_indices, True
        
    def retrieve_values(self, topk_indices, target_size):
        """
        Top-kインデックスからValueを取得・復元（Phase 2の新機能）
        
        Args:
            topk_indices: [B, k] インデックス
            target_size: (H, W) 復元先の解像度
            
        Returns:
            ref_contexts: [B, k, C, H, W] 復元されたコンテキスト
        """
        if not self.enable_value_storage:
            return None
        
        B, k = topk_indices.shape
        H, W = target_size
        
        # インデックスからValue取得
        # memory_values: [1, max_refs, C, value_res, value_res]
        ref_contexts = []
        
        for b in range(B):
            batch_refs = []
            for i in range(k):
                idx = topk_indices[b, i].item()
                if idx < self.max_refs and self.memory_initialized[idx]:
                    value = self.memory_values[0:1, idx:idx+1]  # [1, 1, C, value_res, value_res]
                    value = value.squeeze(1)  # [1, C, value_res, value_res]
                    
                    # 目標解像度にアップサンプル
                    value_upsampled = F.interpolate(value, size=(H, W), mode='bilinear', align_corners=False)
                    
                    # デコード
                    value_decoded = self.value_decoder(value_upsampled)
                    batch_refs.append(value_decoded)
            
            if len(batch_refs) > 0:
                ref_contexts.append(torch.stack(batch_refs, dim=1))  # [1, k, C, H, W]
        
        if len(ref_contexts) > 0:
            return torch.cat(ref_contexts, dim=0)  # [B, k, C, H, W]
        
        return None
        
    def fuse_references(self, current_context, ref_contexts, attn_weights):
        """
        参照コンテキストを現在のコンテキストと統合（改良版）
        
        Args:
            current_context: [B, C, H, W]
            ref_contexts: [B, k, C, H, W] or None
            attn_weights: [B, k]
            
        Returns:
            fused: [B, C, H, W]
        """
        if ref_contexts is None or ref_contexts.size(1) == 0:
            return current_context
        
        B, k, C, H, W = ref_contexts.shape
        
        # Weighted aggregation
        weighted_ref = (ref_contexts * attn_weights.view(B, k, 1, 1, 1)).sum(dim=1)  # [B, C, H, W]
        
        # Enhanced fusion with residual connection
        concat_features = torch.cat([current_context, weighted_ref], dim=1)
        fusion_features = self.fusion_net(concat_features)
        
        # Gated combination
        gate = self.fusion_gate(concat_features)
        fused = gate * fusion_features + (1 - gate) * current_context
        
        return fused
    
    def forward(self, current_context, k=2, temperature=0.1):
        """
        Complete forward pass with memory query and fusion
        
        Args:
            current_context: [B, C, H, W]
            k: Top-k参照数
            temperature: Softmax温度
            
        Returns:
            fused_context: [B, C, H, W]
        """
        # Query memory
        attn_weights, topk_indices, valid = self.query_memory(current_context, k, temperature)
        
        if not valid or not self.enable_value_storage:
            return current_context
        
        # Retrieve values
        H, W = current_context.shape[2:]
        ref_contexts = self.retrieve_values(topk_indices, (H, W))
        
        # Fuse
        fused = self.fuse_references(current_context, ref_contexts, attn_weights)
        
        return fused


class HierarchicalMemoryManager(nn.Module):
    """
    階層間メモリ共有マネージャー（Phase 2の新機能）
    
    s1/s2/s3の3階層間でメモリを共有・伝達する機構
    下位階層の情報を上位階層に効率的に伝達
    """
    
    def __init__(self, context_dim=640):
        super().__init__()
        self.context_dim = context_dim
        
        # 階層間変換（s1 -> s2, s2 -> s3）
        self.s1_to_s2_adapter = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 1),
            nn.GELU(),
        )
        
        self.s2_to_s3_adapter = nn.Sequential(
            nn.Conv2d(context_dim, context_dim, 1),
            nn.GELU(),
        )
        
    def transfer_s1_to_s2(self, memory_bank_s1, memory_bank_s2, scale_factor=2):
        """s1のメモリをs2に転送（アップサンプリング）"""
        if not hasattr(memory_bank_s1, 'memory_values') or not memory_bank_s1.enable_value_storage:
            return
        
        # s1の有効なメモリをs2にコピー
        valid_mask_s1 = memory_bank_s1.memory_initialized
        if valid_mask_s1.any():
            # TODO: 実装の詳細
            pass
    
    def transfer_s2_to_s3(self, memory_bank_s2, memory_bank_s3, scale_factor=2):
        """s2のメモリをs3に転送（アップサンプリング）"""
        if not hasattr(memory_bank_s2, 'memory_values') or not memory_bank_s2.enable_value_storage:
            return
        
        # s2の有効なメモリをs3にコピー
        valid_mask_s2 = memory_bank_s2.memory_initialized
        if valid_mask_s2.any():
            # TODO: 実装の詳細
            pass
