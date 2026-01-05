"""
Multi-Reference Context Memory Bank for HPCM
Phase 1: Lightweight implementation with compressed keys
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class LightweightContextMemoryBank(nn.Module):
    """
    軽量版履歴参照メモリバンク
    
    特徴:
    - 圧縮されたキーのみ保存してメモリ効率化
    - Top-k選択で最も関連性の高い過去ステップを参照
    - Global Average Poolingで空間情報を圧縮
    
    Args:
        context_dim (int): コンテキストの次元数（デフォルト: 640 = M*2）
        max_refs (int): 保持する最大参照数
        compress_ratio (int): キー圧縮率（4なら 640->160）
        num_heads (int): アテンションヘッド数（将来の拡張用）
    """
    def __init__(self, context_dim=640, max_refs=4, compress_ratio=4, num_heads=8):
        super().__init__()
        self.max_refs = max_refs
        self.key_dim = context_dim // compress_ratio
        self.context_dim = context_dim
        self.num_heads = num_heads
        
        # キー圧縮層（軽量化のため1x1 conv + GroupNorm）
        self.key_encoder = nn.Sequential(
            nn.Conv2d(context_dim, self.key_dim, 1, bias=False),
            nn.GroupNorm(min(8, self.key_dim), self.key_dim),
        )
        
        # クエリ生成層
        self.query_proj = nn.Conv2d(context_dim, self.key_dim, 1)
        
        # Value投影層（実際のコンテキスト特徴を調整）
        self.value_proj = nn.Conv2d(context_dim, context_dim, 1)
        
        # Fusion gate（参照情報と現在情報の統合）
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(context_dim * 2, context_dim, 1),
            nn.Sigmoid()
        )
        
        # メモリバッファ（訓練時はバッチごとにリセット）
        # 固定サイズバッファで deque の代わりに実装
        self.register_buffer('memory_keys', torch.zeros(1, max_refs, self.key_dim, 1, 1))
        self.register_buffer('memory_initialized', torch.zeros(max_refs, dtype=torch.bool))
        self.current_step = 0
        
        # Value保存用（Phase 1では簡易実装、Phase 2で拡張）
        self.value_cache = []
        
    def reset(self):
        """エピソード（画像）ごとにメモリをリセット"""
        self.current_step = 0
        self.memory_keys.zero_()
        self.memory_initialized.zero_()
        self.value_cache = []
        
    def add_to_memory(self, context, store_value=True):
        """
        現ステップのコンテキストをメモリに追加
        
        Args:
            context: [B, C, H, W] 現在のコンテキスト特徴
            store_value: Value（実際のコンテキスト）も保存するか
        """
        B, C, H, W = context.shape
        
        # Global Average Pooling でキー圧縮（空間次元を削減）
        key = self.key_encoder(context).mean(dim=[2, 3], keepdim=True)  # [B, key_dim, 1, 1]
        
        # Circular buffer 更新
        idx = self.current_step % self.max_refs
        self.memory_keys[:, idx] = key.detach()  # 勾配を切断
        self.memory_initialized[idx] = True
        
        # Value保存（オプション、Phase 2で本格実装）
        if store_value:
            # メモリ節約のため低解像度化して保存
            value_compressed = F.adaptive_avg_pool2d(
                self.value_proj(context).detach(), 
                (max(1, H//4), max(1, W//4))
            )
            
            # キャッシュサイズ制限
            if len(self.value_cache) >= self.max_refs:
                self.value_cache.pop(0)
            self.value_cache.append(value_compressed)
        
        self.current_step += 1
        
    def query_memory(self, current_context, k=2, temperature=0.1):
        """
        Top-k 参照を取得
        
        Args:
            current_context: [B, C, H, W] 現在のコンテキスト
            k: 取得する参照数
            temperature: softmax の温度パラメータ（小さいほどシャープ）
            
        Returns:
            attn_weights: [B, k] アテンション重み
            topk_indices: [B, k] Top-k インデックス
            valid: bool メモリに有効な参照があるか
        """
        # 有効なメモリ数をチェック
        valid_refs = self.memory_initialized.sum().item()
        
        if valid_refs == 0 or self.current_step == 0:
            # メモリが空の場合
            B = current_context.size(0)
            dummy_weights = torch.zeros(B, 1, device=current_context.device)
            dummy_indices = torch.zeros(B, 1, dtype=torch.long, device=current_context.device)
            return dummy_weights, dummy_indices, False
        
        B, C, H, W = current_context.shape
        
        # クエリ生成（spatial average）
        query = self.query_proj(current_context).mean(dim=[2, 3], keepdim=True)  # [B, key_dim, 1, 1]
        
        # 有効なキーのみ取得
        valid_keys = self.memory_keys[:, self.memory_initialized]  # [B, valid_refs, key_dim, 1, 1]
        
        # Cosine similarity 計算（正規化してドット積）
        query_norm = F.normalize(query, dim=1)  # [B, key_dim, 1, 1]
        keys_norm = F.normalize(valid_keys, dim=2)  # [B, valid_refs, key_dim, 1, 1]
        
        # [B, valid_refs]
        similarities = (query_norm.unsqueeze(1) * keys_norm).sum(dim=2).squeeze(-1).squeeze(-1)
        
        # Top-k 選択（k は有効参照数以下に制限）
        k = min(k, valid_refs)
        topk_values, topk_indices = torch.topk(similarities, k, dim=1)
        
        # Soft attention weights（temperature scaling）
        attn_weights = F.softmax(topk_values / temperature, dim=1)
        
        return attn_weights, topk_indices, True
        
    def retrieve_and_fuse(self, current_context, attn_weights, topk_indices, valid):
        """
        参照コンテキストを取得して現在のコンテキストと統合
        
        Args:
            current_context: [B, C, H, W] 現在のコンテキスト
            attn_weights: [B, k] アテンション重み
            topk_indices: [B, k] Top-k インデックス
            valid: bool メモリが有効か
            
        Returns:
            fused_context: [B, C, H, W] 統合されたコンテキスト
        """
        if not valid or len(self.value_cache) == 0:
            # 参照がない場合は元のコンテキストをそのまま返す
            return current_context
        
        B, C, H, W = current_context.shape
        k = topk_indices.size(1)
        
        # 参照コンテキストを取得
        ref_contexts = []
        for b in range(B):
            batch_refs = []
            for i in range(k):
                idx = topk_indices[b, i].item()
                if idx < len(self.value_cache):
                    # キャッシュから取得してリサイズ
                    ref = self.value_cache[idx]
                    if ref.size(2) != H or ref.size(3) != W:
                        ref = F.interpolate(ref, size=(H, W), mode='bilinear', align_corners=False)
                    batch_refs.append(ref[b:b+1])
            
            if len(batch_refs) > 0:
                ref_contexts.append(torch.stack(batch_refs, dim=1))  # [1, k, C, H, W]
        
        if len(ref_contexts) == 0:
            return current_context
        
        # バッチ次元で結合
        ref_stack = torch.cat(ref_contexts, dim=0)  # [B, k, C, H, W]
        
        # Weighted sum of references
        weighted_ref = (ref_stack * attn_weights.view(B, k, 1, 1, 1)).sum(dim=1)  # [B, C, H, W]
        
        # Gated fusion（学習可能なゲートで統合割合を調整）
        gate = self.fusion_gate(torch.cat([current_context, weighted_ref], dim=1))
        fused = gate * weighted_ref + (1 - gate) * current_context
        
        return fused
    
    def forward(self, current_context, k=2, apply_fusion=True):
        """
        便利なforward関数（query + retrieve を一度に実行）
        
        Args:
            current_context: [B, C, H, W]
            k: Top-k 参照数
            apply_fusion: 統合を適用するか（Falseの場合は重みのみ返す）
            
        Returns:
            fused_context or (attn_weights, topk_indices)
        """
        attn_weights, topk_indices, valid = self.query_memory(current_context, k=k)
        
        if apply_fusion:
            return self.retrieve_and_fuse(current_context, attn_weights, topk_indices, valid)
        else:
            return attn_weights, topk_indices, valid


class ContextMemoryBankV2(nn.Module):
    """
    Phase 2用の拡張版（Value保存機能強化）
    Phase 1で効果が確認されたら実装
    """
    def __init__(self, context_dim=640, max_refs=4, compress_ratio=4):
        super().__init__()
        # TODO: Phase 2 で実装
        raise NotImplementedError("Phase 2 implementation")
