# Phase 2 Implementation - Full Multi-Reference with Value Storage

## 🎯 Phase 2 の主要機能

### ✅ 実装完了項目

1. **FullContextMemoryBank** (`src/layers/multi_ref_phase2.py`)
   - Value保存機能の完全実装
   - 低解像度での効率的な保存・復元
   - Multi-head attention style の類似度計算
   - 改良されたfusion network

2. **HPCM_MultiRef_Phase2** (`src/models/multiref/phase2/hpcm_base_phase2.py`)
   - s1/s2/s3全階層へのMulti-Reference適用
   - 階層ごとに最適化されたメモリバンク設定
   - 階層間メモリ共有の基本実装

3. **HierarchicalMemoryManager**
   - s1→s2、s2→s3の階層間メモリ転送
   - 下位階層の情報を上位階層に効率的に伝達

## 📊 Phase 1 からの主な改善

| 機能 | Phase 1 | Phase 2 |
|------|---------|---------|
| **適用階層** | s3のみ | s1/s2/s3全階層 |
| **Value保存** | 簡易実装（キャッシュ） | 完全実装（固定バッファ） |
| **Value解像度** | 動的 | 固定（8x8など、設定可能） |
| **Fusion機構** | シンプルなGate | 改良版（Residual + Gate） |
| **階層間共有** | なし | あり（オプション） |
| **メモリ管理** | deque風 | 固定バッファ（効率的） |

## 🔧 使用方法

### 基本的な使い方

```python
from src.models.multiref.phase2 import HPCM_MultiRef_Phase2

# Phase 2: 全階層Multi-Reference
model = HPCM_MultiRef_Phase2(
    M=320,
    N=256,
    enable_multiref=True,
    max_refs_s1=2,      # s1: 2ステップなので控えめ
    max_refs_s2=3,      # s2: 4ステップ
    max_refs_s3=4,      # s3: 8ステップなので多め
    topk_refs_s1=1,     # s1: Top-1
    topk_refs_s2=2,     # s2: Top-2
    topk_refs_s3=2,     # s3: Top-2
    value_resolution=8, # Value保存解像度
    enable_hierarchical_transfer=True  # 階層間共有
).cuda()

# Forward
x = torch.randn(1, 3, 256, 256).cuda()
output = model(x)
x_hat = output['x_hat']
```

### メモリ効率重視の設定

```python
# 軽量版（メモリ節約）
model = HPCM_MultiRef_Phase2(
    max_refs_s1=1,
    max_refs_s2=2,
    max_refs_s3=3,
    compress_ratio=8,      # より高圧縮
    value_resolution=4,    # より低解像度
)
```

### 精度重視の設定

```python
# 高精度版
model = HPCM_MultiRef_Phase2(
    max_refs_s1=3,
    max_refs_s2=4,
    max_refs_s3=6,
    topk_refs_s1=2,
    topk_refs_s2=3,
    topk_refs_s3=3,
    compress_ratio=2,      # 低圧縮
    value_resolution=16,   # 高解像度
)
```

## 📈 期待効果

### Phase 1 との比較

| 指標 | Phase 1 | Phase 2 | 改善 |
|------|---------|---------|------|
| **Rate削減** | 2-3% | 4-6% | +100% |
| **PSNR向上** | +0.1dB | +0.2-0.3dB | +200% |
| **メモリ増** | +5% | +15-20% | +3倍 |
| **計算時間増** | +10-15% | +25-30% | +2倍 |

### Baseline との比較

| 指標 | Baseline | Phase 2 | 改善 |
|------|----------|---------|------|
| **Rate (bpp)** | 1.0 | 0.94-0.96 | ↓4-6% |
| **PSNR** | 32.0dB | 32.2-32.3dB | ↑0.2-0.3dB |
| **BD-rate** | 0% | -5~-8% | -5~-8% |

## 🔬 アーキテクチャ詳細

### FullContextMemoryBank

```
Input Context [B, C, H, W]
    ↓
┌───────────────────────────────────┐
│ Key Encoder (圧縮)                 │
│   Conv2d(C, C//4) + GroupNorm      │
│   → Global Average Pooling         │
│   Result: [B, C//4, 1, 1]          │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Value Encoder (保存用)             │
│   Depthwise Conv + Conv1x1 + GELU │
│   → Adaptive Pooling (8x8)        │
│   Result: [B, C, 8, 8]             │
└───────────────────────────────────┘
    ↓
    Store in Memory Buffer
    
Query Time:
    ↓
┌───────────────────────────────────┐
│ Query Generation                   │
│   Conv2d + GELU + Global Pool      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Cosine Similarity (Top-k)          │
│   query_norm · keys_norm           │
│   → Top-k Selection                │
│   → Softmax with Temperature       │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Value Retrieval                    │
│   Fetch from Memory [B, k, C, 8, 8]│
│   → Interpolate to [B, k, C, H, W] │
│   → Value Decoder                  │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ Weighted Fusion                    │
│   weighted_ref = Σ(ref_i × w_i)   │
│   concat = [current, weighted_ref] │
│   fusion = FusionNet(concat)       │
│   gate = Sigmoid(Gate(concat))     │
│   output = gate·fusion + (1-gate)·current │
└───────────────────────────────────┘
```

### 全階層適用フロー

```
s1階層 [H/64, W/64, 2steps]:
  初期化: Memory Bank s1
  Step 0: Baseline処理
  Step 1: + Multi-Reference (Top-1)
  → メモリ蓄積

s2階層 [H/32, W/32, 4steps]:
  初期化: Memory Bank s2
  (オプション) s1からメモリ転送
  Step 0: Baseline処理
  Step 1-3: + Multi-Reference (Top-2)
  → メモリ蓄積

s3階層 [H/16, W/16, 8steps]:
  初期化: Memory Bank s3
  (オプション) s2からメモリ転送
  Step 0-1: Baseline処理
  Step 2-7: + Multi-Reference (Top-2)
  → 最終出力
```

## 🧪 実験プロトコル

### Phase 1 vs Phase 2 比較

```python
# Phase 1
model_p1 = HPCM_MultiRef_Phase1(enable_multiref=True)

# Phase 2
model_p2 = HPCM_MultiRef_Phase2(enable_multiref=True)

# 評価
for model, name in [(model_p1, "Phase1"), (model_p2, "Phase2")]:
    bpp, psnr = evaluate_on_kodak(model)
    print(f"{name}: BPP={bpp:.4f}, PSNR={psnr:.2f}dB")
```

### アブレーションスタディ

1. **階層ごとの効果**
   ```python
   # s3のみ（Phase 1相当）
   model_s3 = Phase2(max_refs_s1=0, max_refs_s2=0, max_refs_s3=4)
   
   # s2+s3
   model_s2s3 = Phase2(max_refs_s1=0, max_refs_s2=3, max_refs_s3=4)
   
   # s1+s2+s3（Full Phase 2）
   model_full = Phase2(max_refs_s1=2, max_refs_s2=3, max_refs_s3=4)
   ```

2. **Value解像度の影響**
   ```python
   for res in [4, 8, 16]:
       model = Phase2(value_resolution=res)
       evaluate(model)
   ```

3. **階層間共有の効果**
   ```python
   model_no_transfer = Phase2(enable_hierarchical_transfer=False)
   model_with_transfer = Phase2(enable_hierarchical_transfer=True)
   ```

## 📝 実装ノート

### Phase 2の制約事項

1. **decompress未完成**: `decompress_hpcm`は基本構造のみ
2. **階層間転送**: `HierarchicalMemoryManager`は基本実装（詳細は要拡張）
3. **メモリ使用量**: Phase 1の約3倍（Value保存のため）

### パフォーマンスチューニング

```python
# メモリ削減優先
config_mem = {
    'compress_ratio': 8,
    'value_resolution': 4,
    'max_refs_s1': 1,
    'max_refs_s2': 2,
    'max_refs_s3': 3,
}

# 速度優先
config_speed = {
    'topk_refs_s1': 1,
    'topk_refs_s2': 1,
    'topk_refs_s3': 2,
    'enable_hierarchical_transfer': False,
}

# 精度優先
config_quality = {
    'compress_ratio': 2,
    'value_resolution': 16,
    'max_refs_s1': 3,
    'max_refs_s2': 4,
    'max_refs_s3': 6,
    'topk_refs_s1': 2,
    'topk_refs_s2': 3,
    'topk_refs_s3': 3,
}
```

## 🚀 次のステップ

### Phase 3への移行

Phase 2で効果が確認できたら:
- [ ] MLIC++のLinearGlobalInterContext統合
- [ ] Linear Attentionで計算量O(N²)→O(N)に削減
- [ ] より洗練されたcross-layer attention

### 実験・評価タスク

- [ ] Kodak/CLIC/TecnickでのRD曲線
- [ ] Phase 1 vs Phase 2のBD-rate計算
- [ ] 階層別の寄与度分析
- [ ] メモリ・計算時間の詳細プロファイリング
- [ ] 異なる画像タイプでの効果検証

## 📚 参考

- Phase 1: `README_MULTIREF.md`, `PHASE1_SUMMARY.md`
- Phase 2テスト: `examples/test_phase2.py`
- MLIC++論文: `/workspace/LIC-HPCM-MultiRef/MLIC/2307.15421v11-5.pdf`

---

**Phase 2実装完了！🎉**

全階層へのMulti-Reference適用により、より大きな性能向上が期待できます。
