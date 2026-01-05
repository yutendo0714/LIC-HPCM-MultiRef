# HPCM × Multi-Reference Implementation

HPCMに履歴参照型エントロピーモデリング（Multi-Reference Memory Bank）を統合した拡張実装。

## 📁 ディレクトリ構造

```
src/
├── layers/
│   └── multi_ref.py                    # Multi-Reference Memory Bank実装
└── models/
    └── multiref/
        ├── __init__.py
        ├── phase1/                      # Phase 1: s3のみ軽量実装
        │   ├── __init__.py
        │   └── hpcm_base_phase1.py
        ├── phase2/                      # Phase 2: 全階層展開（未実装）
        │   └── __init__.py
        └── phase3/                      # Phase 3: Linear Attention統合（未実装）
            └── __init__.py
```

## 🚀 Phase 1: 軽量版Multi-Reference（s3階層のみ）

### 特徴

- ✅ **s3階層のみに適用**: 最も情報量が多い8ステップで効果検証
- ✅ **軽量版実装**: 圧縮されたキーのみ保存（メモリ効率化）
- ✅ **Top-k参照選択**: Cosine類似度でTop-k参照を動的選択
- ✅ **学習可能なGated Fusion**: 参照情報と現在情報の統合割合を自動学習

### 使用方法

```python
from src.models.multiref.phase1 import HPCM_MultiRef_Phase1

# Multi-Reference有効化
model = HPCM_MultiRef_Phase1(
    M=320, 
    N=256,
    enable_multiref=True,   # Multi-Reference機能ON
    max_refs=4,             # 最大4個の過去ステップを保持
    topk_refs=2,            # Top-2参照を選択
    compress_ratio=4,       # キー圧縮率（640→160）
    temperature=0.1         # Softmax温度
)

# Baseline比較用（Multi-Reference無効）
model_baseline = HPCM_MultiRef_Phase1(
    M=320, 
    N=256,
    enable_multiref=False   # 既存HPCMと同等
)

# 訓練
output = model(x)
x_hat = output["x_hat"]
likelihoods = output["likelihoods"]

# 圧縮
compressed = model.compress(x)
strings = compressed["strings"]
shape = compressed["shape"]

# 復元（Phase 1では簡易実装）
# decompressed = model.decompress(strings, shape)
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `M` | 320 | Main latent channel数 |
| `N` | 256 | Hyper latent channel数 |
| `enable_multiref` | True | Multi-Reference機能の有効/無効 |
| `max_refs` | 4 | メモリバンクの最大参照保持数 |
| `topk_refs` | 2 | Top-k参照選択数 |
| `compress_ratio` | 4 | キー圧縮率（大きいほど軽量） |
| `temperature` | 0.1 | Softmax温度（小さいほどシャープ） |

### アーキテクチャ

```
s3階層の各ステップ（6回繰り返し）:
  ┌─────────────────────────────────────┐
  │ 1. Spatial Prior Network            │
  │    y_spatial_prior_s3               │
  └─────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────┐
  │ 2. Local Cross-Attention (既存)      │
  │    attn_s3(context, context_next)   │
  └─────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────┐
  │ 3. Multi-Reference Memory Bank       │
  │    ┌──────────────────────────┐     │
  │    │ Query: 現在のcontext      │     │
  │    │ Key: 過去のcontext (圧縮) │     │
  │    │ Value: 過去のcontext      │     │
  │    └──────────────────────────┘     │
  │    ↓ Top-k Selection (k=2)          │
  │    ↓ Gated Fusion                   │
  └─────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────┐
  │ 4. Add to Memory Bank                │
  │    memory_bank_s3.add_to_memory()   │
  └─────────────────────────────────────┘
            ↓
        context_next (enhanced)
```

## 🔬 実験設定

### A/Bテスト

```python
# Baseline
model_baseline = HPCM_MultiRef_Phase1(enable_multiref=False)

# Phase 1
model_phase1 = HPCM_MultiRef_Phase1(enable_multiref=True, max_refs=4, topk_refs=2)

# 評価
# - RD curve (Kodak, CLIC Pro Valid, Tecnick)
# - BD-rate vs baseline
# - 推論時間、メモリ使用量
```

### ハイパーパラメータ探索

```python
# メモリバンクサイズ
for max_refs in [2, 3, 4, 6]:
    model = HPCM_MultiRef_Phase1(max_refs=max_refs)
    
# Top-k参照数
for topk_refs in [1, 2, 3]:
    model = HPCM_MultiRef_Phase1(topk_refs=topk_refs)
    
# 圧縮率
for compress_ratio in [2, 4, 8]:
    model = HPCM_MultiRef_Phase1(compress_ratio=compress_ratio)
```

## 📊 期待効果

| 指標 | 期待値 |
|-----|-------|
| **Rate削減** | 1-3% (bpp) |
| **PSNR向上** | +0.1dB (同一rate) |
| **メモリ増加** | +5% |
| **計算時間増加** | +10-15% |

## 🔜 Phase 2（予定）

- Value保存機能の強化（低解像度で保存→復元）
- s1/s2階層への展開
- 階層間メモリ共有
- decompress_hpcm の完全実装

## 🔜 Phase 3（予定）

- MLIC++のLinearGlobalInterContextを統合
- MultiRefCrossAttentionCell実装
- Linear Attentionでメモリ効率化（O(N²) → O(N)）

## 📝 実装ノート

### Phase 1の制約事項

1. **Decompress未実装**: `decompress_hpcm`はPhase 2で完全実装予定
2. **s3のみ**: s1/s2は既存HPCMと同じ処理
3. **Value簡易保存**: 低解像度化して保存（Phase 2で改善）

### デバッグ情報

モデル初期化時に以下のメッセージが表示されます:

```
[Phase 1] Multi-Reference Memory Bank enabled for s3:
  - max_refs=4, topk_refs=2
  - compress_ratio=4, temperature=0.1
```

### パフォーマンス最適化Tips

1. **メモリ削減**: `compress_ratio`を大きく（4→8）
2. **速度向上**: `topk_refs`を小さく（2→1）
3. **精度重視**: `max_refs`を大きく（4→6）

## 🐛 トラブルシューティング

### OOM（Out of Memory）

```python
# 圧縮率を上げる
model = HPCM_MultiRef_Phase1(compress_ratio=8)

# メモリバンクサイズを減らす
model = HPCM_MultiRef_Phase1(max_refs=2)
```

### 推論速度が遅い

```python
# Top-k数を減らす
model = HPCM_MultiRef_Phase1(topk_refs=1)

# Baselineモードで実行
model = HPCM_MultiRef_Phase1(enable_multiref=False)
```

## 📚 参考文献

- HPCM: [Original paper]
- MLIC++: `/workspace/LIC-HPCM-MultiRef/MLIC/2307.15421v11-5.pdf`
- Multi-Reference Entropy Modeling: ACM Digital Library

## 🤝 貢献

Phase 2/3の実装、実験結果の報告、バグ修正など歓迎します！
