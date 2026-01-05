# テスト実行ガイド

## 🚀 Quick Start

### pipenv環境の準備

```bash
cd /workspace/LIC-HPCM-MultiRef
pipenv install
```

### 利用可能なコマンド

```bash
# 全Phase統合テスト（最速・推奨）
pipenv run quick-test

# Phase 1のみテスト
pipenv run test-phase1

# Phase 2のみテスト
pipenv run test-phase2

# Phase 3のみテスト
pipenv run test-phase3

# 全Phase詳細テスト（時間がかかります）
pipenv run test-all

# Phase 1クイックスタート
pipenv run quick-start
```

## 📋 各テストの内容

### quick-test（推奨）
- 全Phase (1, 2, 3) の基本動作確認
- 各Phaseで1回のforward pass
- 実行時間: 約30秒～1分
- 環境チェックとサマリー付き

**出力例:**
```
============================================================
HPCM Multi-Reference - Quick Test Suite
Testing all phases (1, 2, 3)
============================================================

Environment Check
- PyTorch version: 2.9.1+cu128
- CUDA available: True

Phase 1: ✅ PASSED
Phase 2: ✅ PASSED
Phase 3: ✅ PASSED

🎉 ALL TESTS PASSED!
```

### test-phase1
- Phase 1の包括的テスト
- 5つのテストケース:
  1. Basic forward pass
  2. Memory bank functionality
  3. Training mode
  4. Compression/decompression
  5. Parameter variations
- 実行時間: 約2-3分

### test-phase2
- Phase 2の包括的テスト
- 5つのテストケース:
  1. Full memory bank with value storage
  2. Phase 2 model basic forward
  3. Multi-reference on all scales
  4. Training mode
  5. Memory efficiency comparison
- 実行時間: 約3-5分

### test-phase3
- Phase 3の包括的テスト
- 6つのテストケース:
  1. Linear attention memory bank
  2. Phase 3 model basic forward
  3. Different kernel types
  4. Computational efficiency (vs Phase 2)
  5. Training mode
  6. Memory efficiency
- 実行時間: 約3-5分

### test-all
- 全Phaseの詳細テストを順次実行
- test-phase1 → test-phase2 → test-phase3
- 実行時間: 約8-13分

## 🔍 個別テスト実行

### Pythonから直接実行

```bash
# Phase 1
python examples/test_phase1.py

# Phase 2
python examples/test_phase2.py

# Phase 3
python examples/test_phase3.py

# 統合テスト
python examples/quick_test_all_phases.py
```

## ⚙️ 環境要件

### 必須
- Python 3.10
- PyTorch with CUDA support
- CUDA対応GPU (推奨: RTX 3090以上)

### 推奨
- VRAM: 8GB以上
- メモリ: 16GB以上

### CUDAが利用できない場合
テストはCPUモードで実行されますが、以下の制限があります:
- 実行速度が大幅に低下
- 一部のテストでメモリ不足の可能性

## 📊 テスト結果の見方

### 成功時
```
✅ Phase X test PASSED
```

### 失敗時
```
❌ Phase X test FAILED
Error: <エラー詳細>
Traceback: <スタックトレース>
```

### 一般的なエラーと対処法

#### 1. CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**対処法:**
- より小さいバッチサイズを使用
- より小さい画像サイズでテスト
- 他のGPUプロセスを終了

#### 2. Module not found
```
ModuleNotFoundError: No module named 'torch'
```
**対処法:**
```bash
pipenv install
```

#### 3. Shape mismatch
```
Shape mismatch, can't divide axis of length X in chunks of Y
```
**対処法:**
- 画像サイズを256x256以上に設定
- window_sizeで割り切れるサイズを使用

## 🎯 CI/CDでの使用

### GitHub Actions例

```yaml
name: Test HPCM Multi-Reference

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install pipenv
        pipenv install
    
    - name: Run quick test
      run: pipenv run quick-test
```

## 📝 トラブルシューティング

### pipenvコマンドが見つからない
```bash
pip install --user pipenv
```

### virtual environmentが作成されない
```bash
pipenv --python 3.10
pipenv install
```

### テストが途中で止まる
- Ctrl+Cで中断
- `pipenv run quick-test`で軽量テストを試す

## 🔧 開発者向け

### 新しいテストの追加

1. `examples/`に新しいテストファイルを作成
2. `Pipfile`の`[scripts]`セクションに追加:
   ```toml
   [scripts]
   test-mytest = "python examples/test_mytest.py"
   ```
3. テスト実行:
   ```bash
   pipenv run test-mytest
   ```

### デバッグモード

```python
# テストファイルの先頭に追加
import torch
torch.autograd.set_detect_anomaly(True)
```

## 📚 関連ドキュメント

- [README_MULTIREF.md](README_MULTIREF.md) - Multi-Reference実装の総合ガイド
- [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) - Phase 1詳細
- [PHASE2_SUMMARY.md](PHASE2_SUMMARY.md) - Phase 2詳細
- [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md) - Phase 3詳細
- [ALL_PHASES_SUMMARY.md](ALL_PHASES_SUMMARY.md) - 全Phase比較

---

**Quick Start:**
```bash
pipenv install
pipenv run quick-test
```

**Full Test:**
```bash
pipenv run test-all
```
