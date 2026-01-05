"""
Phase 1実装の完全性チェックとクイックスタートガイド

実行: python examples/quick_start_phase1.py
"""

import sys
import os
sys.path.insert(0, '/workspace/LIC-HPCM-MultiRef')


def check_installation():
    """必要なモジュールが正しくインストールされているか確認"""
    print("="*70)
    print("Phase 1 Installation Check")
    print("="*70)
    
    checks = []
    
    # 1. PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        checks.append(True)
    except ImportError:
        print("✗ PyTorch not found")
        checks.append(False)
    
    # 2. Multi-Reference Layer
    try:
        from src.layers.multi_ref import LightweightContextMemoryBank
        print("✓ Multi-Reference Layer loaded")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Multi-Reference Layer not found: {e}")
        checks.append(False)
    
    # 3. Phase 1 Model
    try:
        from src.models.multiref.phase1 import HPCM_MultiRef_Phase1
        print("✓ Phase 1 Model loaded")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Phase 1 Model not found: {e}")
        checks.append(False)
    
    # 4. Base HPCM components
    try:
        from src.models.HPCM_Base import g_a, g_s, h_a, h_s
        print("✓ Base HPCM components loaded")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Base HPCM components not found: {e}")
        checks.append(False)
    
    # 5. Layers
    try:
        from src.layers import PConvRB, conv1x1
        print("✓ Custom layers loaded")
        checks.append(True)
    except ImportError as e:
        print(f"✗ Custom layers not found: {e}")
        checks.append(False)
    
    print("\n" + "="*70)
    if all(checks):
        print("✅ All checks passed! Phase 1 is ready to use.")
        return True
    else:
        print(f"⚠ {sum(not c for c in checks)}/{len(checks)} checks failed.")
        return False


def quick_test():
    """クイックテスト"""
    print("\n" + "="*70)
    print("Quick Functionality Test")
    print("="*70)
    
    try:
        import torch
        from src.models.multiref.phase1 import HPCM_MultiRef_Phase1
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")
        
        # 小さいサイズでテスト
        print("\n1. Creating model...")
        model = HPCM_MultiRef_Phase1(
            M=320,
            N=256,
            enable_multiref=True,
            max_refs=4,
            topk_refs=2
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {param_count:,}")
        
        # Forward test
        print("\n2. Testing forward pass...")
        x = torch.randn(1, 3, 128, 128).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(x, training=False)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output['x_hat'].shape}")
        print(f"   Y likelihood shape: {output['likelihoods']['y'].shape}")
        print(f"   Z likelihood shape: {output['likelihoods']['z'].shape}")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"\n3. GPU Memory:")
            print(f"   Allocated: {memory_allocated:.2f} MB")
            print(f"   Reserved: {memory_reserved:.2f} MB")
        
        print("\n✅ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_examples():
    """使用例を表示"""
    print("\n" + "="*70)
    print("Usage Examples")
    print("="*70)
    
    print("""
1. Basic Usage (Multi-Reference enabled):
   
   from src.models.multiref.phase1 import HPCM_MultiRef_Phase1
   
   model = HPCM_MultiRef_Phase1(
       M=320,
       N=256,
       enable_multiref=True,
       max_refs=4,
       topk_refs=2
   ).cuda()
   
   output = model(x)
   x_hat = output['x_hat']

2. Baseline Comparison (Multi-Reference disabled):
   
   model_baseline = HPCM_MultiRef_Phase1(
       M=320,
       N=256,
       enable_multiref=False  # Baseline mode
   ).cuda()

3. Hyperparameter Tuning:
   
   # Memory-efficient configuration
   model = HPCM_MultiRef_Phase1(
       max_refs=2,           # Fewer references
       topk_refs=1,          # Single best reference
       compress_ratio=8      # Higher compression
   ).cuda()
   
   # Accuracy-oriented configuration
   model = HPCM_MultiRef_Phase1(
       max_refs=6,           # More references
       topk_refs=3,          # Multiple references
       compress_ratio=2      # Less compression
   ).cuda()

4. Training:
   
   model.train()
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   
   for images in dataloader:
       output = model(images, training=True)
       loss = compute_rd_loss(output, images)
       loss.backward()
       optimizer.step()

5. Evaluation:
   
   model.eval()
   with torch.no_grad():
       output = model(test_image, training=False)
       psnr = compute_psnr(output['x_hat'], test_image)
       
       # Estimate rate
       y_bpp = -torch.log(output['likelihoods']['y']).sum() / num_pixels
       z_bpp = -torch.log(output['likelihoods']['z']).sum() / num_pixels
       total_bpp = y_bpp + z_bpp

For more examples, see:
  - examples/test_phase1.py          : Comprehensive test suite
  - examples/train_phase1_example.py : Training template
  - README_MULTIREF.md               : Full documentation
""")


def show_next_steps():
    """次のステップを表示"""
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("""
Phase 1 is now ready! Here's what you can do:

📊 1. Run comprehensive tests:
   python examples/test_phase1.py

🎯 2. Compare with baseline:
   - Train Phase 1 model (enable_multiref=True)
   - Train baseline model (enable_multiref=False)
   - Compare RD curves on Kodak/CLIC datasets

📈 3. Hyperparameter search:
   - Try different max_refs: [2, 3, 4, 6]
   - Try different topk_refs: [1, 2, 3]
   - Try different compress_ratio: [2, 4, 8]
   - Try different temperature: [0.05, 0.1, 0.2]

🔬 4. Ablation studies:
   - Effect of memory bank size
   - Effect of Top-k selection
   - Effect of compression ratio
   - Analysis on different image types

📝 5. Prepare for Phase 2:
   - Implement full Value storage
   - Extend to s1/s2 layers
   - Add cross-layer memory sharing

For detailed documentation, see: README_MULTIREF.md
""")


def main():
    """メイン"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "HPCM Multi-Reference Phase 1" + " "*25 + "║")
    print("║" + " "*20 + "Quick Start Guide" + " "*31 + "║")
    print("╚" + "="*68 + "╝")
    
    # インストールチェック
    if not check_installation():
        print("\n⚠ Please fix the installation issues before proceeding.")
        return 1
    
    # クイックテスト
    if not quick_test():
        print("\n⚠ Quick test failed. Please check the error messages.")
        return 1
    
    # 使用例
    show_usage_examples()
    
    # 次のステップ
    show_next_steps()
    
    print("\n" + "="*70)
    print("✅ Phase 1 is ready to use!")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
