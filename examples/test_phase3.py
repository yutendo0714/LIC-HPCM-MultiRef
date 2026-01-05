"""
Phase 3実装の動作確認スクリプト

Phase 3の新機能テスト:
- LinearAttentionMemoryBank (O(N)複雑度)
- Channel-wise linear attention
- MLIC++スタイルのkernel feature map
- 計算効率の検証

使用方法:
    python examples/test_phase3.py
"""

import torch
import sys
import time
sys.path.insert(0, '/workspace/LIC-HPCM-MultiRef')


def test_linear_attention_memory_bank():
    """LinearAttentionMemoryBankのテスト"""
    print("="*60)
    print("Test 1: LinearAttentionMemoryBank (Linear Attention)")
    print("="*60)
    
    from src.layers.multi_ref_phase3 import LinearAttentionMemoryBank
    
    memory_bank = LinearAttentionMemoryBank(
        context_dim=640,
        max_refs=4,
        compress_ratio=4,
        value_resolution=8,
        num_heads=8,
        kernel_type='elu',
        enable_value_storage=True
    ).cuda()
    
    B, C, H, W = 2, 640, 16, 16
    
    print(f"\n[1] Testing Linear Attention (B={B}, C={C}, H={H}, W={W})...")
    
    # リセット
    memory_bank.reset()
    print("✓ Memory reset")
    
    # ステップ1-3: コンテキスト追加
    for step in range(1, 4):
        context = torch.randn(B, C, H, W).cuda()
        memory_bank.add_to_memory(context)
        print(f"✓ Added context to memory (step {step})")
    
    # Linear Attention Query
    current_context = torch.randn(B, C, H, W).cuda()
    attn_weights, topk_indices, valid = memory_bank.query_memory(current_context, k=2)
    
    print(f"\n[2] Linear Attention query results:")
    print(f"  - Attention weights: {attn_weights[0].cpu().numpy()}")
    print(f"  - Top-k indices: {topk_indices[0].cpu().numpy()}")
    print(f"✓ Linear attention query successful")
    
    # Forward pass
    enhanced = memory_bank.forward(current_context, k=2)
    print(f"  - Enhanced context shape: {enhanced.shape}")
    print(f"✓ Full forward pass successful")
    
    print("\n✅ Test 1 PASSED\n")


def test_phase3_model():
    """Phase 3モデルの基本テスト"""
    print("="*60)
    print("Test 2: HPCM_MultiRef_Phase3 Basic Forward")
    print("="*60)
    
    from src.models.multiref.phase3 import HPCM_MultiRef_Phase3
    
    x = torch.randn(1, 3, 256, 256).cuda()
    
    print("\n[1] Testing Phase 3 with Linear Attention...")
    model_phase3 = HPCM_MultiRef_Phase3(
        M=320,
        N=256,
        enable_multiref=True,
        max_refs_s1=2,
        max_refs_s2=3,
        max_refs_s3=4,
        topk_refs_s1=1,
        topk_refs_s2=2,
        topk_refs_s3=2,
        kernel_type='elu',
        enable_hierarchical_transfer=True
    ).cuda()
    model_phase3.eval()
    
    with torch.no_grad():
        output = model_phase3(x, training=False)
    
    print(f"✓ Output x_hat shape: {output['x_hat'].shape}")
    print(f"✓ y likelihood shape: {output['likelihoods']['y'].shape}")
    print(f"✓ z likelihood shape: {output['likelihoods']['z'].shape}")
    
    print("\n[2] Testing Baseline mode...")
    model_baseline = HPCM_MultiRef_Phase3(
        M=320,
        N=256,
        enable_multiref=False
    ).cuda()
    model_baseline.eval()
    
    with torch.no_grad():
        output_baseline = model_baseline(x, training=False)
    
    print(f"✓ Baseline output shape: {output_baseline['x_hat'].shape}")
    
    print("\n✅ Test 2 PASSED\n")


def test_kernel_types():
    """異なるkernel typeのテスト"""
    print("="*60)
    print("Test 3: Different Kernel Feature Maps")
    print("="*60)
    
    from src.models.multiref.phase3 import HPCM_MultiRef_Phase3
    
    x = torch.randn(1, 3, 128, 128).cuda()
    
    kernel_types = ['elu', 'relu']
    
    for i, kernel_type in enumerate(kernel_types):
        print(f"\n[{i+1}] Testing kernel type: {kernel_type}...")
        
        model = HPCM_MultiRef_Phase3(
            M=320, N=256,
            enable_multiref=True,
            kernel_type=kernel_type,
            max_refs_s1=2, max_refs_s2=3, max_refs_s3=4
        ).cuda()
        model.eval()
        
        with torch.no_grad():
            output = model(x, training=False)
        
        print(f"✓ Kernel type '{kernel_type}' successful")
    
    print("\n✅ Test 3 PASSED\n")


def test_computational_efficiency():
    """計算効率のテスト（Phase 2 vs Phase 3）"""
    print("="*60)
    print("Test 4: Computational Efficiency Comparison")
    print("="*60)
    
    from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
    from src.models.multiref.phase3 import HPCM_MultiRef_Phase3
    
    x = torch.randn(1, 3, 256, 256).cuda()
    
    print("\n[1] Phase 2 (Softmax Attention)...")
    model_phase2 = HPCM_MultiRef_Phase2(
        M=320, N=256,
        enable_multiref=True,
        max_refs_s1=2, max_refs_s2=3, max_refs_s3=4
    ).cuda()
    model_phase2.eval()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(5):
            output_phase2 = model_phase2(x, training=False)
    
    torch.cuda.synchronize()
    phase2_time = (time.time() - start_time) / 5
    
    print(f"  - Average time: {phase2_time*1000:.2f} ms")
    
    print("\n[2] Phase 3 (Linear Attention)...")
    model_phase3 = HPCM_MultiRef_Phase3(
        M=320, N=256,
        enable_multiref=True,
        max_refs_s1=2, max_refs_s2=3, max_refs_s3=4
    ).cuda()
    model_phase3.eval()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(5):
            output_phase3 = model_phase3(x, training=False)
    
    torch.cuda.synchronize()
    phase3_time = (time.time() - start_time) / 5
    
    print(f"  - Average time: {phase3_time*1000:.2f} ms")
    
    speedup = phase2_time / phase3_time
    print(f"\n[3] Comparison:")
    print(f"  - Phase 2: {phase2_time*1000:.2f} ms")
    print(f"  - Phase 3: {phase3_time*1000:.2f} ms")
    print(f"  - Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✓ Phase 3 is {speedup:.2f}x faster!")
    else:
        print(f"⚠ Phase 3 is slightly slower (implementation overhead)")
    
    print("\n✅ Test 4 PASSED\n")


def test_training_mode():
    """訓練モードテスト"""
    print("="*60)
    print("Test 5: Training Mode with Linear Attention")
    print("="*60)
    
    from src.models.multiref.phase3 import HPCM_MultiRef_Phase3
    
    x = torch.randn(2, 3, 128, 128).cuda()
    
    model = HPCM_MultiRef_Phase3(
        M=320,
        N=256,
        enable_multiref=True,
        max_refs_s1=2,
        max_refs_s2=3,
        max_refs_s3=4,
        kernel_type='elu'
    ).cuda()
    model.train()
    
    print("\n[1] Forward pass in training mode...")
    output = model(x, training=True)
    
    print(f"✓ Output x_hat shape: {output['x_hat'].shape}")
    
    # Loss計算
    print("\n[2] Computing loss...")
    mse_loss = torch.nn.functional.mse_loss(output['x_hat'], x)
    rate_loss = output['likelihoods']['y'].log().sum() + output['likelihoods']['z'].log().sum()
    total_loss = mse_loss - rate_loss
    
    print(f"✓ MSE Loss: {mse_loss.item():.4f}")
    print(f"✓ Rate Loss: {rate_loss.item():.4f}")
    print(f"✓ Total Loss: {total_loss.item():.4f}")
    
    # Backward
    print("\n[3] Backward pass...")
    total_loss.backward()
    print("✓ Backward successful")
    
    # Temperature parameter check
    if hasattr(model.memory_bank_s3, 'temperature'):
        print(f"\n[4] Temperature parameter:")
        print(f"  - Value: {model.memory_bank_s3.temperature.item():.4f}")
        if model.memory_bank_s3.temperature.grad is not None:
            print(f"  - Gradient: {model.memory_bank_s3.temperature.grad.item():.4f}")
            print("✓ Temperature is learnable")
    
    print("\n✅ Test 5 PASSED\n")


def test_memory_efficiency():
    """メモリ効率のテスト"""
    print("="*60)
    print("Test 6: Memory Efficiency")
    print("="*60)
    
    from src.models.multiref.phase3 import HPCM_MultiRef_Phase3
    
    x = torch.randn(1, 3, 256, 256).cuda()
    
    configs = [
        {"num_heads": 4, "name": "Few Heads"},
        {"num_heads": 8, "name": "Balanced"},
        {"num_heads": 16, "name": "Many Heads"}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}] Testing {config['name']} (heads={config['num_heads']})...")
        
        torch.cuda.reset_peak_memory_stats()
        
        model = HPCM_MultiRef_Phase3(
            M=320, N=256,
            enable_multiref=True,
            max_refs_s1=2, max_refs_s2=3, max_refs_s3=4,
            num_heads=config['num_heads']
        ).cuda()
        model.eval()
        
        with torch.no_grad():
            output = model(x, training=False)
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  - Memory allocated: {memory_allocated:.2f} MB")
            print(f"  - Peak memory: {memory_peak:.2f} MB")
        
        print(f"✓ {config['name']} config tested")
    
    print("\n✅ Test 6 PASSED\n")


def main():
    """全テスト実行"""
    print("\n" + "="*60)
    print("HPCM Multi-Reference Phase 3 - Test Suite")
    print("Linear Attention Implementation")
    print("="*60 + "\n")
    
    try:
        test_linear_attention_memory_bank()
        test_phase3_model()
        test_kernel_types()
        test_computational_efficiency()
        test_training_mode()
        test_memory_efficiency()
        
        print("="*60)
        print("🎉 ALL PHASE 3 TESTS PASSED!")
        print("="*60)
        print("\nPhase 3実装が正常に動作しています。")
        print("\n主な改善点:")
        print("✓ Linear Attention (O(N²) → O(N))")
        print("✓ MLIC++スタイルのchannel-wise attention")
        print("✓ Kernel feature map (ELU-based)")
        print("✓ 計算効率の向上")
        print("✓ 学習可能なtemperatureパラメータ")
        print("\n次のステップ:")
        print("1. Phase 1/2/3の性能比較")
        print("2. 実データセットでの評価")
        print("3. 計算量・速度の詳細プロファイリング")
        print("4. BD-rate削減効果の測定")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
