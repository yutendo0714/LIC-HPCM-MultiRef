"""
Phase 2実装の動作確認スクリプト

Phase 2の新機能テスト:
- FullContextMemoryBank (Value保存完全実装)
- s1/s2/s3全階層へのMulti-Reference適用
- 階層間メモリ共有

使用方法:
    python examples/test_phase2.py
"""

import torch
import sys
sys.path.insert(0, '/workspace/LIC-HPCM-MultiRef')


def test_full_memory_bank():
    """FullContextMemoryBankのテスト"""
    print("="*60)
    print("Test 1: FullContextMemoryBank (Value Storage)")
    print("="*60)
    
    from src.layers.multi_ref_phase2 import FullContextMemoryBank
    
    memory_bank = FullContextMemoryBank(
        context_dim=640,
        max_refs=4,
        compress_ratio=4,
        value_resolution=8,
        num_heads=8,
        enable_value_storage=True
    ).cuda()
    
    B, C, H, W = 2, 640, 16, 16
    
    print(f"\n[1] Testing with value storage (B={B}, C={C}, H={H}, W={W})...")
    
    # リセット
    memory_bank.reset()
    print("✓ Memory reset")
    
    # ステップ1-3: コンテキスト追加
    for step in range(1, 4):
        context = torch.randn(B, C, H, W).cuda()
        memory_bank.add_to_memory(context)
        print(f"✓ Added context to memory (step {step})")
    
    # Value取得テスト
    current_context = torch.randn(B, C, H, W).cuda()
    attn_weights, topk_indices, valid = memory_bank.query_memory(current_context, k=2)
    
    print(f"\n[2] Query and retrieve values:")
    print(f"  - Attention weights: {attn_weights[0].cpu().numpy()}")
    print(f"  - Top-k indices: {topk_indices[0].cpu().numpy()}")
    
    # Value復元
    ref_contexts = memory_bank.retrieve_values(topk_indices, (H, W))
    if ref_contexts is not None:
        print(f"  - Retrieved contexts shape: {ref_contexts.shape}")
        print(f"✓ Value retrieval successful")
    
    # Fusion
    fused = memory_bank.fuse_references(current_context, ref_contexts, attn_weights)
    print(f"  - Fused context shape: {fused.shape}")
    print(f"✓ Fusion successful")
    
    # Forward統合テスト
    enhanced = memory_bank.forward(current_context, k=2)
    print(f"  - Enhanced context shape: {enhanced.shape}")
    print(f"✓ Full forward pass successful")
    
    print("\n✅ Test 1 PASSED\n")


def test_phase2_model():
    """Phase 2モデルの基本テスト"""
    print("="*60)
    print("Test 2: HPCM_MultiRef_Phase2 Basic Forward")
    print("="*60)
    
    from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
    
    x = torch.randn(1, 3, 256, 256).cuda()
    
    print("\n[1] Testing Phase 2 with full multi-reference...")
    model_phase2 = HPCM_MultiRef_Phase2(
        M=320,
        N=256,
        enable_multiref=True,
        max_refs_s1=2,
        max_refs_s2=3,
        max_refs_s3=4,
        topk_refs_s1=1,
        topk_refs_s2=2,
        topk_refs_s3=2,
        enable_hierarchical_transfer=False  # Phase 2基本版
    ).cuda()
    model_phase2.eval()
    
    with torch.no_grad():
        output = model_phase2(x, training=False)
    
    print(f"✓ Output x_hat shape: {output['x_hat'].shape}")
    print(f"✓ y likelihood shape: {output['likelihoods']['y'].shape}")
    print(f"✓ z likelihood shape: {output['likelihoods']['z'].shape}")
    
    print("\n[2] Testing Baseline mode...")
    model_baseline = HPCM_MultiRef_Phase2(
        M=320,
        N=256,
        enable_multiref=False
    ).cuda()
    model_baseline.eval()
    
    with torch.no_grad():
        output_baseline = model_baseline(x, training=False)
    
    print(f"✓ Baseline output shape: {output_baseline['x_hat'].shape}")
    
    print("\n✅ Test 2 PASSED\n")


def test_all_scales():
    """全階層のMulti-Referenceテスト"""
    print("="*60)
    print("Test 3: Multi-Reference on All Scales (s1/s2/s3)")
    print("="*60)
    
    from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
    
    x = torch.randn(1, 3, 128, 128).cuda()
    
    configs = [
        {
            "name": "Phase 2 - Conservative",
            "max_refs_s1": 2, "max_refs_s2": 2, "max_refs_s3": 3,
            "topk_refs_s1": 1, "topk_refs_s2": 1, "topk_refs_s3": 2
        },
        {
            "name": "Phase 2 - Balanced",
            "max_refs_s1": 2, "max_refs_s2": 3, "max_refs_s3": 4,
            "topk_refs_s1": 1, "topk_refs_s2": 2, "topk_refs_s3": 2
        },
        {
            "name": "Phase 2 - Aggressive",
            "max_refs_s1": 3, "max_refs_s2": 4, "max_refs_s3": 6,
            "topk_refs_s1": 2, "topk_refs_s2": 3, "topk_refs_s3": 3
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}] Testing {config['name']}...")
        config_copy = {k: v for k, v in config.items() if k != "name"}
        
        model = HPCM_MultiRef_Phase2(
            M=320, N=256,
            enable_multiref=True,
            **config_copy
        ).cuda()
        model.eval()
        
        with torch.no_grad():
            output = model(x, training=False)
        
        print(f"✓ {config['name']} successful")
    
    print("\n✅ Test 3 PASSED\n")


def test_training_mode():
    """訓練モードテスト"""
    print("="*60)
    print("Test 4: Training Mode with Full Multi-Reference")
    print("="*60)
    
    from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
    
    x = torch.randn(2, 3, 128, 128).cuda()
    
    model = HPCM_MultiRef_Phase2(
        M=320,
        N=256,
        enable_multiref=True,
        max_refs_s1=2,
        max_refs_s2=3,
        max_refs_s3=4
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
    
    # パラメータ数比較
    print("\n[4] Model complexity...")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {param_count:,}")
    
    print("\n✅ Test 4 PASSED\n")


def test_memory_efficiency():
    """メモリ効率のテスト"""
    print("="*60)
    print("Test 5: Memory Efficiency Comparison")
    print("="*60)
    
    from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
    
    x = torch.randn(1, 3, 256, 256).cuda()
    
    configs = [
        {"compress_ratio": 2, "value_resolution": 16, "name": "High Quality"},
        {"compress_ratio": 4, "value_resolution": 8, "name": "Balanced"},
        {"compress_ratio": 8, "value_resolution": 4, "name": "Memory Efficient"}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}] Testing {config['name']}...")
        
        torch.cuda.reset_peak_memory_stats()
        
        model = HPCM_MultiRef_Phase2(
            M=320, N=256,
            enable_multiref=True,
            max_refs_s1=2, max_refs_s2=3, max_refs_s3=4,
            compress_ratio=config['compress_ratio'],
            value_resolution=config['value_resolution']
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
    
    print("\n✅ Test 5 PASSED\n")


def main():
    """全テスト実行"""
    print("\n" + "="*60)
    print("HPCM Multi-Reference Phase 2 - Test Suite")
    print("="*60 + "\n")
    
    try:
        test_full_memory_bank()
        test_phase2_model()
        test_all_scales()
        test_training_mode()
        test_memory_efficiency()
        
        print("="*60)
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("="*60)
        print("\nPhase 2実装が正常に動作しています。")
        print("\n主な改善点:")
        print("✓ Value保存機能の完全実装")
        print("✓ s1/s2/s3全階層へのMulti-Reference適用")
        print("✓ より洗練されたattention機構")
        print("✓ メモリ効率化オプション")
        print("\n次のステップ:")
        print("1. Phase 1 vs Phase 2 の性能比較")
        print("2. 階層間メモリ共有の効果検証")
        print("3. 実データセットでの評価")
        
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
