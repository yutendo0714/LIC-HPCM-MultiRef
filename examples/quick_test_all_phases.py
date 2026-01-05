#!/usr/bin/env python
"""
Quick Test Runner for all Phases

Runs basic smoke tests for Phase 1, 2, and 3.
"""

import torch
import sys
sys.path.insert(0, '/workspace/LIC-HPCM-MultiRef')


def check_environment():
    """環境チェック"""
    print("="*60)
    print("Environment Check")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("")


def test_phase(phase_num, model_class, model_name):
    """Phase別のテスト"""
    print("="*60)
    print(f"Phase {phase_num}: {model_name}")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping GPU test (CUDA not available)")
        print("Testing CPU mode instead...")
        device = 'cpu'
    else:
        device = 'cuda'
    
    try:
        # Create dummy input (larger size to avoid shape mismatch)
        x = torch.randn(1, 3, 256, 256).to(device)
        
        # Create model
        print(f"\n[1] Creating {model_name}...")
        if phase_num == 1:
            model = model_class(M=320, N=256, enable_multiref=True).to(device)
        else:
            model = model_class(
                M=320, N=256, 
                enable_multiref=True,
                max_refs_s1=2, max_refs_s2=3, max_refs_s3=4
            ).to(device)
        
        model.eval()
        print(f"✓ Model created and moved to {device}")
        
        # Forward pass
        print(f"\n[2] Running forward pass...")
        with torch.no_grad():
            output = model(x, training=False)
        
        print(f"✓ Output x_hat shape: {output['x_hat'].shape}")
        print(f"✓ Likelihoods present: y={output['likelihoods']['y'].shape}, z={output['likelihoods']['z'].shape}")
        
        # Parameter count
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n[3] Model statistics:")
        print(f"  Total parameters: {param_count:,}")
        
        print(f"\n✅ Phase {phase_num} test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase {phase_num} test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("HPCM Multi-Reference - Quick Test Suite")
    print("Testing all phases (1, 2, 3)")
    print("="*60 + "\n")
    
    check_environment()
    
    results = []
    
    # Phase 1
    try:
        from src.models.multiref.phase1 import HPCM_MultiRef_Phase1
        results.append(("Phase 1", test_phase(1, HPCM_MultiRef_Phase1, "HPCM_MultiRef_Phase1")))
    except Exception as e:
        print(f"❌ Failed to import Phase 1: {e}\n")
        results.append(("Phase 1", False))
    
    # Phase 2
    try:
        from src.models.multiref.phase2 import HPCM_MultiRef_Phase2
        results.append(("Phase 2", test_phase(2, HPCM_MultiRef_Phase2, "HPCM_MultiRef_Phase2")))
    except Exception as e:
        print(f"❌ Failed to import Phase 2: {e}\n")
        results.append(("Phase 2", False))
    
    # Phase 3
    try:
        from src.models.multiref.phase3 import HPCM_MultiRef_Phase3
        results.append(("Phase 3", test_phase(3, HPCM_MultiRef_Phase3, "HPCM_MultiRef_Phase3")))
    except Exception as e:
        print(f"❌ Failed to import Phase 3: {e}\n")
        results.append(("Phase 3", False))
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
