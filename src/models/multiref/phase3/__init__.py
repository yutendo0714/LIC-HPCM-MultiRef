"""
Phase 3: Linear Attention based Multi-Reference

#"""
Phase 3: Linear Attention based Multi-Reference


:
- ✅ MLIC++スタイルLinear Attention
- ✅ Channel-wise attention (O(N) complexity)
- ✅ s1/s2/s3全階層への適用
- ✅ Kernel feature map (ELU-based)
- ✅ 効率的なメモリ管理
"""

from .hpcm_base_phase3 import HPCM_MultiRef_Phase3

__all__ = ['HPCM_MultiRef_Phase3']
