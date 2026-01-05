"""
Phase 3: Linear Attention based Multi-Reference

Implementation features:
- MLIC++ style Linear Attention
- Channel-wise attention (O(N) complexity)
- Applied to all scales (s1/s2/s3)
- Kernel feature map (ELU-based)
- Efficient memory management
"""

from .hpcm_base_phase3 import HPCM_MultiRef_Phase3

__all__ = ['HPCM_MultiRef_Phase3']
