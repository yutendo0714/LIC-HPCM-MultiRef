"""
Multi-Reference Enhanced HPCM Models

段階的実装:
- Phase 1: s3階層のみに軽量版Multi-Reference Memory Bank適用 ✅
- Phase 2: 全階層への展開 + Value保存機能強化 ✅
- Phase 3: MLIC++スタイルのLinear Attention統合 ✅
"""

from .phase1 import HPCM_MultiRef_Phase1
from .phase2 import HPCM_MultiRef_Phase2
from .phase3 import HPCM_MultiRef_Phase3

__all__ = ['HPCM_MultiRef_Phase1', 'HPCM_MultiRef_Phase2', 'HPCM_MultiRef_Phase3']
