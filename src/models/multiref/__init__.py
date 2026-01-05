"""
Multi-Reference Enhanced HPCM Models

Progressive implementation:
- Phase 1: Lightweight Multi-Reference applied to s3 scale only
- Phase 2: Applied to all scales with enhanced value storage
- Phase 3: MLIC++ style Linear Attention integration
"""

from .phase1 import HPCM_MultiRef_Phase1
from .phase2 import HPCM_MultiRef_Phase2
from .phase3 import HPCM_MultiRef_Phase3

__all__ = ['HPCM_MultiRef_Phase1', 'HPCM_MultiRef_Phase2', 'HPCM_MultiRef_Phase3']
