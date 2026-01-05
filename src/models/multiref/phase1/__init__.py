"""
Phase 1: Lightweight Multi-Reference Memory Bank (s3 scale only)

Implementation features:
- Multi-Reference applied to s3 scale only
- Lightweight memory bank with compressed keys
- Basic value caching
- Proof of concept
"""

from .hpcm_base_phase1 import HPCM_MultiRef_Phase1

__all__ = ['HPCM_MultiRef_Phase1']
