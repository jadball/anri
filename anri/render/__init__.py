"""Render forward-projected peaks into detector frames, sinograms and sparse pixel lists.

Work in progress: only the dense kernel exists so far.
"""

from ._impl.kernel import splat

__all__ = ["splat"]
