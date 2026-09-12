"""Separate infrastructure failures from valid per-image terminal outcomes."""

from __future__ import annotations

import errno


def raise_if_infrastructure_failure(error: BaseException) -> None:
    """Stop on unusable compute or storage, including wrapped worker errors."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        fatal = isinstance(current, MemoryError)
        fatal |= isinstance(current, OSError) and current.errno in {
            errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EROFS, errno.EMFILE,
        }
        fatal |= any(token in message for token in (
            "cuda out of memory", "cuda error: device-side assert",
            "cuda error: an illegal memory access", "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
        ))
        if fatal:
            raise error
        current = current.__cause__ or current.__context__
