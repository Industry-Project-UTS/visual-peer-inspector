"""Seekable read-only view over explicitly ordered file segments."""

from __future__ import annotations

from bisect import bisect_right
import io
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence


class SegmentedFile(io.RawIOBase):
    """Expose several immutable segments as one logical binary file.

    The class never writes to its segments. Reads that cross a segment boundary
    are filled from subsequent segments so callers see the same byte stream as
    an explicit concatenation.
    """

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        expected_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self._handles: list[BinaryIO | None] = []
        self._paths = tuple(Path(path).expanduser().resolve() for path in paths)
        if not self._paths:
            raise ValueError("At least one segment is required")

        for path in self._paths:
            if not path.is_file():
                raise FileNotFoundError(f"Segment is missing: {path}")

        self._sizes = tuple(path.stat().st_size for path in self._paths)
        if expected_sizes is not None:
            expected = tuple(int(size) for size in expected_sizes)
            if len(expected) != len(self._sizes):
                raise ValueError("Expected-size count does not match segment count")
            mismatches = [
                f"{path.name}: expected {wanted}, found {actual}"
                for path, wanted, actual in zip(
                    self._paths, expected, self._sizes, strict=True
                )
                if wanted != actual
            ]
            if mismatches:
                raise ValueError("Segment size mismatch: " + "; ".join(mismatches))

        offsets = [0]
        for size in self._sizes:
            offsets.append(offsets[-1] + size)
        self._offsets = tuple(offsets)
        self._position = 0
        self._handles = [None] * len(self._paths)
        self.name = " + ".join(path.name for path in self._paths)

    @property
    def segment_paths(self) -> tuple[Path, ...]:
        return self._paths

    @property
    def segment_sizes(self) -> tuple[int, ...]:
        return self._sizes

    @property
    def total_size(self) -> int:
        return self._offsets[-1]

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._check_open()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self._position + offset
        elif whence == io.SEEK_END:
            new_position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported whence value: {whence}")
        if new_position < 0:
            raise ValueError("Negative seek position")
        self._position = new_position
        return self._position

    def read(self, size: int = -1) -> bytes:
        self._check_open()
        if self._position >= self.total_size or size == 0:
            return b""

        remaining = self.total_size - self._position
        if size is None or size < 0:
            requested = remaining
        else:
            requested = min(size, remaining)

        blocks: list[bytes] = []
        bytes_left = requested
        while bytes_left:
            segment_index = bisect_right(self._offsets, self._position) - 1
            local_offset = self._position - self._offsets[segment_index]
            available = self._sizes[segment_index] - local_offset
            block_size = min(bytes_left, available)

            handle = self._handles[segment_index]
            if handle is None:
                handle = self._paths[segment_index].open("rb")
                self._handles[segment_index] = handle
            handle.seek(local_offset)
            block = handle.read(block_size)
            if len(block) != block_size:
                raise OSError(
                    f"Short read from {self._paths[segment_index]}: "
                    f"expected {block_size}, received {len(block)}"
                )
            blocks.append(block)
            self._position += block_size
            bytes_left -= block_size

        return b"".join(blocks)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        for handle in self._handles:
            if handle is not None:
                handle.close()
        super().close()

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed segmented file")
