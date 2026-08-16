from __future__ import annotations

from bisect import bisect_right
import io
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence


class SegmentedFile(io.RawIOBase):
    """Read ordered archive chunks as one seekable file."""

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        expected_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self._paths = tuple(Path(path).expanduser().resolve() for path in paths)
        if not self._paths:
            raise ValueError("At least one archive chunk is required")
        for path in self._paths:
            if not path.is_file():
                raise FileNotFoundError(f"Archive chunk is missing: {path}")

        self._sizes = tuple(path.stat().st_size for path in self._paths)
        if expected_sizes is not None:
            expected = tuple(int(size) for size in expected_sizes)
            if expected != self._sizes:
                raise ValueError("Archive chunk sizes do not match the manifest")

        offsets = [0]
        for size in self._sizes:
            offsets.append(offsets[-1] + size)
        self._offsets = tuple(offsets)
        self._handles: list[BinaryIO | None] = [None] * len(self._paths)
        self._position = 0

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
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the start of the archive")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._check_open()
        if self._position >= self.total_size or size == 0:
            return b""

        available = self.total_size - self._position
        bytes_left = available if size is None or size < 0 else min(size, available)
        blocks = []
        while bytes_left:
            chunk_index = bisect_right(self._offsets, self._position) - 1
            chunk_offset = self._position - self._offsets[chunk_index]
            block_size = min(bytes_left, self._sizes[chunk_index] - chunk_offset)
            handle = self._handles[chunk_index]
            if handle is None:
                handle = self._paths[chunk_index].open("rb")
                self._handles[chunk_index] = handle
            handle.seek(chunk_offset)
            block = handle.read(block_size)
            if len(block) != block_size:
                raise OSError(f"Short read from {self._paths[chunk_index].name}")
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
            raise ValueError("I/O operation on a closed archive")
