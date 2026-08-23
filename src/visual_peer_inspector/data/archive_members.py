"""Read and safely extract indexed members from a segmented ZIP archive."""

from __future__ import annotations

from binascii import crc32
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import sqlite3
import struct
from typing import BinaryIO
import zlib


LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
LOCAL_FILE_STRUCT = struct.Struct("<4s5H3I2H")


@dataclass(frozen=True)
class IndexedMember:
    name: str
    compressed_size: int
    uncompressed_size: int
    crc32: int
    compression_method: int
    flags: int
    local_header_offset: int
    is_directory: bool
    is_symlink: bool
    unsafe_reason: str | None


def load_indexed_member(database_path: Path, name: str) -> IndexedMember:
    """Load one unique member record from the local SQLite index."""

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name, compressed_size, uncompressed_size, crc32, "
            "compression_method, flags, local_header_offset, is_directory, "
            "is_symlink, unsafe_reason FROM members WHERE name = ?",
            (name,),
        ).fetchall()
    if not rows:
        raise KeyError(f"Archive member not found: {name}")
    if len(rows) != 1:
        raise ValueError(f"Archive member name is duplicated: {name}")
    row = rows[0]
    return IndexedMember(
        name=str(row[0]),
        compressed_size=int(row[1]),
        uncompressed_size=int(row[2]),
        crc32=int(row[3]),
        compression_method=int(row[4]),
        flags=int(row[5]),
        local_header_offset=int(row[6]),
        is_directory=bool(row[7]),
        is_symlink=bool(row[8]),
        unsafe_reason=row[9],
    )


def _decode_name(raw_name: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    return raw_name.decode(encoding, errors="replace")


def read_indexed_member(
    stream: BinaryIO,
    member: IndexedMember,
    *,
    maximum_uncompressed_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    """Read one member, validate its header, size and CRC, and return bytes."""

    if member.is_directory:
        return b""
    if member.is_symlink:
        raise ValueError(f"Refusing to read symlink member: {member.name}")
    if member.unsafe_reason:
        raise ValueError(
            f"Refusing unsafe member {member.name}: {member.unsafe_reason}"
        )
    if member.flags & 0x1:
        raise ValueError(f"Encrypted ZIP member is unsupported: {member.name}")
    if member.uncompressed_size > maximum_uncompressed_bytes:
        raise ValueError(
            f"Member exceeds read limit ({member.uncompressed_size} > "
            f"{maximum_uncompressed_bytes}): {member.name}"
        )

    stream.seek(member.local_header_offset)
    fixed = stream.read(LOCAL_FILE_STRUCT.size)
    if len(fixed) != LOCAL_FILE_STRUCT.size:
        raise EOFError(f"Truncated local header: {member.name}")
    fields = LOCAL_FILE_STRUCT.unpack(fixed)
    if fields[0] != LOCAL_FILE_SIGNATURE:
        raise ValueError(f"Invalid local-file signature: {member.name}")
    local_flags = fields[2]
    local_method = fields[3]
    name_length = fields[9]
    extra_length = fields[10]
    raw_name = stream.read(name_length)
    if len(raw_name) != name_length:
        raise EOFError(f"Truncated local filename: {member.name}")
    local_name = _decode_name(raw_name, local_flags)
    if local_name != member.name:
        raise ValueError(
            f"Central/local filename mismatch: {member.name!r} != {local_name!r}"
        )
    if local_method != member.compression_method:
        raise ValueError(f"Central/local compression mismatch: {member.name}")
    if extra_length:
        extra = stream.read(extra_length)
        if len(extra) != extra_length:
            raise EOFError(f"Truncated local extra field: {member.name}")

    compressed = stream.read(member.compressed_size)
    if len(compressed) != member.compressed_size:
        raise EOFError(f"Truncated compressed content: {member.name}")
    if member.compression_method == 0:
        content = compressed
    elif member.compression_method == 8:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"Unsupported compression method {member.compression_method}: "
            f"{member.name}"
        )

    if len(content) != member.uncompressed_size:
        raise ValueError(
            f"Uncompressed-size mismatch for {member.name}: "
            f"expected {member.uncompressed_size}, found {len(content)}"
        )
    actual_crc = crc32(content) & 0xFFFFFFFF
    if actual_crc != member.crc32:
        raise ValueError(
            f"CRC mismatch for {member.name}: "
            f"expected {member.crc32:08x}, found {actual_crc:08x}"
        )
    return content


def safe_output_path(root: Path, member_name: str) -> Path:
    """Return a path under root or reject traversal/absolute names."""

    normalised = member_name.replace("\\", "/")
    parts = PurePosixPath(normalised).parts
    if not parts or normalised.startswith("/") or ".." in parts:
        raise ValueError(f"Unsafe output member path: {member_name}")
    output = root.expanduser().resolve().joinpath(*parts)
    root_resolved = root.expanduser().resolve()
    if root_resolved not in output.parents and output != root_resolved:
        raise ValueError(f"Output escapes extraction root: {member_name}")
    return output


def extract_indexed_member(
    stream: BinaryIO,
    member: IndexedMember,
    output_root: Path,
    *,
    maximum_uncompressed_bytes: int = 16 * 1024 * 1024,
) -> Path:
    """CRC-check and exclusively create one indexed archive member."""

    if member.is_directory:
        raise ValueError(f"Use file members for selective extraction: {member.name}")
    content = read_indexed_member(
        stream,
        member,
        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
    )
    output = safe_output_path(output_root, member.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(content)
    return output
