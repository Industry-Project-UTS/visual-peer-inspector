"""Streaming ZIP central-directory inventory without archive reconstruction."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import struct
from typing import BinaryIO, Callable


EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_FILE_SIGNATURE = b"PK\x01\x02"
EOCD_STRUCT = struct.Struct("<4s4H2IH")
ZIP64_EOCD_STRUCT = struct.Struct("<QHHIIQQQQ")
CENTRAL_FILE_STRUCT = struct.Struct("<4s6H3I5H2I")
EXTRA_HEADER_STRUCT = struct.Struct("<HH")
ZIP64_EXTRA_ID = 0x0001
WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class CentralDirectoryRecord:
    total_entries: int
    directory_size: int
    directory_offset: int
    record_offset: int
    zip64: bool


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, received {len(data)}")
    return data


def read_central_directory_record(
    stream: BinaryIO, *, tail_size: int = 128 * 1024
) -> CentralDirectoryRecord:
    """Read standard or ZIP64 end-of-central-directory metadata."""

    stream.seek(0, 2)
    archive_size = stream.tell()
    window_size = min(tail_size, archive_size)
    stream.seek(-window_size, 2)
    window_start = stream.tell()
    tail = _read_exact(stream, window_size)

    eocd_index = tail.rfind(EOCD_SIGNATURE)
    if eocd_index < 0 or eocd_index + EOCD_STRUCT.size > len(tail):
        raise ValueError("ZIP end-of-central-directory record not found")

    (
        _,
        _disk_number,
        _directory_disk,
        entries_on_disk,
        total_entries,
        directory_size,
        directory_offset,
        comment_length,
    ) = EOCD_STRUCT.unpack_from(tail, eocd_index)
    if eocd_index + EOCD_STRUCT.size + comment_length > len(tail):
        raise ValueError("ZIP comment extends beyond the available tail window")

    uses_zip64 = (
        entries_on_disk == 0xFFFF
        or total_entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    )
    if not uses_zip64:
        return CentralDirectoryRecord(
            total_entries=total_entries,
            directory_size=directory_size,
            directory_offset=directory_offset,
            record_offset=window_start + eocd_index,
            zip64=False,
        )

    zip64_index = tail.rfind(ZIP64_EOCD_SIGNATURE, 0, eocd_index)
    if zip64_index < 0:
        raise ValueError("ZIP64 EOCD record not found before standard EOCD")
    fields = ZIP64_EOCD_STRUCT.unpack_from(tail, zip64_index + 4)
    (
        _record_size,
        _version_made,
        _version_needed,
        _zip64_disk,
        _zip64_directory_disk,
        _zip64_entries_on_disk,
        zip64_total_entries,
        zip64_directory_size,
        zip64_directory_offset,
    ) = fields
    return CentralDirectoryRecord(
        total_entries=zip64_total_entries,
        directory_size=zip64_directory_size,
        directory_offset=zip64_directory_offset,
        record_offset=window_start + zip64_index,
        zip64=True,
    )


def _expand_zip64_values(
    extra: bytes,
    *,
    uncompressed_size: int,
    compressed_size: int,
    local_header_offset: int,
    disk_start: int,
) -> tuple[int, int, int, int]:
    cursor = 0
    zip64_payload: bytes | None = None
    while cursor + EXTRA_HEADER_STRUCT.size <= len(extra):
        field_id, field_size = EXTRA_HEADER_STRUCT.unpack_from(extra, cursor)
        cursor += EXTRA_HEADER_STRUCT.size
        payload_end = cursor + field_size
        if payload_end > len(extra):
            raise ValueError("Malformed ZIP extra field")
        if field_id == ZIP64_EXTRA_ID:
            zip64_payload = extra[cursor:payload_end]
            break
        cursor = payload_end

    requires_zip64 = (
        uncompressed_size == 0xFFFFFFFF
        or compressed_size == 0xFFFFFFFF
        or local_header_offset == 0xFFFFFFFF
        or disk_start == 0xFFFF
    )
    if not requires_zip64:
        return uncompressed_size, compressed_size, local_header_offset, disk_start
    if zip64_payload is None:
        raise ValueError("ZIP64 sentinel found without ZIP64 extra field")

    cursor = 0

    def read_uint64() -> int:
        nonlocal cursor
        if cursor + 8 > len(zip64_payload):
            raise ValueError("Truncated ZIP64 extra field")
        value = struct.unpack_from("<Q", zip64_payload, cursor)[0]
        cursor += 8
        return value

    if uncompressed_size == 0xFFFFFFFF:
        uncompressed_size = read_uint64()
    if compressed_size == 0xFFFFFFFF:
        compressed_size = read_uint64()
    if local_header_offset == 0xFFFFFFFF:
        local_header_offset = read_uint64()
    if disk_start == 0xFFFF:
        if cursor + 4 > len(zip64_payload):
            raise ValueError("Truncated ZIP64 disk-start value")
        disk_start = struct.unpack_from("<I", zip64_payload, cursor)[0]

    return uncompressed_size, compressed_size, local_header_offset, disk_start


def _decode_name(raw_name: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    return raw_name.decode(encoding, errors="replace")


def _unsafe_path_reason(name: str) -> str | None:
    normalised = name.replace("\\", "/")
    if "\x00" in normalised:
        return "NUL byte"
    if normalised.startswith("/") or WINDOWS_DRIVE.match(normalised):
        return "absolute path"
    if ".." in PurePosixPath(normalised).parts:
        return "parent traversal"
    return None


def build_archive_index(
    stream: BinaryIO,
    database_path: Path,
    summary_path: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Stream the central directory into SQLite and write a compact summary."""

    record = read_central_directory_record(stream)
    if record.directory_offset + record.directory_size != record.record_offset:
        raise ValueError(
            "Central-directory size/offset does not end at its EOCD record"
        )

    database_path = database_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    for output in (database_path, summary_path):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    temporary_database = database_path.with_suffix(database_path.suffix + ".tmp")
    if temporary_database.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary output: {temporary_database}"
        )

    top_levels: Counter[str] = Counter()
    path_prefixes: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    compression_methods: Counter[int] = Counter()
    unsafe_samples: list[dict[str, str]] = []
    total_compressed = 0
    total_uncompressed = 0
    directory_count = 0
    symlink_count = 0
    maximum_member = {"name": "", "uncompressed_size": -1}
    first_names: list[str] = []

    connection = sqlite3.connect(temporary_database)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;
            CREATE TABLE members (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                compressed_size INTEGER NOT NULL,
                uncompressed_size INTEGER NOT NULL,
                crc32 INTEGER NOT NULL,
                compression_method INTEGER NOT NULL,
                flags INTEGER NOT NULL,
                local_header_offset INTEGER NOT NULL,
                is_directory INTEGER NOT NULL,
                is_symlink INTEGER NOT NULL,
                unsafe_reason TEXT,
                top_level TEXT,
                extension TEXT
            );
            """
        )
        connection.execute("BEGIN")
        stream.seek(record.directory_offset)
        rows: list[tuple[object, ...]] = []

        for index in range(record.total_entries):
            fixed = _read_exact(stream, CENTRAL_FILE_STRUCT.size)
            fields = CENTRAL_FILE_STRUCT.unpack(fixed)
            if fields[0] != CENTRAL_FILE_SIGNATURE:
                raise ValueError(
                    f"Invalid central-file signature at entry {index}: {fields[0]!r}"
                )

            version_made = fields[1]
            flags = fields[3]
            compression_method = fields[4]
            crc32 = fields[7]
            compressed_size = fields[8]
            uncompressed_size = fields[9]
            name_length = fields[10]
            extra_length = fields[11]
            comment_length = fields[12]
            disk_start = fields[13]
            external_attributes = fields[15]
            local_header_offset = fields[16]

            raw_name = _read_exact(stream, name_length)
            extra = _read_exact(stream, extra_length)
            if comment_length:
                _read_exact(stream, comment_length)

            (
                uncompressed_size,
                compressed_size,
                local_header_offset,
                disk_start,
            ) = _expand_zip64_values(
                extra,
                uncompressed_size=uncompressed_size,
                compressed_size=compressed_size,
                local_header_offset=local_header_offset,
                disk_start=disk_start,
            )
            if disk_start != 0:
                raise ValueError(
                    f"Entry {index} unexpectedly references ZIP disk {disk_start}"
                )

            name = _decode_name(raw_name, flags)
            normalised = name.replace("\\", "/")
            parts = PurePosixPath(normalised).parts
            top_level = parts[0] if parts else ""
            path_prefix = "/".join(parts[:2]) if parts else ""
            final_name = parts[-1] if parts else ""
            extension = Path(final_name).suffix.lower()
            is_directory = normalised.endswith("/")
            creator_system = version_made >> 8
            unix_mode = (external_attributes >> 16) & 0xFFFF
            is_symlink = creator_system == 3 and (unix_mode & 0o170000) == 0o120000
            unsafe_reason = _unsafe_path_reason(name)

            total_compressed += compressed_size
            total_uncompressed += uncompressed_size
            directory_count += int(is_directory)
            symlink_count += int(is_symlink)
            top_levels[top_level] += 1
            path_prefixes[path_prefix] += 1
            extensions[extension or "[none]"] += 1
            compression_methods[compression_method] += 1
            if len(first_names) < 25:
                first_names.append(name)
            if unsafe_reason and len(unsafe_samples) < 100:
                unsafe_samples.append({"name": name, "reason": unsafe_reason})
            if uncompressed_size > int(maximum_member["uncompressed_size"]):
                maximum_member = {
                    "name": name,
                    "uncompressed_size": uncompressed_size,
                }

            rows.append(
                (
                    index,
                    name,
                    compressed_size,
                    uncompressed_size,
                    crc32,
                    compression_method,
                    flags,
                    local_header_offset,
                    int(is_directory),
                    int(is_symlink),
                    unsafe_reason,
                    top_level,
                    extension,
                )
            )
            if len(rows) >= 5_000:
                connection.executemany(
                    "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                rows.clear()
            if progress and (index + 1) % 100_000 == 0:
                progress(index + 1, record.total_entries)

        if rows:
            connection.executemany(
                "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

        if stream.tell() != record.directory_offset + record.directory_size:
            raise ValueError("Central-directory parser did not end at expected offset")

        connection.execute("CREATE INDEX members_name_idx ON members(name)")
        connection.execute(
            "CREATE INDEX members_top_level_idx ON members(top_level)"
        )
        connection.execute(
            "CREATE INDEX members_extension_idx ON members(extension)"
        )
        duplicate_group_count = connection.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT name FROM members GROUP BY name HAVING COUNT(*) > 1"
            ")"
        ).fetchone()[0]
        duplicate_samples = [
            {"name": row[0], "count": row[1]}
            for row in connection.execute(
                "SELECT name, COUNT(*) AS copies FROM members "
                "GROUP BY name HAVING copies > 1 ORDER BY copies DESC, name LIMIT 100"
            )
        ]
        unsafe_count = connection.execute(
            "SELECT COUNT(*) FROM members WHERE unsafe_reason IS NOT NULL"
        ).fetchone()[0]
        connection.commit()

        summary: dict[str, object] = {
            "archive": asdict(record),
            "entries_indexed": record.total_entries,
            "file_entries": record.total_entries - directory_count,
            "directory_entries": directory_count,
            "total_compressed_member_bytes": total_compressed,
            "total_uncompressed_member_bytes": total_uncompressed,
            "symlink_entries": symlink_count,
            "unsafe_path_entries": unsafe_count,
            "unsafe_path_samples": unsafe_samples,
            "duplicate_name_groups": duplicate_group_count,
            "duplicate_name_samples": duplicate_samples,
            "largest_member": maximum_member,
            "compression_methods": dict(compression_methods.most_common()),
            "top_levels": dict(top_levels.most_common(50)),
            "path_prefixes": dict(path_prefixes.most_common(100)),
            "extensions": dict(extensions.most_common(100)),
            "first_member_names": first_names,
        }
        temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        with temporary_summary.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_database.replace(database_path)
        temporary_summary.replace(summary_path)
        return summary
    except Exception:
        connection.rollback()
        connection.close()
        if temporary_database.exists():
            temporary_database.unlink()
        raise
    finally:
        if connection:
            connection.close()
