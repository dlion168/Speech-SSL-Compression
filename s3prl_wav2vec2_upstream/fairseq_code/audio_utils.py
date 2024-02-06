import mmap
from pathlib import Path
from typing import List, Tuple

FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}

def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
    1. a .npy/.wav/.flac/.ogg file
    2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          slice_ptr (list of int): empty in case 1;
            byte offset and length for the slice in case 2
    """

    if Path(path).suffix in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        _path, slice_ptr = path, []
    else:
        _path, *slice_ptr = path.split(":")
        if not Path(_path).is_file():
            raise FileNotFoundError(f"File not found: {_path}")
    assert len(slice_ptr) in {0, 2}, f"Invalid path: {path}"
    slice_ptr = [int(i) for i in slice_ptr]
    return _path, slice_ptr

def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset : offset + length]
    return data


def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)

def is_sf_audio_data(data: bytes) -> bool:
    is_wav = data[0] == 82 and data[1] == 73 and data[2] == 70
    is_flac = data[0] == 102 and data[1] == 76 and data[2] == 97
    is_ogg = data[0] == 79 and data[1] == 103 and data[2] == 103
    return is_wav or is_flac or is_ogg