"""공유 상수 및 문자 인코딩 유틸."""

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 영대문자만
NUM_CLASSES = len(CHARS)               # 26
NUM_POS = 6                            # 항상 6글자 고정

IMG_H = 64
IMG_W = 192

CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}


def encode(text: str):
    """'ABCDEF' -> [0, 1, 2, 3, 4, 5]"""
    return [CHAR_TO_IDX[c] for c in text]


def decode(indices) -> str:
    """[0, 1, 2, 3, 4, 5] -> 'ABCDEF'"""
    return "".join(IDX_TO_CHAR[int(i)] for i in indices)
