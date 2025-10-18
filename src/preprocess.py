# src/preprocess.py
import re
from typing import Iterable
import pandas as pd


TOKEN_SAFE_CHARS = r"[A-Za-z0-9_]+"  

def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").strip()

def basic_clean_code(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = normalize_newlines(s)
    # collapse multiple blank lines
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s

def add_length_features(df: pd.DataFrame, code_col: str = "code") -> pd.DataFrame:
    df = df.copy()
    df[code_col] = df[code_col].astype(str)
    df["char_count"] = df[code_col].map(len)
    df["line_count"] = df[code_col].map(lambda s: s.count("\n") + 1)
    df["avg_line_len"] = df.apply(lambda r: r["char_count"] / max(1, r["line_count"]), axis=1)
    # comment ratio (simple heuristic for common languages: lines starting with # or // or /* */)
    def comment_ratio(s: str):
        lines = s.splitlines()
        if not lines:
            return 0.0
        comment_lines = 0
        for L in lines:
            Ls = L.strip()
            if Ls.startswith("#") or Ls.startswith("//") or Ls.startswith("/*") or Ls.startswith("*"):
                comment_lines += 1
        return comment_lines / max(1, len(lines))
    df["comment_ratio"] = df[code_col].map(comment_ratio)
    return df
