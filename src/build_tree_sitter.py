from tree_sitter import Language
from pathlib import Path

ROOT = Path(__file__).parent
OUT = str(ROOT / "tree_sitter_langs.so")   

LANG_DIRS = [
    str(ROOT / "tree-sitter-javascript"),
    str(ROOT / "tree-sitter-java"),
    str(ROOT / "tree-sitter-c"),
    str(ROOT / "tree-sitter-cpp"),
]

print("Building tree-sitter languages into:", OUT)
Language.build_library(
    OUT,
    LANG_DIRS
)
print("Done. Built:", OUT)