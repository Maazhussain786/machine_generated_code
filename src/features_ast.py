# src/features_ast.py
"""
AST & structural feature extraction for code snippets.
- Python: uses built-in `ast`
- JavaScript/Java/C/C++: uses tree-sitter if installed (optional)
- Fallback: regex-based heuristics (token counts, punctuation)
"""

from typing import Dict, Any, Tuple, Optional
import ast
import re
import math
import numpy as np
import pandas as pd
import os
import warnings
import logging

# Progress bar for dataframe extraction
from tqdm import tqdm

TRY_TREE_SITTER = True
try:
    from tree_sitter import Language, Parser
except Exception:
    TRY_TREE_SITTER = False

# If tree-sitter is used, we expect language libs built at src/tree_sitter_langs.so (user step)
TREE_SITTER_LIB = os.path.join(os.path.dirname(__file__), "tree_sitter_langs.so")

# Languages mapping: language column values to tree-sitter grammar name
TS_LANG_MAP = {
    "javascript": "javascript",
    "js": "javascript",
    "python": "python",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "csharp": "c_sharp",
}

# Simple regex patterns for fallback tokenization / comment detection
RE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RE_COMMENT_PY = re.compile(r"^\s*#")
RE_COMMENT_C = re.compile(r"^\s*(//|/\*|\*)")

# Silence noisy SyntaxWarnings caused by regex literals shown earlier
warnings.filterwarnings("ignore", category=SyntaxWarning)

logger = logging.getLogger(__name__)


def extract_python_ast_features(code: str, max_len_for_ast: int = 10000) -> Dict[str, Any]:
    """
    Safe Python AST extraction. If code is too long or fails to parse, returns fallback-lite features with parse_error=1.
    """
    feats = {
        "lang": "python",
        "parse_error": 0,
        "num_functions": 0,
        "num_classes": 0,
        "num_imports": 0,
        "num_if": 0,
        "num_for": 0,
        "num_while": 0,
        "num_assign": 0,
        "num_try": 0,
        "num_calls": 0,
        "max_ast_depth": 0,
        "avg_identifier_len": 0.0,
        "num_unique_identifiers": 0.0,
    }

    if not isinstance(code, str):
        feats["parse_error"] = 1
        return feats

    # Guard: avoid costly parsing for extremely large inputs
    if len(code) > max_len_for_ast:
        feats["parse_error"] = 1
        feats.update({
            "char_count": float(len(code)),
            "line_count": float(code.count("\n") + 1),
            "comment_ratio": float(sum(1 for L in code.splitlines() if L.strip().startswith("#"))) / max(1, code.count("\n") + 1)
        })
        return feats

    try:
        tree = ast.parse(code)
    except Exception:
        feats["parse_error"] = 1
        return feats

    names = []
    max_depth = 0

    def visit(node, depth=1):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        t = type(node).__name__
        if t in ("FunctionDef", "AsyncFunctionDef"):
            feats["num_functions"] += 1
            names.append(getattr(node, "name", ""))
        elif t == "ClassDef":
            feats["num_classes"] += 1
            names.append(getattr(node, "name", ""))
        elif t in ("Import", "ImportFrom"):
            feats["num_imports"] += 1
        elif t == "If":
            feats["num_if"] += 1
        elif t in ("For", "AsyncFor"):
            feats["num_for"] += 1
        elif t == "While":
            feats["num_while"] += 1
        elif t == "Assign":
            feats["num_assign"] += 1
        elif t == "Try":
            feats["num_try"] += 1
        elif t == "Call":
            feats["num_calls"] += 1

        # collect identifier names
        if hasattr(node, "name"):
            names.append(getattr(node, "name"))

        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree, depth=1)
    feats["max_ast_depth"] = max_depth

    ids = [n for n in names if isinstance(n, str) and n]
    if ids:
        lens = [len(x) for x in ids]
        feats["avg_identifier_len"] = float(np.mean(lens))
        feats["num_unique_identifiers"] = float(len(set(ids)))
    else:
        feats["avg_identifier_len"] = 0.0
        feats["num_unique_identifiers"] = 0.0
    return feats


def init_tree_sitter_parser(lang_name: str) -> Optional[Parser]:
    """
    Initialize and return a tree-sitter Parser for a given language name.
    Requires a built shared lib with the required languages.
    """
    if not TRY_TREE_SITTER:
        return None
    if not os.path.exists(TREE_SITTER_LIB):
        return None
    try:
        pass
    except Exception:
        pass
    try:
        LANG = Language(TREE_SITTER_LIB, lang_name)
        parser = Parser()
        parser.set_language(LANG)
        return parser
    except Exception:
        return None

def extract_ts_features(code: str, lang: str, parser: Parser) -> Dict[str, Any]:
    """
    Generic tree-sitter traversal to collect node-type counts and depth.
    """
    feats = {
        "lang": lang,
        "parse_error": 0,
        "num_functions": 0,
        "num_classes": 0,
        "num_imports": 0,
        "num_if": 0,
        "num_for": 0,
        "num_while": 0,
        "num_assign": 0,
        "num_calls": 0,
        "max_ast_depth": 0,
        "avg_identifier_len": 0.0,
        "num_unique_identifiers": 0.0,
    }
    if parser is None:
        feats["parse_error"] = 1
        return feats
    try:
        tree = parser.parse(bytes(code, "utf8"))
    except Exception:
        feats["parse_error"] = 1
        return feats

    root = tree.root_node
    from collections import deque, Counter
    q = deque([(root, 1)])
    ident_names = []
    node_counter = Counter()
    max_depth = 0

    while q:
        node, depth = q.popleft()
        max_depth = max(max_depth, depth)
        node_type = node.type
        node_counter[node_type] += 1

        # heuristics for function/class/import/call based on node types
        if "function" in node_type or node_type in ("function_declaration", "function_definition", "method_declaration"):
            feats["num_functions"] += 1
        if "class" in node_type or node_type in ("class_declaration", "class_definition"):
            feats["num_classes"] += 1
        if "import" in node_type:
            feats["num_imports"] += 1
        if node_type in ("if_statement", "if"):
            feats["num_if"] += 1
        if node_type in ("for_statement", "for"):
            feats["num_for"] += 1
        if node_type in ("while_statement", "while"):
            feats["num_while"] += 1
        if "call" in node_type or node_type.endswith("_call"):
            feats["num_calls"] += 1
        # identifier extraction (language-specific node names)
        if node_type in ("identifier", "variable_name", "field_identifier", "name"):
            try:
                ident = code[node.start_byte:node.end_byte].decode("utf8")
            except Exception:
                ident = code[node.start_byte:node.end_byte]
            ident_names.append(ident)

        for c in node.children:
            q.append((c, depth + 1))

    feats["max_ast_depth"] = max_depth
    if ident_names:
        lens = [len(x) for x in ident_names]
        feats["avg_identifier_len"] = float(np.mean(lens))
        feats["num_unique_identifiers"] = float(len(set(ident_names)))
    else:
        feats["avg_identifier_len"] = 0.0
        feats["num_unique_identifiers"] = 0.0

    # add some node counts as extra features
    feats["total_node_types"] = float(len(node_counter))
    feats["most_common_node_type_count"] = float(node_counter.most_common(1)[0][1]) if node_counter else 0.0

    return feats

def fallback_features(code: str, lang_hint: str = "unknown") -> Dict[str, Any]:
    feats = {
        "lang": lang_hint,
        "parse_error": 0,
        "num_functions": 0,
        "num_classes": 0,
        "num_imports": 0,
        "num_if": len(re.findall(r"\bif\b", code)),
        "num_for": len(re.findall(r"\bfor\b", code)),
        "num_while": len(re.findall(r"\bwhile\b", code)),
        "num_assign": len(re.findall(r"=", code)),
        "num_calls": len(re.findall(r"\w+\s*\(", code)),
        "max_ast_depth": 0,
        "avg_identifier_len": 0.0,
        "num_unique_identifiers": 0.0,
    }
    # identifiers
    ids = RE_IDENTIFIER.findall(code)
    if ids:
        lens = [len(x) for x in ids]
        feats["avg_identifier_len"] = float(np.mean(lens))
        feats["num_unique_identifiers"] = float(len(set(ids)))
    # comment ratio (simple)
    lines = code.splitlines()
    if not lines:
        feats["comment_ratio"] = 0.0
    else:
        ccount = 0
        for L in lines:
            if RE_COMMENT_PY.match(L) or RE_COMMENT_C.match(L):
                ccount += 1
        feats["comment_ratio"] = float(ccount) / max(1, len(lines))
    return feats


# Cache parsers per language to avoid reinitializing
_PARSER_CACHE = {}

def get_parser_for_lang(lang_str: str):
    lang_str = (lang_str or "").strip().lower()
    if lang_str in ("python", "py"):
        return "python"  # sentinel - handled by built-in ast
    ts_name = TS_LANG_MAP.get(lang_str)
    if not ts_name:
        return None
    if not TRY_TREE_SITTER:
        return None
    if ts_name in _PARSER_CACHE:
        return _PARSER_CACHE[ts_name]
    parser = init_tree_sitter_parser(ts_name)
    _PARSER_CACHE[ts_name] = parser
    return parser

def extract_structural_features(code: str, lang: Optional[str] = None) -> Dict[str, Any]:
    """
    Unified feature extractor.
    - code: source code string
    - lang: optional language hint (e.g., 'python', 'javascript', 'java')
    Returns a dictionary of numeric features.
    """
    code = code or ""
    lang_hint = (lang or "").strip().lower()

    # quick basic features
    n_chars = len(code)
    n_lines = code.count("\n") + 1 if code else 0
    blank_lines = sum(1 for L in code.splitlines() if not L.strip())

    # comment ratio (best-effort)
    comment_lines = 0
    for L in code.splitlines():
        Ls = L.strip()
        if RE_COMMENT_PY.match(Ls) or RE_COMMENT_C.match(Ls):
            comment_lines += 1
    comment_ratio = (comment_lines / max(1, n_lines)) if n_lines else 0.0


    is_probably_python = False
    if lang_hint in ("python", "py"):
        is_probably_python = True
    elif len(code) < 5000 and (("def " in code) or ("class " in code) or ("import " in code) or ("if " in code and ":" in code)):
        is_probably_python = True

    if is_probably_python:
        
        feats = extract_python_ast_features(code)
        
        if feats.get("parse_error", 0) == 1:
            fb = fallback_features(code, lang_hint)
            for k, v in fb.items():
                if k not in feats or (isinstance(feats[k], (int, float)) and feats[k] == 0):
                    feats[k] = v
    else:
        parser = get_parser_for_lang(lang_hint)
        if parser and parser != "python":
            feats = extract_ts_features(code, lang_hint, parser)
            if feats.get("parse_error", 0) == 1:
                feats = fallback_features(code, lang_hint)
        else:
            feats = fallback_features(code, lang_hint)

    feats["char_count"] = float(n_chars)
    feats["line_count"] = float(n_lines)
    feats["blank_line_ratio"] = float(blank_lines) / max(1, n_lines) if n_lines else 0.0
    feats["comment_ratio"] = float(comment_ratio)
    feats["num_semicolons"] = float(code.count(";"))
    feats["num_braces"] = float(code.count("{") + code.count("}"))
    feats["num_parens"] = float(code.count("(") + code.count(")"))
    feats["num_colons"] = float(code.count(":"))

    
    for k, v in list(feats.items()):
        try:
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                feats[k] = float(v)
        except Exception:
            pass

    return feats

def extract_features_df(df: pd.DataFrame, code_col: str = "code", lang_col: str = "language") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Convert a dataframe with code into features dataframe and label series (if present).
    Returns: (feature_df, y_series or None)
    This function is robust: it iterates with tqdm and catches per-row exceptions,
    falling back to heuristics if AST parsing fails for any sample.
    """
    rows = []
    total = len(df)
    # Use tqdm progress bar for large dataframes
    for i, (_, r) in enumerate(tqdm(df.iterrows(), total=total, desc="Extracting AST features")):
        try:
            code = r.get(code_col, "") if isinstance(r.get(code_col, ""), str) else ""
            lang = r.get(lang_col, None) if lang_col in r.index else None
            feats = extract_structural_features(code, lang)
            rows.append(feats)
        except Exception as e:
            logger.warning(f"Feature extraction failed at row {i}: {e}. Using fallback features.")
            try:
                fb = fallback_features(r.get(code_col, "") if isinstance(r.get(code_col, ""), str) else "", r.get(lang_col, None) if lang_col in r.index else "unknown")
                rows.append(fb)
            except Exception:
                # extreme fallback: minimal empty features
                rows.append({"char_count": 0.0, "line_count": 0.0, "comment_ratio": 0.0, "parse_error": 1})

    feat_df = pd.DataFrame(rows).fillna(0.0)
    # ensure deterministic column order
    feat_df = feat_df.reindex(sorted(feat_df.columns), axis=1)
    y = df["label"].astype(int).reset_index(drop=True) if "label" in df.columns else None
    return feat_df, y