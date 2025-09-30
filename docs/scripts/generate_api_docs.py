#!/usr/bin/env python3
import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
PKG_DIR = ROOT / 'metbit'
OUT_DIR = ROOT / 'docs' / 'app' / 'docs' / 'api'
INIT_FILE = PKG_DIR / '__init__.py'

EXCLUDE_MODULES = {
    '__init__',
    'dev',  # skip experimental dev module
}

# Categorization for API index ordering
CATEGORY_TITLES = {
    'data_processing': 'Data Processing',
    'statistical_model': 'Statistical Models',
    'data_visualization': 'Data Visualization',
    'other': 'Other'
}

MODULE_CATEGORY = {
    # Data Processing
    'baseline': 'data_processing',
    'calibrate': 'data_processing',
    'nmr_preprocess': 'data_processing',
    'pretreatment': 'data_processing',
    'scaler': 'data_processing',
    'spec_norm': 'data_processing',
    'peak_processe': 'data_processing',
    'denoise_spec': 'data_processing',
    'utility': 'data_processing',

    # Statistical Models
    'opls': 'statistical_model',
    'pls': 'statistical_model',
    'lazy_opls_da': 'statistical_model',
    'metbit': 'statistical_model',
    'cross_validation': 'statistical_model',
    'vip': 'statistical_model',

    # Data Visualization
    'boxplot': 'data_visualization',
    'plotting': 'data_visualization',
    'pca_ellipse': 'data_visualization',
    'ui_picky_peak': 'data_visualization',
    'ui_stocsy': 'data_visualization',
    'annotate_peak': 'data_visualization',
    'STOCSY': 'data_visualization',
    'genpage': 'data_visualization',
    'take_intensity': 'data_visualization',
}

CATEGORY_ORDER = ['data_processing', 'statistical_model', 'data_visualization', 'other']

# Preferred categorization for specific public symbols
SYMBOL_CATEGORY = {
    # Data Processing
    'nmr_preprocessing': 'data_processing',
    'calibrate': 'data_processing',
    'Normalization': 'data_processing',
    'Normalise': 'data_processing',
    'peak_chops': 'data_processing',

    # Statistical Models
    'opls_da': 'statistical_model',
    'lazy_opls_da': 'statistical_model',
    'pca': 'statistical_model',
    'UnivarStats': 'statistical_model',
    'vip_scores': 'statistical_model',

    # Visualization / UI
    'STOCSY': 'data_visualization',
    'STOCSY_app': 'data_visualization',
    'pickie_peak': 'data_visualization',
    'get_intensity': 'data_visualization',
    'annotate_peak': 'data_visualization',
}

def mdx_escape(text: str) -> str:
    # Basic sanitation: normalize CRLF, trim trailing spaces
    s = text.replace('\r\n', '\n').rstrip()
    return s

def parse_numpy_doc(text: str):
    """Parse a lightweight subset of NumPy-style docstrings.

    Returns a dict with keys: summary (str), params (list), returns (list).
    Each item in params/returns: {name, type, desc} (name may be '').
    """
    if not text:
        return {"summary": "", "params": [], "returns": []}
    lines = [l.rstrip() for l in text.split('\n')]
    state = 'summary'
    summary_lines: list[str] = []
    params: list[dict] = []
    returns: list[dict] = []
    current = None
    current_list = None

    def flush_current():
        nonlocal current, current_list
        if current and current_list is not None:
            # collapse whitespace in desc
            current['desc'] = ' '.join(current['desc']).strip()
            current_list.append(current)
        current = None
        current_list = None

    i = 0
    while i < len(lines):
        raw = lines[i]
        l = raw.strip()
        # section headers
        if l.lower() == 'parameters':
            flush_current()
            state = 'params'
            # skip possible underline
            if i + 1 < len(lines) and set(lines[i+1].strip()) <= {'-'} and len(lines[i+1].strip()) >= 3:
                i += 1
        elif l.lower() == 'returns':
            flush_current()
            state = 'returns'
            if i + 1 < len(lines) and set(lines[i+1].strip()) <= {'-'} and len(lines[i+1].strip()) >= 3:
                i += 1
        else:
            if state == 'summary':
                summary_lines.append(raw)
            else:
                # try to parse item line
                m = None
                # patterns: name (type): desc  OR  name : type  desc
                m = re.match(r"^\s*([A-Za-z_][\w]*)\s*(?:\(([^)]*)\)|\s*:\s*([^:]+))?\s*:\s*(.*)$", raw)
                if not m and state == 'returns':
                    # returns sometimes: type: desc (without name)
                    m = re.match(r"^\s*(?:\(([^)]*)\)|([^:]+))\s*:\s*(.*)$", raw)
                    if m:
                        name = ''
                        typ = m.group(1) or (m.group(2).strip() if m.group(2) else '')
                        desc = m.group(3)
                        flush_current()
                        current = {"name": name, "type": (typ or '').strip(), "desc": [desc.strip()]}
                        current_list = returns
                        i += 1
                        continue
                if m:
                    name = m.group(1)
                    typ = (m.group(2) or (m.group(3) or '')).strip()
                    desc = m.group(4)
                    flush_current()
                    current = {"name": name, "type": typ, "desc": [desc.strip()]}
                    current_list = params if state == 'params' else returns
                else:
                    # continuation line
                    if current is not None and raw.strip():
                        current['desc'].append(raw.strip())
        i += 1
    flush_current()
    # clean summary: first block until empty line
    summary = []
    for ln in summary_lines:
        if ln.strip() == '':
            if summary:
                break
            else:
                continue
        summary.append(ln)
    # Improve readability for ad-hoc bullets using '•'
    summary_text = '\n'.join(summary).strip()
    if '•' in summary_text:
        # Introduce line breaks before bullets and convert to markdown list
        summary_text = re.sub(r"\s*•\s*", "\n- ", summary_text)
    return {
        'summary': summary_text,
        'params': params,
        'returns': returns,
    }

def fmt_signature(name: str, args: list[str]) -> str:
    return f"{name}({', '.join(args)})"

def parse_module(path: Path):
    src = path.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(src)
    except Exception as e:
        return {
            'error': f'Failed to parse: {e}',
            'module': path.stem,
        }
    moddoc = ast.get_docstring(tree) or ''
    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cdoc = ast.get_docstring(node) or ''
            methods = []
            for b in node.body:
                if isinstance(b, ast.FunctionDef):
                    # include public and dunder __init__
                    if b.name.startswith('_') and b.name != '__init__':
                        continue
                    args = [a.arg for a in b.args.args]
                    if args and args[0] == 'self':
                        args = args[1:]
                    methods.append({
                        'name': b.name,
                        'args': args,
                        'doc': ast.get_docstring(b) or ''
                    })
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    bases.append(getattr(base, 'id', getattr(base, 'attr', '')))  # best-effort
            classes.append({
                'name': node.name,
                'bases': bases,
                'doc': cdoc,
                'methods': methods,
            })
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith('_'):
                continue
            args = [a.arg for a in node.args.args]
            functions.append({
                'name': node.name,
                'args': args,
                'doc': ast.get_docstring(node) or ''
            })
    return {
        'module': path.stem,
        'doc': moddoc,
        'classes': classes,
        'functions': functions,
    }

def categorize_module(name: str) -> str:
    return MODULE_CATEGORY.get(name, 'other')

def write_module_page(mod):
    slug = mod['module']
    out_dir = OUT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'page.mdx'

    lines = []
    lines.append(f"# {slug}\n")
    cat = CATEGORY_TITLES.get(categorize_module(slug), 'Other')
    lines.append(f"> Category: {cat}\n")
    if 'error' in mod:
        lines.append(f"> Error parsing module: {mod['error']}\n")
    doc = mdx_escape(mod.get('doc', ''))
    if doc:
        parsed = parse_numpy_doc(doc)
        if parsed['summary']:
            lines.append(parsed['summary'] + "\n")
        if parsed['params']:
            lines.append("### Parameters\n")
            for p in parsed['params']:
                name = f"`{p['name']}`" if p['name'] else ''
                typ = f" ({p['type']})" if p['type'] else ''
                lines.append(f"- {name}{typ}: {p['desc']}")
            lines.append("")
        if parsed['returns']:
            lines.append("### Returns\n")
            for r in parsed['returns']:
                name = f"`{r['name']}`" if r['name'] else ''
                typ = f" ({r['type']})" if r['type'] else ''
                colon = ': ' if name or typ else ''
                lines.append(f"- {name}{typ}{colon}{r['desc']}")
            lines.append("")

    # Classes
    classes = mod.get('classes', [])
    if classes:
        lines.append("## Classes\n")
        for cls in classes:
            cname = cls['name']
            bases = f"({', '.join(cls['bases'])})" if cls['bases'] else ''
            lines.append(f"### {cname} {bases}\n")
            cdoc = mdx_escape(cls.get('doc', ''))
            if cdoc:
                cparsed = parse_numpy_doc(cdoc)
                if cparsed['summary']:
                    lines.append(cparsed['summary'] + "\n")
                if cparsed['params']:
                    lines.append("#### Constructor Parameters\n")
                    for p in cparsed['params']:
                        name = f"`{p['name']}`" if p['name'] else ''
                        typ = f" ({p['type']})" if p['type'] else ''
                        lines.append(f"- {name}{typ}: {p['desc']}")
                    lines.append("")
            methods = cls.get('methods', [])
            if methods:
                lines.append("#### Methods\n")
                for m in methods:
                    sig = fmt_signature(m['name'], m.get('args', []))
                    lines.append(f"<details>\n<summary><code>{sig}</code></summary>\n")
                    mdoc = mdx_escape(m.get('doc', ''))
                    if mdoc:
                        p = parse_numpy_doc(mdoc)
                        if p['summary']:
                            lines.append(p['summary'] + "\n")
                        if p['params']:
                            lines.append("##### Parameters\n")
                            for pp in p['params']:
                                name = f"`{pp['name']}`" if pp['name'] else ''
                                typ = f" ({pp['type']})" if pp['type'] else ''
                                lines.append(f"- {name}{typ}: {pp['desc']}")
                            lines.append("")
                        if p['returns']:
                            lines.append("##### Returns\n")
                            for rr in p['returns']:
                                name = f"`{rr['name']}`" if rr['name'] else ''
                                typ = f" ({rr['type']})" if rr['type'] else ''
                                colon = ': ' if name or typ else ''
                                lines.append(f"- {name}{typ}{colon}{rr['desc']}")
                            lines.append("")
                    lines.append("</details>\n")

    # Functions
    funcs = mod.get('functions', [])
    if funcs:
        lines.append("## Functions\n")
        for f in funcs:
            sig = fmt_signature(f['name'], f.get('args', []))
            lines.append(f"### `{sig}`\n")
            fdoc = mdx_escape(f.get('doc', ''))
            if fdoc:
                p = parse_numpy_doc(fdoc)
                if p['summary']:
                    lines.append(p['summary'] + "\n")
                if p['params']:
                    lines.append("#### Parameters\n")
                    for pp in p['params']:
                        name = f"`{pp['name']}`" if pp['name'] else ''
                        typ = f" ({pp['type']})" if pp['type'] else ''
                        lines.append(f"- {name}{typ}: {pp['desc']}")
                    lines.append("")
                if p['returns']:
                    lines.append("#### Returns\n")
                    for rr in p['returns']:
                        name = f"`{rr['name']}`" if rr['name'] else ''
                        typ = f" ({rr['type']})" if rr['type'] else ''
                        colon = ': ' if name or typ else ''
                        lines.append(f"- {name}{typ}{colon}{rr['desc']}")
                    lines.append("")

    out.write_text("\n".join(lines), encoding='utf-8')

def write_index(mod_names):
    out = OUT_DIR / 'page.mdx'
    lines = ["# API Reference\n", "Browse the public API exposed by `metbit` (sorted by category).\n"]

    # Build symbol -> module map from __init__.py
    public_api = parse_public_api()

    # Group symbols by category
    sym_groups = {k: [] for k in CATEGORY_ORDER}
    for sym, mod in sorted(public_api, key=lambda x: x[0].lower()):
        cat = SYMBOL_CATEGORY.get(sym) or MODULE_CATEGORY.get(mod) or 'other'
        if cat not in sym_groups:
            sym_groups['other'].append((sym, mod))
        else:
            sym_groups[cat].append((sym, mod))

    # Write index sections
    for key in CATEGORY_ORDER:
        items = sym_groups.get(key, [])
        if not items:
            continue
        lines.append(f"## {CATEGORY_TITLES.get(key, key.title())}\n")
        for sym, mod in items:
            lines.append(f"- [{sym}](/docs/api/{mod})")
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding='utf-8')

def parse_public_api():
    """Return list of (symbol, module) exported by metbit.__init__.

    Supports 'from .mod import name1, name2' and star imports by
    enumerating public classes/functions in the module.
    """
    symbols: list[tuple[str, str]] = []
    if not INIT_FILE.exists():
        return symbols
    src = INIT_FILE.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(src)
    except Exception:
        return symbols
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            # Only relative imports from this package
            if node.module is None:
                continue
            mod = node.module.lstrip('.')
            if mod.startswith('.'):  # overly defensive
                mod = mod.strip('.')
            if any(alias.name == '*' for alias in node.names):
                # Expand star: scan module for public top-level classes/functions
                target = PKG_DIR / f"{mod}.py"
                if target.exists():
                    m = parse_module(target)
                    for cls in m.get('classes', []):
                        if not cls['name'].startswith('_'):
                            symbols.append((cls['name'], mod))
                    for fn in m.get('functions', []):
                        if not fn['name'].startswith('_'):
                            symbols.append((fn['name'], mod))
                continue
            for alias in node.names:
                name = alias.asname or alias.name
                if not name.startswith('_'):
                    symbols.append((name, mod))
    return symbols

def main():
    if not PKG_DIR.exists():
        raise SystemExit(f"Package directory not found: {PKG_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modules = []
    mod_names = []
    for entry in sorted(PKG_DIR.iterdir()):
        if entry.is_dir():
            # skip subpackages and dev assets
            continue
        if entry.suffix != '.py':
            continue
        name = entry.stem
        if name in EXCLUDE_MODULES or name.startswith('_'):
            continue
        mod = parse_module(entry)
        modules.append(mod)
        mod_names.append(name)
        write_module_page(mod)
    write_index(mod_names)
    print(f"Generated API docs for {len(mod_names)} modules into {OUT_DIR}")

if __name__ == '__main__':
    main()
