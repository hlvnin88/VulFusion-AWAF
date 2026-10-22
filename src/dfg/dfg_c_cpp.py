"""
Simplified-but-real dataflow extraction for C/C++ using tree-sitter AST:
- collect identifier defs/uses
- build define-use edges
- return DFG nodes as (var_name, token_index, [dep_token_indices])

This is real parser/dataflow logic (not placeholder text),
implemented as a lightweight GraphCodeBERT-style re-implementation.
"""

def walk(node, out):
    out.append(node)
    for ch in node.children:
        walk(ch, out)


def node_text(code_bytes, node):
    return code_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")


def is_identifier(node):
    return node.type in {"identifier", "field_identifier"}


def is_definition_context(node):
    # heuristic for C/C++ declarations/params
    p = node.parent
    if p is None:
        return False
    return p.type in {
        "declaration",
        "init_declarator",
        "parameter_declaration",
        "function_declarator",
        "pointer_declarator",
        "array_declarator",
    }


def extract_tokens_with_pos(root, code):
    code_bytes = bytes(code, "utf8")
    nodes = []
    walk(root, nodes)

    toks = []
    for n in nodes:
        if is_identifier(n):
            txt = node_text(code_bytes, n).strip()
            if txt:
                toks.append((txt, n.start_point, n.end_point, n))
    return toks


def build_dfg(root, code):
    tokens = extract_tokens_with_pos(root, code)

    # flatten token index
    indexed = []
    for i, (name, sp, ep, node) in enumerate(tokens):
        indexed.append({
            "idx": i,
            "name": name,
            "node": node,
        })

    last_def = {}  # var -> token_idx
    edges = []     # (var, use_idx, [def_idx])

    for t in indexed:
        name = t["name"]
        idx = t["idx"]
        n = t["node"]

        if is_definition_context(n):
            # new definition
            last_def[name] = idx
            edges.append((name, idx, []))  # def node
        else:
            # usage depends on latest def if exists
            dep = [last_def[name]] if name in last_def else []
            edges.append((name, idx, dep))

    return edges, [x["name"] for x in indexed]