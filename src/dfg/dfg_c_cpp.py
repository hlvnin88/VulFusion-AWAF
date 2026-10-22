"""
Dataflow extraction C/C++ bằng tree-sitter (define-use edges).

Trả về DFG nodes dạng (var_name, token_index, [dep_token_indices])
- Đây là DFG "define-use" đơn giản, đủ để build dfg_to_code / dfg_to_dfg.
- Không phải full reaching-definitions analysis, nhưng là logic thật (không placeholder).
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
    indexed = [{"idx": i, "name": name, "node": n}
               for i, (name, _sp, _ep, n) in enumerate(tokens)]

    last_def = {}
    edges = []

    for t in indexed:
        name = t["name"]
        idx = t["idx"]
        n = t["node"]

        if is_definition_context(n):
            last_def[name] = idx
            edges.append((name, idx, []))
        else:
            dep = [last_def[name]] if name in last_def else []
            edges.append((name, idx, dep))

    return edges, [x["name"] for x in indexed]