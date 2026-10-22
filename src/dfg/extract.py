from .parser_utils import build_parsers, parse_code
from .dfg_c_cpp import build_dfg


class DFGExtractor:
    def __init__(self, tree_sitter_so: str):
        self.parsers = build_parsers(tree_sitter_so)

    def norm_lang(self, language: str):
        l = (language or "").lower().strip()
        if l in {"c++", "cpp", "cc", "cxx"}:
            return "cpp"
        if l == "c":
            return "c"
        return "cpp"

    def extract(self, code: str, language: str):
        lang = self.norm_lang(language)
        parser = self.parsers[lang]
        tree = parse_code(parser, code)
        root = tree.root_node
        dfg, token_names = build_dfg(root, code)
        return {
            "dfg": dfg,                # list[(var, idx, deps)]
            "token_names": token_names
        }