from tree_sitter import Language, Parser


def build_parsers(so_path: str):
    lang_c = Language(so_path, "c")
    lang_cpp = Language(so_path, "cpp")

    parser_c = Parser()
    parser_c.set_language(lang_c)

    parser_cpp = Parser()
    parser_cpp.set_language(lang_cpp)

    return {
        "c": parser_c,
        "cpp": parser_cpp,
    }


def parse_code(parser, code: str):
    return parser.parse(bytes(code, "utf8"))