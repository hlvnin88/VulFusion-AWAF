import argparse
from tree_sitter import Language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--repos", nargs="+", required=True)
    args = ap.parse_args()

    Language.build_library(args.output, args.repos)
    print(f"[OK] Built language library: {args.output}")


if __name__ == "__main__":
    main()