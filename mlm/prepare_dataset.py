#!/usr/bin/env python3
"""Utility to prepare dataset files."""
import argparse


def prepare(paths_file: str, output_file: str) -> None:
    with open(paths_file, "r") as f:
        paths = f.readlines()

    with open(output_file, "w") as out:
        for path in paths:
            result = path.split(":")
            if len(result) == 2:
                mutations = result[1].strip()
                if len(mutations.split(',')) > 2:
                    out.write(mutations + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from path list")
    parser.add_argument("paths", help="File containing paths")
    parser.add_argument("output", help="Output file for mutations")
    args = parser.parse_args()
    prepare(args.paths, args.output)


if __name__ == "__main__":
    main()
