"""
Profiler for Huffman Compression

Instructions:
Place this document in the same directory as your compress.py file
Run as you would any Python file (this uses default settings):
    python profile.py
For help and usage:
    python profile.py -h

There are two optional command-line arguments:
    -f, --file : to specify the path of the file you'd like to compress
    -n, --number : the number of iterations of compress/decompress

For example, to run with only 1 iteration:
    python profile.py -n 1
"""

import argparse
from filecmp import cmp
from timeit import timeit
import os
import sys
from compress import compress_file, decompress_file


def wrapper(f, *args):
    """Wrapper function for timeit compatibility."""
    def wrapped():
        return f(*args)
    return wrapped


def profile(infile, number):
    """Profile compression and decompression performance.
    
    Args:
        infile: Path to the input file to compress
        number: Number of iterations to run
    """
    compressed = infile + ".huf"
    decompressed = compressed + ".orig"

    compress_run_param = wrapper(compress_file, infile, compressed)
    decompress_run_param = wrapper(decompress_file, compressed, decompressed)

    if not os.path.exists(infile) or not os.path.isfile(infile):
        print(
            "ERROR! The path {} is either not a file or does not exist.".format(
                infile
            )
        )
        return

    print("======= Start Profile =======")

    print("Input File: {}".format(infile))
    print("Iterations: {}\n".format(number))

    print("Running Compression...")
    # Suppress "Bits per symbol" output from compress_file
    sys.stdout = open(os.devnull, "w")
    total_compress = timeit(compress_run_param, number=number)
    sys.stdout = sys.__stdout__
    print("Done Compression.\n")

    print("Running Decompression...")
    total_decompress = timeit(decompress_run_param, number=number)
    print("Done Decompression.\n")

    # Verify that decompression worked correctly
    if cmp(infile, decompressed):
        print("Success! Files match.\n")
        print("---- Stats ----")
        insize = os.path.getsize(infile)
        outsize = os.path.getsize(compressed)
        print("Original File Size: {:,} bytes".format(insize))
        print("Compressed File Size: {:,} bytes".format(outsize))
        print("Filesize Reduction: {:.2%}".format(1 - (outsize / insize)))
        print("Avg. Compression Time: {:.4f}s".format(total_compress / number))
        print(
            "Avg. Decompression Time: {:.4f}s".format(total_decompress / number)
        )
        print("-" * 15)
    else:
        print("ERROR! Decompressed file and original file are NOT the same.")

    print("=" * 29)

    # Clean up generated files
    try:
        if os.path.exists(compressed):
            os.remove(compressed)
        if os.path.exists(decompressed):
            os.remove(decompressed)
    except Exception as e:
        print("Warning: Could not clean up temporary files: {}".format(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profiler for CSC148 A2 Huffman Encoding",
        epilog="Must be placed in same directory as compress.py",
    )

    parser.add_argument(
        "-f",
        "--file",
        help="path to the target file, defaults to 'TestFiles/beemoviescript.txt'",
        default="TestFiles/beemoviescript.txt",
    )
    parser.add_argument(
        "-n",
        "--number",
        help="number of times to run compression/decompression, defaults to 10",
        default=10,
        type=int,
    )
    args = parser.parse_args()

    profile(infile=args.file, number=args.number)