"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True

    >>> d = build_frequency_dict(bytes([]))
    >>> d == {}
    True
    """
    dictionary = {}
    for byte in text:
        if byte in dictionary:
            dictionary[byte] += 1
        else:
            dictionary[byte] = 1
    return dictionary


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, None, None)
    >>> t == result
    True
    >>> freq = {1: 1, 2: 1, 3:2, 4:2, 5:3, 6:3}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
    ... HuffmanTree(1, None, None), HuffmanTree(2, None, None)), HuffmanTree(5,
    ... None, None)), HuffmanTree(None, HuffmanTree(6, None, None),
    ... HuffmanTree(None, HuffmanTree(3, None, None), HuffmanTree(4,
    ... None, None))))
    >>> t == result
    True
    >>> freq = {1: 1, 2: 1, 3:1, 4:1, 5:1}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(None, HuffmanTree(3, None, None),
    ... HuffmanTree(4, None, None)), HuffmanTree(None,
    ... HuffmanTree(5, None, None), HuffmanTree(None,
    ... HuffmanTree(1, None, None), HuffmanTree(2, None, None))))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    # If freq_dict is empty
    if len(freq_dict) == 0:
        return HuffmanTree(None)
    # If freq_dict only has one symbol-frequency pair
    elif len(freq_dict) == 1:
        symbol = 0
        for key, _ in freq_dict.items():
            symbol = key

        left = HuffmanTree((symbol + 1) % 256)
        right = HuffmanTree(symbol)
        return HuffmanTree(None, left, right)

    # Create, sort and store symbol trees in list
    sorted_value_dicts = _sort_dict_by_values(freq_dict)
    tree_nodes = []
    for sorted_value in sorted_value_dicts:
        for symbol, frequency in sorted_value.items():
            new_node = HuffmanTree(symbol)
            # Use number attribute to temporarily store frequency value
            new_node.number = frequency
            tree_nodes.append(new_node)

    # Combine symbol trees to make and return main tree
    while len(tree_nodes) > 1:
        left = tree_nodes.pop(0)
        right = tree_nodes.pop(0)
        connection_node = HuffmanTree(None, left, right)
        connection_node.number = left.number + right.number
        tree_nodes.append(connection_node)
        tree_nodes = _sort_tree_nodes(tree_nodes)
    final_tree = tree_nodes[0]
    _strip_numbering(final_tree)
    return final_tree


def _sort_tree_nodes(tree_nodes: list[HuffmanTree]) -> list[HuffmanTree]:
    """
    Re-sort the nodes in tree_nodes from lowest to largest in frequency
    """
    new_tree_nodes = []
    tree_nodes_len = len(tree_nodes)
    counter = 0
    while counter != tree_nodes_len:
        lowest_index = 0
        for check in range(len(tree_nodes)):
            if tree_nodes[lowest_index].number > tree_nodes[check].number:
                lowest_index = check
            # Gives preference to symbol trees over non-symbol trees
            elif tree_nodes[lowest_index].symbol is None \
                    and tree_nodes[check].symbol is not None \
                    and tree_nodes[lowest_index].number == \
                    tree_nodes[check].number:
                lowest_index = check
        new_tree_nodes.append(tree_nodes.pop(lowest_index))
        counter = len(new_tree_nodes)
    return new_tree_nodes


def _strip_numbering(tree: HuffmanTree) -> None:
    """
    Set <number> attribute for all nodes of tree to None
    """
    tree.number = None
    if tree.left is not None:
        _strip_numbering(tree.left)
    if tree.right is not None:
        _strip_numbering(tree.right)
    return


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> d = get_codes(HuffmanTree(None))
    >>> d == {}
    True
    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> freq = {1: 1, 2: 1, 3:2, 4:2, 5:3, 6:3}
    >>> t = build_huffman_tree(freq)
    >>> d = get_codes(t)
    >>> d == {1:"000", 2:"001", 3:"110", 4:"111", 5:"01", 6:"10"}
    True
    >>> freq = {1: 1, 2: 1, 3:1, 4:1, 5:1}
    >>> t = build_huffman_tree(freq)
    >>> d = get_codes(t)
    >>> d == {1:"110", 2:"111", 3:"00", 4:"01", 5:"10"}
    True
    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> d = get_codes(t)
    >>> d == {2:"1", 3:"0"}
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> d = get_codes(t)
    >>> d == {2:"0", 3:"10", 7:"11"}
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> d = get_codes(t)
    >>> d == {t.left.symbol:"0", symbol:"1"}
    True
    """
    return _get_codes_helper(tree, "0", "1")


def _get_codes_helper(tree: HuffmanTree, left: str,
                      right: str) -> dict[int, str]:
    """
    This helper function returns a dictionary which maps symbols from the
    Huffman tree <tree> to codes. Function relies on normally traversing the
    tree and on recursion.
    """
    codes = {}
    if tree.right is not None:
        if tree.right.symbol is not None:
            codes.update({tree.right.symbol: right})
        # Update parameters to match next tree level
        codes.update(_get_codes_helper(tree.right, right + "0", right + "1"))

    if tree.left is not None:
        if tree.left.symbol is not None:
            codes.update({tree.left.symbol: left})
        codes.update(_get_codes_helper(tree.left, left + "0", left + "1"))

    return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> tree = HuffmanTree(None)
    >>> number_nodes(tree)
    >>> tree.number == None
    True
    >>> tree = HuffmanTree(None, HuffmanTree(4), HuffmanTree(3))
    >>> number_nodes(tree)
    >>> tree.number == 0 and tree.right.number == None
    True
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    _number_nodes_helper(tree, 0)


def _number_nodes_helper(tree: HuffmanTree, numbering: int) -> int:
    """
    Helper function that numbers internal nodes in <tree> according to postorder
    traversal. The numbering starts at 0. Updates numbering parameter to reflect
    current node number.
    """
    if tree.left is None and tree.right is None:
        return numbering
    else:
        lefty = _number_nodes_helper(tree.left, numbering)
        righty = _number_nodes_helper(tree.right, lefty)
        tree.number = righty
        return tree.number + 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq1 = {3: 2, 2: 7}
    >>> tree1 = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> freq2 = {9:2}
    >>> tree2 = HuffmanTree(9)
    >>> freq3 = {}
    >>> tree3 = HuffmanTree(None)
    >>> avg_length(tree1, freq1)  # (2*1 + 7*1) / (2 + 7)
    1.0
    >>> avg_length(tree2, freq2) # (9*0) / (2)
    0.0
    >>> avg_length(tree3, freq3)
    0.0
    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    if tree.right is None and tree.left is None:
        return 0.0
    weighted_sum = 0
    total = sum(i for i in freq_dict.values())
    # Use code length to find level of each node
    code_dict = get_codes(tree)
    for key in freq_dict:
        weighted_sum += freq_dict[key] * len(code_dict[key])
    return weighted_sum / total


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> compress_bytes([], {}) == bytes([])
    True
    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> result = compress_bytes(b'helloworld', get_codes(tree))
    >>> [byte_to_bits(byte) for byte in result]
    ['00000110', '10111010', '11101110', '11000000']
    """
    combined = ""
    bytes_lst = []
    # Accumulate codes in order as one large string
    for symbol in text:
        combined += codes[symbol]
    # Separate giant code string into bytes
    for i in range(0, len(combined), 8):
        bytes_lst.append(bits_to_byte(combined[i:i + 8]))
    return bytes(bytes_lst)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    []
    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None),
    ... HuffmanTree(12, None, None)), HuffmanTree(None,
    ... HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 10, 0, 12, 0, 5, 0, 7, 1, 0, 1, 1]
    """
    bytes_lst = []  # Relies on aliasing
    bytes_lst = _tree_to_bytes_helper(tree, bytes_lst)[1]
    return bytes(bytes_lst)


def _tree_to_bytes_helper(tree: HuffmanTree, bytes_lst: list) -> list[int]:
    """ Helper function that numbers internal nodes in <tree> according to
    postorder traversal. The numbering starts at 0.
    """
    if tree.left is None and tree.right is None:
        return tree, bytes_lst
    else:
        lefty = _tree_to_bytes_helper(tree.left, bytes_lst)
        righty = _tree_to_bytes_helper(tree.right, bytes_lst)
        bytes_lst = righty[1]

        # Appends the 4 bytes based on whether tree is leaf or connecting node
        if lefty[0].number is None:
            bytes_lst.append(0)
            bytes_lst.append(lefty[0].symbol)
        else:
            bytes_lst.append(1)
            bytes_lst.append(lefty[0].number)
        if righty[0].number is None:
            bytes_lst.append(0)
            bytes_lst.append(righty[0].symbol)
        else:
            bytes_lst.append(1)
            bytes_lst.append(righty[0].number)
        return tree, bytes_lst


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(0, 3, 0, 2)]
    >>> generate_tree_general(lst, 0)
    HuffmanTree(None, HuffmanTree(3, None, None), \
HuffmanTree(2, None, None))
    >>> lst = [ReadNode(1, 1, 0, 5), ReadNode(0, 3, 0, 2)]
    >>> generate_tree_general(lst, 0)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(3, None, None), \
HuffmanTree(2, None, None)), HuffmanTree(5, None, None))
    >>> lst = [ReadNode(0, 100, 0, 111), ReadNode(0, 119, 0, 114), \
    ReadNode(1, 3, 1, 1), ReadNode(0, 104, 0, 101), ReadNode(1, 2, 1, 5), \
    ReadNode(0, 108, 1, 0)]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> generate_tree_general(lst, 4) == tree
    True
    >>> lst = [ReadNode(0, 10, 0, 12), ReadNode(0, 5, 0, 7), \
    ReadNode(1, 0, 1, 1)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)))
    """
    # Create list of non-symbol node trees
    new_node_lst = [HuffmanTree(None) for _ in range(len(node_lst))]
    for i, node in enumerate(node_lst):
        new_node_lst[i].number = i
        if node.l_type == 0:  # Create and attach leaf
            new_node_lst[i].left = HuffmanTree(node.l_data)
        else:  # Connect non-symbol node to next node
            new_node_lst[i].left = new_node_lst[node.l_data]
        if node.r_type == 0:
            new_node_lst[i].right = HuffmanTree(node.r_data)
        else:
            new_node_lst[i].right = new_node_lst[node.r_data]
    return new_node_lst[root_index]  # Returns root node


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> lst = [ReadNode(0, 3, 0, 2)]
    >>> generate_tree_postorder(lst, 0)
    HuffmanTree(None, HuffmanTree(3, None, None), HuffmanTree(2, None, None))
    >>> lst = [ReadNode(0, 3, 0, 2), ReadNode(1, 0, 0, 5)]
    >>> generate_tree_postorder(lst, 1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(3, None, None), \
HuffmanTree(2, None, None)), HuffmanTree(5, None, None))
    >>> lst = [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114), \
    ReadNode(1, 0, 1, 1), ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, 3), \
    ReadNode(1, 2, 1, 4)]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> tree == generate_tree_postorder(lst, 5)
    True
    >>> lst = [ReadNode(0, 10, 0, 12), ReadNode(0, 5, 0, 7), \
    ReadNode(1, 0, 1, 1)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), HuffmanTree(None, \
HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(0, 60, 0, 61), ReadNode(0, 62, 0, 63), \
    ReadNode(1, 0, 1, 1), ReadNode(1, 2, 0, 64), ReadNode(1, 3, 0, 65)]
    >>> generate_tree_postorder(lst, 4)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
HuffmanTree(60, None, None), HuffmanTree(61, None, None)), \
HuffmanTree(None, HuffmanTree(62, None, None), HuffmanTree(63, None, \
None))), HuffmanTree(64, None, None)), HuffmanTree(65, None, None))
    >>> lst = [ReadNode(0, 60, 0, 61), ReadNode(1, 0, 0, 62), \
    ReadNode(0, 63, 0, 64), ReadNode(0, 65, 0, 66), ReadNode(1, 2, 1, 3), \
    ReadNode(0, 67, 0, 68), ReadNode(1, 4, 1, 5), ReadNode(1, 1, 1, 6)]
    >>> generate_tree_postorder(lst, 7)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(60, \
None, None), HuffmanTree(61, None, None)), HuffmanTree(62, None, None)), \
HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(63, \
None, None), HuffmanTree(64, None, None)), HuffmanTree(None, \
HuffmanTree(65, None, None), HuffmanTree(66, None, None))), \
HuffmanTree(None, HuffmanTree(67, None, None), HuffmanTree(68, None, None))))
    >>> lst = [ReadNode(0, 60, 0, 61), ReadNode(1, 0, 0, 62), \
    ReadNode(0, 63, 0, 64), ReadNode(1, 2, 0, 65), ReadNode(0, 67, 0, 68), \
    ReadNode(1, 3, 1, 4), ReadNode(1, 1, 1, 5)]
    >>> generate_tree_postorder(lst, 6)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(60, \
None, None), HuffmanTree(61, None, None)), HuffmanTree(62, None, None)), \
HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(63, None, \
None), HuffmanTree(64, None, None)), HuffmanTree(65, None, None)), \
HuffmanTree(None, HuffmanTree(67, None, None), HuffmanTree(68, None, None))))
    >>> lst = [ReadNode(0, 1, 0, 2), ReadNode(1, -1, 0, 3), \
    ReadNode(0, 4, 0, 5), ReadNode(0, 6, 1, -1), ReadNode(1, -1, 1, -1), \
    ReadNode(0, 7, 0, 8), ReadNode(0, 9, 0, 10), ReadNode(1, -1, 1, -1), \
    ReadNode(0, 11, 0, 12), ReadNode(0, 13, 0, 14), ReadNode(1, -1, 1, -1), \
    ReadNode(1, -1, 1, -1), ReadNode(1, -1, 0, 15), ReadNode(1, -1, 0, 16), \
    ReadNode(0, 17, 0, 18), ReadNode(1, -1, 1, -1), ReadNode(0, 19, 1, -1), \
    ReadNode(1, -1, 1, -1), ReadNode(0, 20, 0, 21), ReadNode(0, 22, 0, 23), \
    ReadNode(0, 24, 0, 25), ReadNode(1, -1, 1, -1), ReadNode(1, -1, 1, -1), \
    ReadNode(1, -1, 0, 26), ReadNode(1, -1, 1, -1)]
    >>> generate_tree_postorder(lst, len(lst) - 1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(1, None, None), HuffmanTree(2, None, None)), \
HuffmanTree(3, None, None)), HuffmanTree(None, HuffmanTree(6, None, None), \
HuffmanTree(None, HuffmanTree(4, None, None), HuffmanTree(5, None, None)))), \
HuffmanTree(None, HuffmanTree(19, None, None), HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(8, None, None)), \
HuffmanTree(None, HuffmanTree(9, None, None), HuffmanTree(10, None, None))), \
HuffmanTree(None, HuffmanTree(None, HuffmanTree(11, None, None), \
HuffmanTree(12, None, None)), HuffmanTree(None, HuffmanTree(13, None, None), \
HuffmanTree(14, None, None)))), HuffmanTree(15, None, None)), HuffmanTree(16, \
None, None)), HuffmanTree(None, HuffmanTree(17, None, None), HuffmanTree(18, \
None, None))))), HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
HuffmanTree(20, None, None), HuffmanTree(21, None, None)), HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(22, None, None), HuffmanTree(23, None, None)), \
HuffmanTree(None, HuffmanTree(24, None, None), HuffmanTree(25, None, None)))), \
HuffmanTree(26, None, None)))
    >>> generate_tree_postorder(lst, 23)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(20, \
None, None), HuffmanTree(21, None, None)), HuffmanTree(None, HuffmanTree(None, \
HuffmanTree(22, None, None), HuffmanTree(23, None, None)), HuffmanTree(None, \
HuffmanTree(24, None, None), HuffmanTree(25, None, None)))), HuffmanTree(26, \
None, None))
    >>> generate_tree_postorder(lst, 11)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(7, \
None, None), HuffmanTree(8, None, None)), HuffmanTree(None, HuffmanTree(9, \
None, None), HuffmanTree(10, None, None))), HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(11, None, None), HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(13, None, None), HuffmanTree(14, None, None))))
    >>> generate_tree_postorder(lst, 4)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, HuffmanTree(1, \
None, None), HuffmanTree(2, None, None)), HuffmanTree(3, None, None)), \
HuffmanTree(None, HuffmanTree(6, None, None), HuffmanTree(None, HuffmanTree(4, \
None, None), HuffmanTree(5, None, None))))
    >>> generate_tree_postorder(lst, 14)
    HuffmanTree(None, HuffmanTree(17, None, None), HuffmanTree(18, None, None))
    """
    # Labels the node_list properly in post_order, then creates tree
    root_number = root_index
    new_node_lst = node_lst[:root_index + 1]
    _generate_tree_postorder_helper(new_node_lst, root_number, root_number - 1)
    return _generate_tree_postorder_helper2(new_node_lst, root_index)


def _generate_tree_postorder_helper2(node_lst: list[ReadNode],
                                     root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    Similar/same as generate_tree_general
    """
    # Create list of non-symbol node trees
    new_node_lst = [HuffmanTree(None) for _ in range(len(node_lst))]
    for i, node in enumerate(node_lst):
        new_node_lst[i].number = i
        if node.l_type == 0:  # Create and attach leaf
            new_node_lst[i].left = HuffmanTree(node.l_data)
        else:  # Connect non-symbol node to next node
            new_node_lst[i].left = new_node_lst[node.l_data]
        if node.r_type == 0:
            new_node_lst[i].right = HuffmanTree(node.r_data)
        else:
            new_node_lst[i].right = new_node_lst[node.r_data]
    return new_node_lst[root_index]  # Returns root node


def _generate_tree_postorder_helper(node_lst: list[ReadNode], current_node: int,
                                    post_numbering: int) -> int:
    """
    >>> hidden = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 2, 1)
    >>> hidden
    [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
ReadNode(1, 0, 1, 1)]

    >>> hidden = [ReadNode(0, 3, 0, 2)]
    >>> hi = _generate_tree_postorder_helper(hidden, 0, -1)
    >>> hidden
    [ReadNode(0, 3, 0, 2)]

    >>> hidden = [ReadNode(0, 3, 0, 2), ReadNode(1, -1, 0, 5)]
    >>> hi = _generate_tree_postorder_helper(hidden, 1, 0)
    >>> hidden
    [ReadNode(0, 3, 0, 2), ReadNode(1, 0, 0, 5)]

    >>> hidden = [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114), \
    ReadNode(1, -1, 1, -1), ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, -1), \
    ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 5, 4)
    >>> hidden
    [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114), \
ReadNode(1, 0, 1, 1), ReadNode(0, 100, 0, 111), ReadNode(0, 108, 1, 3), \
ReadNode(1, 2, 1, 4)]

    >>> hidden = [ReadNode(0, 10, 0, 12), ReadNode(0, 5, 0, 7), \
    ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 2, 1)
    >>> hidden
    [ReadNode(0, 10, 0, 12), ReadNode(0, 5, 0, 7), ReadNode(1, 0, 1, 1)]

    >>> hidden = [ReadNode(0, 60, 0, 61), ReadNode(0, 62, 0, 63), \
    ReadNode(1, 0, 1, 1), ReadNode(1, 2, 0, 64), ReadNode(1, 3, 0, 65)]
    >>> hi = _generate_tree_postorder_helper(hidden, 4, 3)
    >>> hidden
    [ReadNode(0, 60, 0, 61), ReadNode(0, 62, 0, 63), ReadNode(1, 0, 1, 1), \
ReadNode(1, 2, 0, 64), ReadNode(1, 3, 0, 65)]

    >>> hidden = [ReadNode(0, 60, 0, 61), ReadNode(1, -1, 0, 62), \
    ReadNode(0, 63, 0, 64), ReadNode(0, 65, 0, 66), ReadNode(1, -1, 1, -1), \
    ReadNode(0, 67, 0, 68), ReadNode(1, -1, 1, -1), ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 7, 6)
    >>> hidden
    [ReadNode(0, 60, 0, 61), ReadNode(1, 0, 0, 62), ReadNode(0, 63, 0, 64), \
ReadNode(0, 65, 0, 66), ReadNode(1, 2, 1, 3), ReadNode(0, 67, 0, 68), \
ReadNode(1, 4, 1, 5), ReadNode(1, 1, 1, 6)]

    >>> hidden = [ReadNode(0, 60, 0, 61), ReadNode(1, -1, 0, 62), \
    ReadNode(0, 63, 0, 64), ReadNode(1, -1, 0, 65), ReadNode(0, 67, 0, 68), \
    ReadNode(1, -1, 1, -1), ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 6, 5)
    >>> hidden
    [ReadNode(0, 60, 0, 61), ReadNode(1, 0, 0, 62), ReadNode(0, 63, 0, 64), \
ReadNode(1, 2, 0, 65), ReadNode(0, 67, 0, 68), ReadNode(1, 3, 1, 4), \
ReadNode(1, 1, 1, 5)]

    >>> hidden = [ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), \
    ReadNode(0, 5, 0, 6), ReadNode(0, 7, 0, 8), ReadNode(1, -1, 1, -1), \
    ReadNode(1, -1, 1, -1), ReadNode(1, -1, 0, 9), ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 7, 6)
    >>> hidden
    [ReadNode(0, 1, 0, 2), ReadNode(0, 3, 0, 4), ReadNode(0, 5, 0, 6), \
ReadNode(0, 7, 0, 8), ReadNode(1, 2, 1, 3), ReadNode(1, 1, 1, 4), \
ReadNode(1, 5, 0, 9), ReadNode(1, 0, 1, 6)]

    >>> hidden = [ReadNode(0, 1, 0, 2), ReadNode(1, -1, 0, 3), \
    ReadNode(0, 4, 0, 5), ReadNode(0, 6, 1, -1), ReadNode(1, -1, 1, -1), \
    ReadNode(0, 7, 0, 8), ReadNode(1, -1, 0, 9), ReadNode(0, 10, 0, 11), \
    ReadNode(1, -1, 1, -1), ReadNode(0, 12, 1, -1), ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 10, 9)
    >>> hidden
    [ReadNode(0, 1, 0, 2), ReadNode(1, 0, 0, 3), \
ReadNode(0, 4, 0, 5), ReadNode(0, 6, 1, 2), ReadNode(1, 1, 1, 3), \
ReadNode(0, 7, 0, 8), ReadNode(1, 5, 0, 9), ReadNode(0, 10, 0, 11), \
ReadNode(1, 6, 1, 7), ReadNode(0, 12, 1, 8), ReadNode(1, 4, 1, 9)]

    >>> hidden = [ReadNode(0, 1, 0, 2), ReadNode(1, -1, 0, 3), \
    ReadNode(0, 4, 0, 5), ReadNode(0, 6, 1, -1), ReadNode(1, -1, 1, -1), \
    ReadNode(0, 7, 0, 8), ReadNode(0, 9, 0, 10), ReadNode(1, -1, 1, -1), \
    ReadNode(0, 11, 0, 12), ReadNode(0, 13, 0, 14), ReadNode(1, -1, 1, -1), \
    ReadNode(1, -1, 1, -1), ReadNode(1, -1, 0, 15), ReadNode(1, -1, 0, 16), \
    ReadNode(0, 17, 0, 18), ReadNode(1, -1, 1, -1), ReadNode(0, 19, 1, -1), \
    ReadNode(1, -1, 1, -1), ReadNode(0, 20, 0, 21), ReadNode(0, 22, 0, 23), \
    ReadNode(0, 24, 0, 25), ReadNode(1, -1, 1, -1), ReadNode(1, -1, 1, -1), \
    ReadNode(1, -1, 0, 26), ReadNode(1, -1, 1, -1)]
    >>> hi = _generate_tree_postorder_helper(hidden, 24, 23)
    >>> hidden
    [ReadNode(0, 1, 0, 2), ReadNode(1, 0, 0, 3), \
ReadNode(0, 4, 0, 5), ReadNode(0, 6, 1, 2), ReadNode(1, 1, 1, 3), \
ReadNode(0, 7, 0, 8), ReadNode(0, 9, 0, 10), ReadNode(1, 5, 1, 6), \
ReadNode(0, 11, 0, 12), ReadNode(0, 13, 0, 14), ReadNode(1, 8, 1, 9), \
ReadNode(1, 7, 1, 10), ReadNode(1, 11, 0, 15), ReadNode(1, 12, 0, 16), \
ReadNode(0, 17, 0, 18), ReadNode(1, 13, 1, 14), ReadNode(0, 19, 1, 15), \
ReadNode(1, 4, 1, 16), ReadNode(0, 20, 0, 21), ReadNode(0, 22, 0, 23), \
ReadNode(0, 24, 0, 25), ReadNode(1, 19, 1, 20), ReadNode(1, 18, 1, 21), \
ReadNode(1, 22, 0, 26), ReadNode(1, 17, 1, 23)]
    """
    if node_lst[current_node].r_type == 1:
        node_lst[current_node].r_data = post_numbering
        post_numbering = _generate_tree_postorder_helper(node_lst,
                                                         current_node - 1,
                                                         post_numbering - 1)
    if node_lst[current_node].l_type == 1:
        node_lst[current_node].l_data = post_numbering
        post_numbering = _generate_tree_postorder_helper(node_lst,
                                                         post_numbering,
                                                         post_numbering - 1)
    return post_numbering


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), 5)
    b'hello'
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), 0)
    b''
    """
    # Create dict that maps symbol bits to symbol
    codes = get_codes(tree)
    codes = dict([(value, key) for key, value in codes.items()])
    # Combine bytes into one string
    bits = "".join([byte_to_bits(byte) for byte in text])

    # Map bits to symbol and append to list
    symbol_lst = []
    counter = 0
    start_index = 0
    end_index = 1
    while counter != size:
        if bits[start_index:end_index] in codes:
            symbol_lst.append(codes[bits[start_index:end_index]])
            start_index = end_index
            end_index += 1
            counter += 1
        else:
            end_index += 1
    return bytes(symbol_lst)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> tree = HuffmanTree(9)
    >>> improve_tree(tree, {9:1})
    >>> avg_length(tree, {})
    0.0
    >>> tree = HuffmanTree(None, HuffmanTree(9))
    >>> improve_tree(tree, {9:1})
    >>> avg_length(tree, {9:1})
    1.0
    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    if tree.left is None or tree.right is None:
        return
    # Helper functions map symbols and then update leaves to optimized symbol
    symbol_map = _symbol_mapping(tree, freq_dict)
    _update_tree(tree, symbol_map)


def _update_tree(tree: HuffmanTree, symbol_map: dict[int, int]) -> None:
    """
    Traverse tree and update leaf symbols to optimized frequency symbols
    """
    if tree.left is None and tree.right is None:
        tree.symbol = symbol_map[tree.symbol]
    else:
        _update_tree(tree.left, symbol_map)
        _update_tree(tree.right, symbol_map)


def _symbol_mapping(tree: HuffmanTree,
                    freq_dict: dict[int, int]) -> dict[int, int]:
    """
    Creates a dict that maps original tree symbols to what the symbols should
    actually be. Ex: {original tree symbol:optimized tree symbol} OR {99:97}
    """
    # Find the levels the tree's leaves are at
    codes = get_codes(tree)
    code_lst = []
    for symbol, s_code in codes.items():
        code_lst.append((symbol, len(s_code)))
    codes = dict(code_lst)
    # Sorts the leaves and their level from highest to lowest levels
    sorted_codes = _sort_dict_by_values(codes)
    # Sort symbols and frequency from highest to lowest frequency
    sorted_freq_dict = _sort_dict_by_values(freq_dict)
    sorted_freq_dict.reverse()

    # Create the dict that maps symbols to optimized symbols
    leave_dict = {}
    for i, original in enumerate(sorted_codes):
        before = list(original.keys())[0]
        after = list(sorted_freq_dict[i].keys())[0]
        leave_dict[before] = after
    return leave_dict


def _sort_dict_by_values(dictionary: dict) -> list[dict]:
    """
    Creates sorted miniature dicts based on value and appends to list
    Ex: [{:}, {:}]
    """
    sorted_values = sorted(dictionary.values())
    sorted_value_dicts = []
    for value in sorted_values:
        for key in dictionary:
            # Prevents re-adding similar value dicts
            if dictionary[key] == value and {key: value} \
                    not in sorted_value_dicts:
                sorted_value_dicts.append({key: value})
    return sorted_value_dicts


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
