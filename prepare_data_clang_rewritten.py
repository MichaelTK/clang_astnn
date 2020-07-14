import pandas as pd
import os
import re
import sys
import random
from gensim.models.word2vec import Word2Vec
import pickle
from tree_clang_rewritten import ASTNode, SingleNode
#from tree_clang_rewritten_generic import ASTNode, SingleNode
import numpy as np

index = 0

def get_sequences(node, sequence, entry):
    print("GETTING SEQUENCES")
    current = SingleNode(node)
    sequence.append(current.get_token())
    children = findChildren(node,entry)
    for child in children:
        get_sequences(child, sequence, entry)
    if current.get_token().lower() == 'compound_stmt':
        print("appending end")
        sequence.append('End')

def findChildren(node,entry):
    nodeSplit = node.split(",")
    children = []
    node_identifier = nodeSplit[0]
    entry_split = entry.split("¬")
    for entry_x in entry_split:
        node_split = entry_x.split(",")
        if node_split[-1] == node_identifier:
            children.append(entry_x)
    return children

def get_blocks_for_node(node,block_seq,entry):
    node_split = node.split(",")
    global index
    try:
        name = node_split[2]
    except Exception as e:
        return

    if 'KLc4GjIzs27prwhu' in entry:
        index = index + 1
        print(index)
        print(node)

    children = findChildren(node,entry)

    for child in children:
        child_split = child.split(",")
        ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chars=[]
        for i in range(16):
            chars.append(random.choice(ALPHABET))

        salt = "".join(chars)
        #depth = int(child_split[1])
        depth = 111

    children = findChildren(node,entry)

    for child in children:
        child_split = child.split(",")

        block_seq.append(ASTNode(child,entry))
        get_blocks_for_node(child,block_seq,entry)

        if child_split[2] == 'COMPOUND_STMT':
            block_seq.append(ASTNode('End',entry))

def find_translation_unit_node(entry):
    entry_split = entry.split('¬')
    node_to_return = ""
    for node in entry_split:
        if 'TRANSLATION_UNIT' in node:
            node_to_return = node

    return node_to_return

def get_blocks(node, block_seq, entry):

    entry_split = entry.split("¬")

    start_node = find_translation_unit_node(entry)
    node_split = start_node.split(",")

    name = node_split[2]
    children = findChildren(start_node,entry)

    block_seq.append(ASTNode(node,entry))

    for child in children:
        block_seq.append(ASTNode(child,entry))
        get_blocks_for_node(child,block_seq,entry)
        child_split = child.split(",")

        if child_split[2] is 'COMPOUND_STMT':
            block_seq.append(ASTNode('End',entry))

def get_blocks_pycparser(node, block_seq):
    children = node.get_children()
    name = node.kind.name
    if name in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
        block_seq.append(ASTNode(node))
        if name is not 'FOR_STMT':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.kind.name not in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT', 'COMPOUND_STMT']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name is 'COMPOUND_STMT':
        block_seq.append(ASTNode(name))
        for _, child in node.children():
            if child.kind.name not in ['IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode('End'))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)
