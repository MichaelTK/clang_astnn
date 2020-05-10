import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
import pickle
from tree_clang import ASTNode, SingleNode
import numpy as np


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
    print(node)
    node_split = node.split(",")
    #print(node_split)
    try:
        name = node_split[2]
    except Exception as e:
        return
    children = findChildren(node,entry)
    if name in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
        block_seq.append(ASTNode(node,entry))
        if name is not 'FOR_STMT':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            child_split = node.split(",")
            child_name = child_split[2]
            if child_name not in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT', 'COMPOUND_STMT']:
                block_seq.append(ASTNode(child,entry))
            get_blocks_for_node(child,block_seq,entry)
    elif name is 'COMPOUND_STMT':
        block_seq.append(ASTNode(name,entry))
        for child in children:
            child_split = node.split(",")
            child_name = child_split[2]
            if child_name not in ['IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
                block_seq.append(ASTNode(child,entry))
            get_blocks_for_node(child, block_seq, entry)
        block_seq.append(ASTNode('End',entry))
    else:
        for child in children:
            get_blocks_for_node(child, block_seq, entry)


def get_blocks(node, block_seq, entry):
    entry_split = entry.split("¬")
    start_node = entry_split[0]
    node_split = start_node.split(",")

    name = node_split[2]
    children = findChildren(start_node,entry)

    if name in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
        block_seq.append(ASTNode(node,entry))
        if name is not 'FOR_STMT':
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            child_split = node.split(",")
            child_name = child_split[2]
            if child_name not in ['FUNCTION_DECL', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT', 'COMPOUND_STMT']:
                block_seq.append(ASTNode(child,entry))
            get_blocks_for_node(child, block_seq, entry)
    elif name is 'COMPOUND_STMT':
        block_seq.append(ASTNode(name,entry))
        for child in children:
            child_split = node.split(",")
            child_name = child_split[2]
            if child_name not in ['IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']:
                block_seq.append(ASTNode(child,entry))
            get_blocks_for_node(child, block_seq, entry)
        block_seq.append(ASTNode('End',entry))
    else:
        for child in children:
            get_blocks_for_node(child, block_seq, entry)


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
