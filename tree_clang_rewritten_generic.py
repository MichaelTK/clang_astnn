from javalang.ast import Node
import sys

class ASTNode(object):
    def __init__(self, node, entry):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token(entry)
        self.children = self.add_children(entry)

    def is_leaf(self,entry):
        children = self.findChildren(entry)
        return len(children) == 0

    def get_token(self, entry):
        if 'End' in self.node.split(',')[0]:
            return 'End'
        return self.node.split(',')[2]


    def findChildren(self,entry):
        nodeSplit = self.node.split(",")
        children = []
        node_identifier = nodeSplit[0]
        entry_split = entry.split("¬")
        for entry_x in entry_split:
            node_split = entry_x.split(",")
            if node_split[-1] == node_identifier:
                children.append(entry_x)
        return children

    def add_children(self,entry):
        children = self.findChildren(entry)
        if len(children) == 0:
            return []
        return [ASTNode(child,entry) for child in children]


class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        children = self.findChildren(entry)
        return len(children) == 0

    def get_token(self, lower=True):
        if self.is_leaf(entry):
            if self.node[-3] is not '' and self.node[-3] is not ' ':
                return self.node[-3]
            else:
                return self.node[2]
        else:
            return self.node[2]

    def findChildren(self,entry):
        nodeSplit = self.node.split(",")
        children = []
        node_identifier = nodeSplit[0]
        entry_split = entry.split("¬")
        for entry_x in entry_split:
            node_split = entry_x.split(",")
            if node_split[-1] == node_identifier:
                children.append(entry_x)
        return children

    def add_children(self,entry):

        children = self.findChildren(entry)
        if len(children) == 0:
            return []
        return [ASTNode(child,entry) for child in children]
