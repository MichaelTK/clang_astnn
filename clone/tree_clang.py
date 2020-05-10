from javalang.ast import Node
import sys

class ASTNode(object):
    def __init__(self, node, entry):
        self.node = node
        # self.vocab = word_map
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token(entry)
        #print(node)
        # self.index = self.token_to_index(self.token)
        self.children = self.add_children(entry)

    def is_leaf_pycparser(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def is_leaf(self,entry):
        children = self.findChildren(entry)
        return len(children) == 0

    def get_token(self, entry):
        if self.is_leaf(entry):
            if self.node.split(',')[-3] is not '' and self.node.split(',')[-3] is not ' ':
                return self.node.split(',')[-3]
            else:
                return self.node.split(',')[2]
        else:
            return self.node.split(',')[2]

    def add_children_pycparser(self):
        if self.is_str:
            return []
        children = self.node.children()
        if self.token in ['FUNCTION_DECL', 'IF_STMT', 'WHILE_STMT', 'DO_STMT']:
            return [ASTNode(children[0][1])]
        elif self.token == 'FOR_STMT':
            return [ASTNode(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [ASTNode(child) for _, child in children]

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
        #print(self.node)
        #print("NODE_END")

        children = self.findChildren(entry)
        if len(children) == 0:
            return []
        return [ASTNode(child,entry) for child in children]


class BlockNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        children = self.findChildren(entry)
        return len(children) == 0

    def is_leaf_pycparser(self):
        if self.is_str:
            return True
        return len(self.node.children) == 0

    def get_token(self, node):
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):
            token = 'Modifier'
        elif isinstance(node, Node):
            token = node.kind.name
        else:
            token = ''
        return token

    def ori_children(self, root):
        if isinstance(root, Node):
            if self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
                children = root.children[:-1]
            else:
                children = root.children
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def add_children(self):
        if self.is_str:
            return []
        logic = ['SWITCH_STMT', 'IF_STMT', 'FOR_STMT', 'WHILE_STMT', 'DO_STMT']
        children = self.ori_children(self.node)
        if self.token in logic:
            return [BlockNode(children[0])]
        elif self.token in ['MethodDeclaration', 'ConstructorDeclaration']:
            return [BlockNode(child) for child in children]
        else:
            return [BlockNode(child) for child in children if self.get_token( child) not in logic]

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
        #print(self.node)
        #print("NODE_END")

        children = self.findChildren(entry)
        if len(children) == 0:
            return []
        return [ASTNode(child,entry) for child in children]
