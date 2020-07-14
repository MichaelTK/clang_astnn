import pandas as pd
import os
import clang.cindex
import random
import argparse
import pickle
import copy

clang.cindex.Config.set_library_file("/home/k1462425/llvm/llvm-project/build/lib/libclang.so")

transIndex = 0

class Pipeline:
    def __init__(self,  ratio, root, source_path):
        self.source_code_input = source_path
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None


    # parse source code
    def parse_source(self, output_file, option):

        # Function to convert a char list to a string
        def listToString(s):
            str1 = ""
            for ele in s:
                str1 += ele
            return str1

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

        def find_parent(node, all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            node_split = node.split(',')
            parent_identifier = node_split[-1]
            parent = ""
            for nod in all_nodes_split:
                nod_split = nod.split(',')
                if nod_split[0] == parent_identifier:
                    parent = nod
            return parent

        def engineer_var_decls(all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for node in all_nodes_split:
                node_split = node.split(',')
                if len(node_split) > 1:
                    if node_split[2] == 'VAR_DECL':
                        parent = find_parent(node,all_nodes_string)
                        parent_split = parent.split(',')
                        depth = int(parent_split[1]) + 1
                        child_chars = []
                        for i in range(16):
                            child_chars.append(random.choice(ALPHABET))
                        all_nodes_string = all_nodes_string + listToString(child_chars) + "," + str(depth) + "," + "VAR_TYPE" + "," + str(parent_split[3]) + "," + str(node_split[0]) + "¬"
            for node in all_nodes_split:
                node_split = node.split(',')
                if len(node_split) > 1:
                    if node_split[2] == 'VAR_DECL':
                        parent = find_parent(node,all_nodes_string)
                        parent_split = parent.split(',')
                        if parent_split[2] == 'DECL_STMT':
                            var_decl_children = find_var_decl_children(parent,all_nodes_string)
                            for var_decl_child in var_decl_children:
                                all_nodes_string = create_decl_stmt_parent(var_decl_child,all_nodes_string)
            for node in all_nodes_split:
                nodesplit = node.split(',')
                children = findChildren(node,all_nodes_string)
                if len(children) == 0 and nodesplit[2] == 'DECL_STMT':
                    all_nodes_string = all_nodes_string.replace(node,'')
            return all_nodes_string


        def find_var_decl_children(node, all_nodes_string):
            var_decl_children = []
            children = findChildren(node,all_nodes_string)
            for child in children:
                childsplit = child.split(',')
                if childsplit[2] == 'VAR_DECL':
                    var_decl_children.append(child)

            return var_decl_children

        def create_decl_stmt_parent(node,all_nodes_string):
            parent_of_obsolete_parent = find_parent(find_parent(node,all_nodes_string),all_nodes_string)
            identifier_of_parent_of_obsolete_parent = parent_of_obsolete_parent.split(',')[0]
            ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            child_chars = []
            for i in range(16):
                child_chars.append(random.choice(ALPHABET))
            depth = 1
            new_decl_statement = listToString(child_chars)+","+str(depth)+","+"DECL_STMT"+","+","+identifier_of_parent_of_obsolete_parent+"¬"
            all_nodes_string = all_nodes_string+new_decl_statement
            nodesplit = node.split(',')
            nodesplit[-1] = listToString(child_chars)

            y = 0
            while y < len(nodesplit):
                if y != len(nodesplit) - 1:
                    nodesplit[y] = nodesplit[y] + ","
                y = y + 1
            this_node_with_updated_parent = listToString(nodesplit)

            all_nodes_string = all_nodes_string.replace(node,this_node_with_updated_parent)
            return all_nodes_string

        def engineer_unary_operators(all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            unary_operators = ['+','-','&','!','~','*','++','--']
            for node in all_nodes_split:
                node_split = node.split(',')
                old_node = node
                if len(node_split) > 1:
                    #print("Length bigger than 1")
                    if node_split[2] == 'UNARY_OPERATOR':
                        print("Unary operator found")
                        print(node_split[3])
                        print(node_split[4])
                        if node_split[4] in unary_operators:

                            node_split[3] = node_split[4]
                            y = 0
                            while y < len(node_split):
                                if y != len(node_split) - 1:
                                    node_split[y] = node_split[y] + ","
                                y = y + 1
                            subs_node = listToString(node_split)
                            print("Replacing " +old_node)
                            print("With "+subs_node)
                            all_nodes_string = all_nodes_string.replace(old_node,subs_node)

            return all_nodes_string

        def engineer_expr_list(all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            parents_done = []
            callexprs_exprlist_dict = {}
            ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for node in all_nodes_split:
                node_split = node.split(',')
                if len(node_split) > 1:
                    if node_split[2] == 'UNEXPOSED_EXPR' or node_split[2] == 'UNARY_OPERATOR':
                        parent = find_parent(node, all_nodes_string)
                        if len(parent) > 1:
                            parent_split = parent.split(',')
                            depth = int(parent_split[1]) + 1
                            if parent_split[2] == 'CALL_EXPR':
                                #print("Node: "+node)
                                #print("Parent: "+parent)
                                child_chars = []
                                if parent not in parents_done:
                                    for i in range(16):
                                        child_chars.append(random.choice(ALPHABET))
                                    new_node = listToString(child_chars) + "," + str(depth) + "," + "EXPR_LIST" + ",," + str(parent_split[0]) + "¬"
                                    old_node = node
                                    subs_node = node_split
                                    subs_node[-1] = listToString(child_chars)
                                    y = 0
                                    while y < len(subs_node):
                                        if y != len(subs_node) - 1:
                                            subs_node[y] = subs_node[y] + ","
                                        y = y + 1
                                    subs_node = listToString(subs_node)
                                    all_nodes_string = all_nodes_string.replace(old_node,subs_node)

                                    all_nodes_string = all_nodes_string + new_node
                                    parents_done.append(parent)
                                    callexprs_exprlist_dict[parent] = new_node
                                else:
                                    old_node = node
                                    new_node = callexprs_exprlist_dict.get(parent)
                                    new_node_split = new_node.split(',')
                                    subs_node = node_split
                                    subs_node[-1] = new_node_split[0]
                                    y = 0
                                    while y < len(subs_node):
                                        if y != len(subs_node) - 1:
                                            subs_node[y] = subs_node[y] + ","
                                        y = y + 1
                                    subs_node = listToString(subs_node)

                                    all_nodes_string = all_nodes_string.replace(old_node,subs_node)
            return all_nodes_string


        def remove_useless_unexposed(all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            for node in all_nodes_split:
                if 'UNEXPOSED_EXPR'  in node:
                    parent = find_parent(node, all_nodes_string)
                    if 'UNEXPOSED_EXPR' in parent:
                        all_nodes_string = all_nodes_string.replace(node, '')

                if 'DECL_REF_EXPR'  in node:
                    parent = find_parent(node, all_nodes_string)
                    if 'UNEXPOSED_EXPR' in parent:
                        nodesplit = node.split(',')
                        parentsplit = parent.split(',')
                        if nodesplit[3] == parentsplit[3]:
                            all_nodes_string = all_nodes_string.replace(node, '')
            return all_nodes_string


        def remove_top_level_compound(all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            for node in all_nodes_split:
                nodesplit = node.split(',')
                if len(nodesplit) > 1:
                    if nodesplit[2] == 'COMPOUND_STMT':
                        parent = find_parent(node,all_nodes_string)
                        parentsplit = parent.split(',')
                        if len(parentsplit) > 1:
                            if parentsplit[2] == 'FUNCTION_DECL':
                                children = findChildren(node,all_nodes_string)
                                for child in children:
                                    oldchild = child
                                    childsplit = child.split(',')
                                    childsplit[-1] = parentsplit[0]
                                    y = 0
                                    while y < len(childsplit):
                                        if y != len(childsplit) - 1:
                                            childsplit[y] = childsplit[y] + ","
                                        y = y + 1
                                    all_nodes_string = all_nodes_string.replace(oldchild,listToString(childsplit))
                                all_nodes_string = all_nodes_string.replace(node,'')
            return all_nodes_string


        def recursive_produce_nodes_string(cursor,nodes_string,depth,parent_identifier):
            if cursor.spelling != '':
                addition_string = str(depth)+","+cursor.kind.name+","+cursor.spelling+","
            else:
                addition_string = str(depth)+","+cursor.kind.name+","
            tokens = []
            for token in cursor.get_tokens():
                tokens.append(token)

            if cursor.kind.name == 'BINARY_OPERATOR':
                if len(tokens) > 1:
                    addition_string = addition_string + tokens[1].spelling + ","

            elif cursor.kind.name == 'UNARY_OPERATOR':
                unary_operators = ['+','-','&','!','~','*','++','--']
                if len(tokens) > 1:
                    if tokens[0].spelling in unary_operators:
                        #print(tokens[0].spelling)
                        addition_string = addition_string + tokens[0].spelling + ","
                    elif tokens[1].spelling in unary_operators:
                        #print(tokens[1].spelling)
                        addition_string = addition_string + tokens[1].spelling + ","

            elif len(tokens) > 0:
                addition_string = addition_string + tokens[0].spelling + ","

            ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            chars=[]
            for i in range(16):
                chars.append(random.choice(ALPHABET))

            salt = "".join(chars)

            addition_string = salt+","+addition_string+","+parent_identifier+"¬"
            child_string = ""

            addition_split = addition_string.split(',')

            if cursor.kind.name == 'CALL_EXPR':
                child_chars=[]
                for i in range(16):
                    child_chars.append(random.choice(ALPHABET))
                child_string = listToString(child_chars) + "," + str(depth+1) + "," + "CALL_NAME" + "," + addition_split[3] + "," + addition_split[0] + "¬"

            child_string2 = ""
            if cursor.kind.name == 'INTEGER_LITERAL':
                child_chars = []
                for i in range(16):
                    child_chars.append(random.choice(ALPHABET))

                child_string2 = listToString(child_chars) + "," + str(depth+1) + "," + "INTEGER_VALUE" + "," + addition_split[3] + "," + addition_split[0] + "¬"

            child_string3 = ""
            child_string4 = ""
            if cursor.kind.name == 'FUNCTION_DECL':
                child_chars = []
                for i in range(16):
                    child_chars.append(random.choice(ALPHABET))
                child_string3 = listToString(child_chars) + "," + str(depth+1) + "," + "FUNCTION_NAME" + "," + addition_split[3] + "," + addition_split[0] + "¬"
                parent_chars = child_chars
                child_chars = []
                for i in range(16):
                    child_chars.append(random.choice(ALPHABET))
                child_string4 = listToString(child_chars) + "," + str(depth+2) + "," + "FUNCTION_TYPE" + "," + addition_split[4] + "," + str(parent_chars) + "¬"


            nodes_string = nodes_string + addition_string + child_string + child_string2 + child_string3
            children = cursor.get_children()
            for child in children:
                nodes_string = recursive_produce_nodes_string(child,nodes_string,depth+1,salt)
            return nodes_string

        path = self.root+output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
            source = pd.read_pickle(self.root+self.source_code_input)
            source.columns = ['id', 'code', 'label']

            x = 0
            while x < len(source['code']):
                #if x == 8200:
                filename = "program.c"
                with open(filename,'w+') as fp:
                    fp.write(source['code'][x])
                    fp.close()

                index = clang.cindex.Index.create()
                tu = index.parse(filename)
                all_nodes_string = recursive_produce_nodes_string(tu.cursor,"",0,"NONE")
                all_nodes_string = engineer_var_decls(all_nodes_string)
                all_nodes_string = remove_useless_unexposed(all_nodes_string)
                all_nodes_string = engineer_expr_list(all_nodes_string)
                #all_nodes_string = engineer_unary_operators(all_nodes_string)
                all_nodes_string = remove_top_level_compound(all_nodes_string)
                source['code'][x] = all_nodes_string
                if(x % 100 == 0):
                    print(x)
                x = x + 1
            #sys.exit(0)


            #       x = x + 1

            source.to_pickle(path)
        self.sources = source

        return source


    # split data for training, developing and testing
    def split_data_cparser(self):
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = self.root+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = self.root+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = self.root+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root+'/'
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = data_path+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = data_path+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = data_path+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)


    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):

        # Function to convert a char list to a string
        def listToString(s):
            str1 = ""
            for ele in s:
                str1 += ele
            return str1

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

        def find_parent(node, all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            node_split = node.split(',')
            parent_identifier = node_split[-1]
            parent = ""
            for nod in all_nodes_split:
                nod_split = nod.split(',')
                if nod_split[0] == parent_identifier:
                    parent = nod
            return parent

        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from prepare_data_clang_rewritten import get_sequences

        def preprocess_trees(code):
            trees1 = []

            code_split = code.split('¬')
            code = []
            for elem in code_split:
                node_split = elem.split(',')
                new_node = node_split[2:-1] #gets rid of node identifier and node depth and parent identifier
                new_node_string = ''
                for info in new_node:
                    if info != '' and '.' not in info:
                        code.append(info)

            #print(code)

            return code

        def preprocess_trees_bigrams(code):
            bigrams = []
            code_split = code.split('¬')

            for elem in code_split:
                elem_truncated = elem.split(',')[2:-1]
                if not elem_truncated in nodesDone:
                    bigram = get_bigram(elem,code)
                    if len(bigram) > 1:
                        nodesDone.append(bigram[0])
                        nodesDone.append(bigram[1])
                        bigram[0] = tuple(bigram[0])
                        bigram[1] = tuple(bigram[1])
                        bigram = tuple(bigram)
                        bigram = hash(bigram)
                        bigrams.append(str(bigram))

            return bigrams

        def get_token(node):
            if 'End' in node.split(',')[0]:
                return 'End'
            else:
                if 'VAR_DECL' in node.split(',')[2]:
                    #print(self.node.split(',')[3])
                    return node.split(',')[3] #instead of putting VAR_DECL, put the name of the variable declared

                if 'CALL_NAME' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'FUNCTION_NAME' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'FUNCTION_TYPE' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'VAR_TYPE' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'INTEGER_VALUE' in node.split(',')[2] and node.split(',')[-2] is not '':
                    #print("integer_value")
                    return node.split(',')[3]

                if 'DECL_REF_EXPR' in node.split(',')[2]:
                    return node.split(',')[3]

                if 'UNARY_OPERATOR' in node.split(',')[2]:
                    #print(self.node.split(',')[2])
                    return node.split(',')[3]

                if 'BINARY_OPERATOR' in node.split(',')[2]:
                    return node.split(',')[3]

                if 'UNEXPOSED_EXPR' in node.split(',')[2]:
                    return node.split(',')[3]

            return node.split(',')[2]

        def get_bigram(node,entry):
            children = findChildren(node,entry)
            bigram = []
            nodeSplit = node.split(',')
            newNode = nodeSplit[2:-1]
            new_node_string = ''
            code = []
            if len(newNode) > 0:
                if 'VAR_DECL' in newNode or 'CALL_NAME' in newNode or 'FUNCTION_TYPE' in newNode or 'VAR_TYPE' in newNode or 'INTEGER_VALUE' in newNode or 'DECL_REF_EXPR' in newNode or 'UNARY_OPERATOR' in newNode or 'BINARY_OPERATOR' in newNode or 'UNEXPOSED_EXPR' in newNode:
                    #print(newNode)
                    if 'INTEGER_LITERAL' in newNode and len(INTEGER_LITERAL) < 2:
                        code.append(newNode[0])

                    else:
                        code.append(newNode[1])
                else:
                    #print(newNode)
                    code.append(newNode[0])
            #for info in newNode:
            #    if info != '' and '.' not in info:
            #        code.append(info)

            bigram.append(code)
            code = []
            if len(children) != 0:
                nodeSplit = children[0].split(',')
                newNode = nodeSplit[2:-1]
                new_node_string = ''
                if len(newNode) > 0:
                    if 'VAR_DECL' in newNode or 'CALL_NAME' in newNode or 'FUNCTION_TYPE' in newNode or 'VAR_TYPE' in newNode or 'INTEGER_VALUE' in newNode or 'DECL_REF_EXPR' in newNode or 'UNARY_OPERATOR' in newNode or 'BINARY_OPERATOR' in newNode or 'UNEXPOSED_EXPR' in newNode:
                        if 'INTEGER_LITERAL' in newNode and len(INTEGER_LITERAL) < 2:
                            code.append(newNode[0])

                        else:
                            code.append(newNode[1])
                    else:
                        code.append(newNode[0])
                #for info in newNode:
                #    if info != '' and '.' not in info:
                #        code.append(info)
                bigram.append(code)

            return bigram

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence

        nodesDone = []

        trees_copy = copy.deepcopy(trees['code'])
        #trees_copy.apply(preprocess_trees_bigrams)
        corpus = trees_copy.apply(preprocess_trees_bigrams)

        print("CORPUS:")
        print(corpus)
        print(corpus[0])

        orig_corpus = trees['code'].apply(preprocess_trees)

        print("ORIG CORPUS:")
        print(orig_corpus)
        print(orig_corpus[0])

        #corpus = trees['code'].apply(preprocess_trees)
        #corpus = trees['code']
        #str_corpus = [' '.join(c) for c in corpus]
        #trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        print(trees['code'])
        print(corpus)

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self,data_path,part):
        from prepare_data_clang_rewritten import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]
        nodesDone = []

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

        def get_bigram(node,entry):
            children = findChildren(node,entry)
            bigram = []
            nodeSplit = node.split(',')
            newNode = nodeSplit[2:-1]
            new_node_string = ''
            code = []
            if len(newNode) > 0:
                if 'VAR_DECL' in newNode or 'CALL_NAME' in newNode or 'FUNCTION_TYPE' in newNode or 'VAR_TYPE' in newNode or 'INTEGER_VALUE' in newNode or 'DECL_REF_EXPR' in newNode or 'UNARY_OPERATOR' in newNode or 'BINARY_OPERATOR' in newNode or 'UNEXPOSED_EXPR' in newNode:
                    if 'INTEGER_LITERAL' in newNode and len(INTEGER_LITERAL) < 2:
                        code.append(newNode[0])

                    else:
                        code.append(newNode[1])
                else:
                    code.append(newNode[0])

            #for info in newNode:
            #    if info != '' and '.' not in info:
            #        code.append(info)

            bigram.append(code)
            code = []
            if len(children) != 0:
                nodeSplit = children[0].split(',')
                newNode = nodeSplit[2:-1]
                new_node_string = ''
                if len(newNode) > 0:
                    if 'VAR_DECL' in newNode or 'CALL_NAME' in newNode or 'FUNCTION_TYPE' in newNode or 'VAR_TYPE' in newNode or 'INTEGER_VALUE' in newNode or 'DECL_REF_EXPR' in newNode or 'UNARY_OPERATOR' in newNode or 'BINARY_OPERATOR' in newNode or 'UNEXPOSED_EXPR' in newNode:
                        if 'INTEGER_LITERAL' in newNode and len(INTEGER_LITERAL) < 2:
                            code.append(newNode[0])

                        else:
                            code.append(newNode[1])
                    else:
                        code.append(newNode[0])
                #for info in newNode:
                #    if info != '' and '.' not in info:
                #        code.append(info)
                bigram.append(code)

            return bigram


        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def preprocess_trees_bigrams(code):
            bigrams = []
            code_split = code.split('¬')

            for elem in code_split:
                elem_truncated = elem.split(',')[2:-1]
                if not elem_truncated in nodesDone:
                    bigram = get_bigram(elem,code)
                    if len(bigram) > 1:
                        nodesDone.append(bigram[0])
                        nodesDone.append(bigram[1])
                        bigram[0] = tuple(bigram[0])
                        bigram[1] = tuple(bigram[1])
                        bigram = tuple(bigram)
                        bigram = hash(bigram)
                        tokenNumber = vocab[str(bigram)].index if str(bigram) in vocab else max_token
                        bigrams.append(tokenNumber)

            return bigrams

        def get_token(node):
            if 'End' in node.split(',')[0]:
                return 'End'
            else:
                if 'VAR_DECL' in node.split(',')[2]:
                    #print(self.node.split(',')[3])
                    return node.split(',')[3] #instead of putting VAR_DECL, put the name of the variable declared

                if 'CALL_NAME' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'FUNCTION_NAME' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'FUNCTION_TYPE' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'VAR_TYPE' in node.split(',')[2]:
                    #print("call_name")
                    return node.split(',')[3]

                if 'INTEGER_VALUE' in node.split(',')[2] and node.split(',')[-2] is not '':
                    #print("integer_value")
                    return node.split(',')[3]

                if 'DECL_REF_EXPR' in node.split(',')[2]:
                    return node.split(',')[3]

                if 'UNARY_OPERATOR' in node.split(',')[2]:
                    #print(self.node.split(',')[2])
                    return node.split(',')[3]

                if 'BINARY_OPERATOR' in node.split(',')[2]:
                    return node.split(',')[3]

                if 'UNEXPOSED_EXPR' in node.split(',')[2]:
                    return node.split(',')[3]

            return node.split(',')[2]

        def trans2seq(r):
            global transIndex
            blocks = []
            func(r, blocks, r)
            tree = []
            if(len(blocks) > 0):
                btree = tree_to_index(blocks[0])
                tree.append(btree)
                transIndex = transIndex + 1
                print("Generated " +str(transIndex)+ " sequences.")
                return tree

            transIndex = transIndex + 1
            print("Generated " +str(transIndex)+ " sequences.")
            if transIndex % 1000 == 0:
                print("Generated " +str(transIndex) +" sequences.")
            return []

        print("VOCAB")
        print(vocab)

        trees = pd.read_pickle(data_path)
        print("BEFORE:")
        print(trees['code'])
        trees['code'] = trees['code'].apply(preprocess_trees_bigrams)
        print("AFTER:")
        print(trees['code'])

        bigram_hashes_and_indices = {}
        for elem in vocab:
            bigram_hashes_and_indices[elem] = vocab[elem].index

        with open('./data/bigram_hashes_and_indices.pkl', 'wb') as handle:
            pickle.dump(bigram_hashes_and_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

        trees.to_pickle(self.root+part+'/blocks.pkl')


    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        #self.generate_block_seqs(self.dev_file_path, 'dev')
        #self.generate_block_seqs(self.test_file_path, 'test')

parser = argparse.ArgumentParser(description="Specify the source code pickle file.")
parser.add_argument('--source')
args = parser.parse_args()
if not args.source:
    print("No specified source code pickle file.")
    exit(1)
ppl = Pipeline('1:0:0', 'data/', args.source)
ppl.run()
