import pandas as pd
import os
import sys
import warnings
import clang.cindex
import copy
import random #debug
import pickle #debug
import chardet
warnings.filterwarnings('ignore')

#clang.cindex.Config.set_library_file("/usr/lib/llvm-6.0/lib/libclang.so")
clang.cindex.Config.set_library_file("/home/k1462425/llvm/llvm-project/build/lib/libclang.so")
transIndex = 0
globalNodes = []
missingTokens = []
indexO = 0
corpus = 0

class Pipeline:

    def __init__(self,  ratio, root, language):
        self.ratio = ratio
        self.root = root
        self.language = language
        self.sources = None
        self.blocks = None
        self.pairs = None
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


        def engineer_field_decls(all_nodes_string):
            all_nodes_split = all_nodes_string.split('¬')
            ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for node in all_nodes_split:
                nodesplit = node.split(',')
                if len(nodesplit) > 1:
                    if nodesplit[2] == 'FIELD_DECL':
                        new_child = ""
                        child_chars = []
                        for i in range(16):
                            child_chars.append(random.choice(ALPHABET))
                        node_name = 'VAR_TYPE'
                        token = nodesplit[3]
                        parent_identifier = nodesplit[0]
                        depth = int(nodesplit[1]) + 1
                        all_nodes_string = all_nodes_string + listToString(child_chars) + "," + str(depth) + "," + node_name + "," + token + "," + parent_identifier + "¬"

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

        path = self.root+self.language+'/'+output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
            if self.language is 'c':
                source = pd.read_pickle(self.root+self.language+'/programs.pkl')
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
                    all_nodes_string = engineer_field_decls(all_nodes_string)
                    source['code'][x] = all_nodes_string
                    if(x % 100 == 0):
                        print(x)
                    x = x + 1
                #sys.exit(0)


                #       x = x + 1

                source.to_pickle(path)
        self.sources = source

        return source


    # create clone pairs
    def read_pairs(self, filename):
        pairs = pd.read_pickle(self.root+self.language+'/'+filename)
        self.pairs = pairs

    # split data for training, developing and testing
    def split_data(self):
        data_path = self.root+self.language+'/'
        data = self.pairs
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
        global corpus
        self.size = size
        data_path = self.root+self.language+'/'
        if not input_file:
            input_file = self.train_file_path
        pairs = pd.read_pickle(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id',drop=False).loc[train_ids]
        if not os.path.exists(data_path+'train/embedding'):
            os.mkdir(data_path+'train/embedding')
        if self.language is 'c':
            sys.path.append('../')
            from prepare_data_clang_rewritten import get_sequences as func
        else:
            from utils import get_sequence as func


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

            return code


        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence, ast)
            return sequence

        #trees = trees[:1000]

        trees['code'] = trees['code'].apply(preprocess_trees)

        corpus = trees['code']

        from gensim.models.word2vec import Word2Vec
        print("Corpus:")
        print(corpus)
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data_clang_rewritten import get_blocks as func
        else:
            from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        def save_obj(obj, name ):
            with open('obj/'+ name + '.pkl', 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        def load_obj(name ):
            with open('obj/' + name + '.pkl', 'rb') as f:
                return pickle.load(f)

        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]
        print("Original vocab length: "+str(max_token))


        data_path = self.root+self.language+'/'
        debugfile = data_path+"vocabDict"

        dict = {}
        for elem in vocab:
            dict[vocab[elem].index] = elem

        save_obj(dict,"tokenDict")

        with open(debugfile,'a+') as fp:
            for elem in vocab:
                fp.write(str(vocab[elem].index)+","+elem+"\n")
        fp.close()


        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            global transIndex

            blocks = []
            func(r, blocks, r)
            tree = []

            btree = tree_to_index(blocks[0])
            tree.append(btree)

            transIndex = transIndex + 1
            if transIndex % 1000 == 0:
                print("Generated " +str(transIndex) +" sequences.")
            return tree

        def tree_to_index_debug(node):
            global globalNodes
            token = node.token

            if not token in globalNodes:
                globalNodes.append(token)
            #result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                tree_to_index_debug(child)

        def listToString(s):
            str1 = ""
            for ele in s:
                str1 += ele
            return str1

        def correctTrees(code):
            #print(code)
            #codesplit = code.split(',')
            new_entry_list = []
            for elem in code:
                if elem in globalNodes:
                    new_entry_list.append(elem)

            return new_entry_list


        def cleanVocab(code):
            global globalNodes, missingTokens, indexO
            vocabUsed = []
            #missingTokens = []
            for elem in code:
                blocks = []
                func(elem, blocks, elem)
                tree_to_index_debug(blocks[0])
                indexO = indexO + 1
                if indexO % 100 == 0:
                    print(indexO)
            #tree.append(btree)
            #for token in vocab:
            #    if not token in globalNodes:
            #        missingTokens.append(token)
            #        vocab.pop(token)

            print("Tokens found: -----------------------------------")
            tokensFound = []
            for i in list(vocab):
                if not i in globalNodes:
                    missingTokens.append(i)
                    vocab.pop(i)
                else:
                    #print(i)
                    tokensFound.append(i)
            #print("Number: "+str(len(tokensFound)))


        print("Old vocab length: ")
        print(len(vocab))
        global indexO,globalNodes, corpus
        trees = pd.DataFrame(self.sources, copy=True)
        #trees = trees[:1000]
        #copy_of_trees = copy.deepcopy(trees)
        #print("Vocab:")
        #print(vocab)

        print("Cleaning vocab...")
        print(trees['code'])
        #print(trees['code'])
        cleanVocab(trees['code'])

        corpus = corpus.apply(correctTrees)
        print(corpus)

        os.remove(data_path+'train/embedding/node_w2v_' + str(self.size))
        w2v = Word2Vec(corpus, size=self.size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(self.size))
        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab

        print("New vocab length: ")
        print(len(vocab))

        #word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        #vocab2 = word2vec.vocab
        #print("Elements of vocab2:")
        #for i in list(vocab2):
        #    print(vocab2.get(i))
        max_token = word2vec.syn0.shape[0]
        #print("Vocab2: ")
        #print(vocab2)
        #print("New vocab length: "+str(word2vec.syn0.shape[0]))
        #trees['code'].apply(cleanVocab)
        #print("Missing tokens: "+str(len(missingTokens)))
        #print(missingTokens)
        #print("Global nodes: ")
        #print(globalNodes)
        #print("Length of global nodes: "+str(len(globalNodes)))
        #max_token = word2vec.syn0.shape[0]
        print("Vocab cleaned.")
        print("generate_block_seqs() before applying anything: --------------------------")
        print(trees)
        trees['code'] = trees['code'].apply(trans2seq)
        #trees['code'][8200] = trans2seq(trees['code'][8200])

        #print(trees['code'][8200])

        #sys.exit(0)

        print("after applying transformation: ------------------------------------")
        print(trees['code'])

        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self,data_path,part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1,inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root+self.language+'/'+part+'/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast_clang.pkl',option='existing')
        print('read id pairs...')
        if self.language is 'c':
            self.read_pairs('oj_clone_ids.pkl')
        else:
            self.read_pairs('bcb_pair_ids.pkl')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')


import argparse
parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
parser.add_argument('--lang')
args = parser.parse_args()
if not args.lang:
    print("No specified dataset")
    exit(1)
ppl = Pipeline('3:1:1', 'data/', str(args.lang))
ppl.run()
