import pandas as pd
import os
import clang.cindex
import random
import argparse
import pickle

clang.cindex.Config.set_library_file("/home/k1462425/llvm/llvm-project/build/lib/libclang.so")

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
    def parse_source_cparser(self, output_file, option):
        path = self.root+output_file
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_pickle(self.root+self.source_code_input)

            source.columns = ['id', 'code', 'label']
            source['code'] = source['code'].apply(parser.parse)

            source.to_pickle(path)
        self.sources = source
        return source

    # parse source code
    def parse_source(self, output_file, option):

        def recursive_produce_nodes_string(cursor,nodes_string,depth,parent_identifier):
            if cursor.kind.name == "UNEXPOSED_EXPR" or cursor.kind.name == "DECL_REF_EXPR":
                return nodes_string
            addition_string = str(depth)+","+cursor.kind.name+","+cursor.spelling
            tokens = []
            for token in cursor.get_tokens():
                tokens.append(token)

            if len(tokens) == 1:
                addition_string = addition_string + token.spelling + ","

            ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            chars=[]
            for i in range(16):
                chars.append(random.choice(ALPHABET))

            salt = "".join(chars)

            addition_string = salt+","+addition_string+","+parent_identifier+"¬"

            nodes_string = nodes_string + addition_string
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
                filename = "program.c"
                with open(filename,'w+') as fp:
                    fp.write(source['code'][x])
                    fp.close()

                index = clang.cindex.Index.create()
                tu = index.parse(filename)
                all_nodes_string = recursive_produce_nodes_string(tu.cursor,"",0,"NONE")

                source['code'][x] = all_nodes_string
                if(x % 100 == 0):
                    print(x)
                x = x + 1

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
    def dictionary_and_embedding_cparser(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from prepare_data import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from prepare_data_clang import get_sequences

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
            get_sequences(ast, sequence)
            return sequence

        corpus = trees['code'].apply(preprocess_trees)
        corpus = trees['code']
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
        from prepare_data_clang import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks, r)
            tree = []
            #for b in blocks:
            if(len(blocks) > 0):
                btree = tree_to_index(blocks[0])
                tree.append(btree)
                return tree
            return []

        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root+part+'/blocks.pkl')

    # generate block sequences with index representations
    def generate_block_seqs_cparser(self,data_path,part):
        from prepare_data_clang import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
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
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')

parser = argparse.ArgumentParser(description="Specify the source code pickle file.")
parser.add_argument('--source')
args = parser.parse_args()
if not args.source:
    print("No specified source code pickle file.")
    exit(1)
ppl = Pipeline('3:1:1', 'data/', args.source)
ppl.run()
