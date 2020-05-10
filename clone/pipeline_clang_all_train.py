import pandas as pd
import os
import sys
import warnings
import clang.cindex
import random
import copy
import chardet
warnings.filterwarnings('ignore')

clang.cindex.Config.set_library_file("/home/k1462425/llvm/llvm-project/build/lib/libclang.so")

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

        def recursive_produce_nodes_string(cursor,nodes_string,depth,parent_identifier):
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

        path = self.root+self.language+'/'+output_file
        if os.path.exists(path) and option == 'existing':
            source = pd.read_pickle(path)
        else:
            if self.language is 'c':
                source = pd.read_pickle(self.root+self.language+'/programs.pkl')
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


    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
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
            from prepare_data_clang import get_sequences as func
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


        trees['code'] = trees['code'].apply(preprocess_trees)
        corpus = trees['code']

        #str_corpus = [' '.join(c) for c in corpus]
        #print(corpus)

        #trees['code'] = pd.Series(str_corpus)

        # trees.to_csv(data_path+'train/programs_ns.tsv')
        #print(trees['code'])
        #print(trees['code'][9566])

        print(corpus[9566])
        print(len(corpus[9566]))

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        if self.language is 'c':
            from prepare_data_clang import get_blocks as func
        else:
            from utils import get_blocks_v1 as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+self.language+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        #print(len(vocab))
        #for word in vocab:
        #    print(word)
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
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = pd.DataFrame(self.sources, copy=True)
        #print("generate_block_seqs() before applying anything: --------------------------")
        #print(trees)
        #print(trees['code'][9566])
        #print(trees['code'][0])
        trees['code'] = trees['code'].apply(trans2seq)
        result = trees['code'][9566]

        #print("after applying transformation: ------------------------------------")
        #print(trees['code'])
        #print(result)

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
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')


import argparse
parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
parser.add_argument('--lang')
args = parser.parse_args()
if not args.lang:
    print("No specified dataset")
    exit(1)
ppl = Pipeline('3:1:1', 'data/', str(args.lang))
ppl.run()
