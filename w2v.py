# -*- coding: utf-8 -*-

import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('w2v.model')
print(model['6302736951'])
print(model['B004NSUXHU'])
print(model.similarity('6302736951', '6305432260'))

exit()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname):
            if len(line) == 0 or line == '':
                continue
            yield line.split()


sentences = MySentences('data/Movies_and_TV_5/data_Movies_and_TV_5.txt')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences)
model.wv.save_word2vec_format('w2v.model')
print(model.similarity('6301977467', 'B00004RERE'))
