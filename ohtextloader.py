import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
from ted import TED

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
        #     print("reading text file")
        #     self.preprocess(input_file, vocab_file, tensor_file)
        # else:
        #     print("loading preprocessed files")
        #     self.load_preprocessed(vocab_file, tensor_file)

        self.preprocess(input_file, vocab_file, tensor_file)

        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        # with codecs.open(input_file, "r", encoding=self.encoding) as f:
        #     data = f.read()

        ted = TED()

        # counter = collections.Counter(data)
        # count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        # self.chars, _ = zip(*count_pairs)
        # self.vocab_size = len(self.chars)
        # self.vocab = dict(zip(self.chars, range(len(self.chars))))

        self.vocab_size = ted.vocab_size
        self.vocab_list = ted.vocab_list
        self.vocab = dict(zip(ted.vocab_list, range(ted.vocab_size)))
        self.tensor = np.array([self.vocab[word] for word in ted.stripped_talks])

        # with open(vocab_file, 'wb') as f:
        #     cPickle.dump(self.chars, f)
        # self.tensor = np.array(list(map(self.vocab.get, data)))
        # np.save(tensor_file, self.tensor)

    # def load_preprocessed(self, vocab_file, tensor_file):
    #     with open(vocab_file, 'rb') as f:
    #         self.chars = cPickle.load(f)
    #     self.vocab_size = len(self.chars)
    #     self.vocab = dict(zip(self.chars, range(len(self.chars))))
    #     self.tensor = np.load(tensor_file)
    #     self.num_batches = int(self.tensor.size / (self.batch_size *
    #                                                self.seq_length))

    def create_batches(self):
        # self.num_batches = int(self.tensor.size / (self.batch_size *
        #                                            self.seq_length))

        self.num_batches = int(len(self.tensor) / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        if self.pointer >= len( self.x_batches ):
            self.reset_batch_pointer()

        return x, y

    def random_batch(self):
        pointer = np.random.randint( len( self.x_batches ) )
        return self.x_batches[pointer], self.y_batches[pointer]

    def reset_batch_pointer(self):
        self.pointer = 0
        

t = TextLoader( ".", 50, 50 )