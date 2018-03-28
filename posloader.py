
from collections import Counter
from ted.ted import TED
import numpy as np
import nltk


class PosLoader():

    def __init__(self, batch_size=30, seq_len=30, data_dir="."):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dir = data_dir

        self.process()
        self.batchify()
        self.reset_batch_pointer()


    def process(self):

        tags  = TED().get_tags()
        self.vocab_list = [word for word, freq in Counter(tags).items()]
        self.vocab_size = len(self.vocab_list)
        self.vocab = dict(zip(self.vocab_list, range(len(self.vocab_list))))
        self.data = np.array([self.vocab[t] for t in tags])


    def batchify(self):

        self.num_batches = int(len(self.data) / (self.batch_size * self.seq_len))
        self.data = self.data[:self.num_batches * self.batch_size * self.seq_len]

        x, y = self.data, np.copy(self.data)
        y[:-1] = x[1:]
        y[-1] = x[0]

        self.x_batches = np.split(x.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(y.reshape(self.batch_size, -1), self.num_batches, 1)


    def next(self):

        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        if self.pointer >= len(self.x_batches):
            self.reset_batch_pointer()

        return x, y


    def rand(self):
        pointer = np.random.randint(len(self.x_batches))
        return self.x_batches[pointer], self.y_batches[pointer]


    def reset_batch_pointer(self):
        self.pointer = 0


if __name__=="__main__":
    l = PosLoader()