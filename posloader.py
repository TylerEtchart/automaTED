from ted.ted import TED
import numpy as np
import nltk


class PosLoader():

    def __init__(self, batch_size=30, seq_len=30, data_dir="."):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dir = data_dir

        # self.v_list = self.process()                                # [ word1, word2, word3 ... ]
        # self.v_size = len(self.vocab)                               # word3 --> [ 0, 0, 1, 0, 0 ... ]
        # self.v_nums = dict(zip(self.v_list, range(self.v_size)))    # { word1: 1, word2: 2, word3: 3 ...}

        self.process()
        self.batchify()


    def process(self):



        tags  = TED().get_tags()
        vocab = set(tags)
        vocab = dict(zip(vocab, range(len(vocab))))
        
        self.data = np.array([vocab[t] for t in tags])


    def batchify(self):

        num_batches = int(len(self.data) / (self.batch_size * self.seq_len))
        self.data = self.data[:num_batches * self.batch_size * self.seq_len]

        x, y = self.data, np.copy(self.data)
        y[:-1] = x[1:]
        y[-1] = x[0]

        self.x_batches = np.split(x.reshape(self.batch_size, -1), num_batches, 1)
        self.y_batches = np.split(y.reshape(self.batch_size, -1), num_batches, 1)

        print(type(self.x_batches))


    def next(self):

        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        if self.pointer >= len(self.x_batches):
            self.reset_batch_pointer()

        return x, y


    def reset_batch_pointer(self):
        self.pointer = 0


if __name__=="__main__":
    l = BatchLoader()