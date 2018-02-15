import codecs
import os
import re
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, scholar, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        tensor_file = os.path.join(data_dir, "data.npy")
        self.s = scholar

        if not os.path.exists(tensor_file):
            print("reading text file")
            self.preprocess(input_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        data = self.strip_data(data)
        self.turn_data_into_word_vectors(data)
        np.save(tensor_file, self.tensor)

    def strip_data(self, data):
        pattern = re.compile("[^a-zA-Z]")
        data = data.split(" ")
        data = [pattern.sub("", x) for x in data]
        data = [x.lower() for x in data if x != ""]
        return data

    def turn_data_into_word_vectors(self, data):
        tensor_data = []
        for x in data:
            try:
                x = x.encode("utf-8")
                tag = "_" + self.s.get_most_common_tag(x)
                x += tag
                vec = self.s.get_vector(x)
                tensor_data.append(list(vec))
            except:
                pass
        self.tensor = np.array(tensor_data)


    def load_preprocessed(self, tensor_file):
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.shape[0] / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        self.x_batches = np.array_split(xdata, self.num_batches)
        self.x_batches = [x.reshape(50, 50, 100) for x in self.x_batches]
        self.y_batches = np.array_split(ydata, self.num_batches)
        self.y_batches = [y.reshape(50, 50, 100) for y in self.y_batches]

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