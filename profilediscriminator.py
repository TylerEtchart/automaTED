import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from ohtextloader import TextLoader
from tensorflow.python.ops import rnn_cell
import tensorflow.contrib.legacy_seq2seq as seq2seq

class ProfileDiscriminator:

    def __init__(self, sample_batch_size, sample_sequence_length):
        # -------------------------------------------
        # Global variables

        self.batch_size = 50
        self.sequence_length = 50
        self.state_dim = 512
        self.profile_size = 14
        self.num_layers = 3
        self.data_loader = TextLoader(".", self.batch_size, self.sequence_length)
        self.vocab_size = self.data_loader.vocab_size  # dimension of one-hot encodings

        self.sample_state = None
        self.sample_batch_size = sample_batch_size
        self.sample_sequence_length = sample_sequence_length

        # tf.reset_default_graph()
        
        # with tf.variable_scope("wordRNN"):
        #     self.createGraph()
        self.createGraph()

        # self.sess = tf.Session()
        # self.sess.run(tf.global_variables_initializer())
        self.path = "./pd_tf_logs"
        self.summary_writer = tf.summary.FileWriter(self.path)
        profRNN_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='wordRNN/profRNN')
        self.saver = tf.train.Saver(profRNN_vars)


    def restore_weights(self, sess):
        self.saver.restore(sess, self.path + "/model.ckpt")


    def createGraph(self):
        # -------------------------------------------
        # Inputs

        self.in_ph = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name='inputs')
        self.target_profile = tf.placeholder(tf.float32, [self.batch_size, self.profile_size], name="target")

        in_onehot = tf.one_hot(self.in_ph, self.vocab_size, name="input_onehot")
        inputs = tf.split(in_onehot, self.sequence_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # -------------------------------------------
        # Computation Graph

        with tf.variable_scope("profRNN"):
            cells = [rnn_cell.GRUCell(self.state_dim) for i in range(self.num_layers)]
            # cells = [GORUCell( state_dim, str(i) ) for i in range(num_layers)]

            self.stacked_cells = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            self.initial_state = self.stacked_cells.zero_state(self.batch_size, tf.float32)
            # call seq2seq.rnn_decoder
            outputs, self.final_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.stacked_cells)
            # transform the list of state outputs to a list of logits.
            # use a linear transformation.
            self.W = tf.get_variable("W", [self.state_dim, self.profile_size], tf.float32,
                                        tf.random_normal_initializer(stddev=0.02))
            self.b = tf.get_variable("b", [self.profile_size],
                                        initializer=tf.constant_initializer(0.0))
            self.logits = tf.nn.softmax(tf.matmul(outputs[-1], self.W) + self.b)
            # call seq2seq.sequence_loss
            self.loss = tf.reduce_sum(tf.abs(self.target_profile - self.logits))
            self.loss_summary = tf.summary.scalar("loss", self.loss)
            # create a training op using the Adam optimizer
            self.optim = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(self.loss)

        
        # -------------------------------------------
        # Compute Graph

        # with tf.variable_scope("RNN", reuse=True):
        #     # inputs
        #     self.s_inputs = tf.placeholder(tf.int32,
        #             [self.sample_batch_size, self.sample_sequence_length], name='s_inputs')
        #     s_onehot = tf.one_hot(self.s_inputs, self.vocab_size, name="s_input_onehot")
        #     s_onehot = tf.split(s_onehot, self.sample_sequence_length, 1)
        #     s_onehot = [tf.squeeze(input_, [1]) for input_ in s_onehot]

        #     # initialize
        #     self.s_initial_state = self.stacked_cells.zero_state(self.sample_batch_size, tf.float32)

        #     # call seq2seq.rnn_decoder
        #     s_outputs, self.s_final_state = seq2seq.rnn_decoder(s_onehot,
        #                                         self.s_initial_state, self.stacked_cells)

        #     # transform the list of state outputs to a list of logits.
        #     # use a linear transformation.
        #     # s_outputs = tf.reshape(s_outputs, [1, self.state_dim])
        #     self.s_probs = tf.nn.softmax(tf.matmul(s_outputs[-1], self.W) + self.b)

    def compute_profile_from_within(self, x):
        with tf.variable_scope("profRNN", reuse=True):
            # inputs
            self.s_inputs = x
            s_onehot = tf.one_hot(self.s_inputs, self.vocab_size, name="s_input_onehot")
            s_onehot = tf.split(s_onehot, self.sample_sequence_length, 1)
            s_onehot = [tf.squeeze(input_, [1]) for input_ in s_onehot]

            # initialize
            self.s_initial_state = self.stacked_cells.zero_state(self.sample_batch_size, tf.float32)

            # call seq2seq.rnn_decoder
            s_outputs, self.s_final_state = seq2seq.rnn_decoder(s_onehot,
                                                self.s_initial_state, self.stacked_cells)

            # transform the list of state outputs to a list of logits.
            # use a linear transformation.
            # s_outputs = tf.reshape(s_outputs, [1, self.state_dim])
            self.s_probs = tf.nn.softmax(tf.matmul(s_outputs[-1], self.W) + self.b)
        return self.s_probs

        
    def compute_profile_from_without(self, x):
        if self.sample_state == None:
            self.sample_state = self.sess.run(self.s_initial_state)

        feed = {self.s_inputs: x}
        for k, s in enumerate(self.s_initial_state):
            feed[s] = self.sample_state[k]

        ops = [self.s_probs]
        ops.extend(list(self.s_final_state))

        retval = self.sess.run(ops, feed_dict=feed)
        self.sample_state = retval[1:]
        return retval[0]
    

    def train(self):
        lts = []

        print("FOUND %d BATCHES" % self.data_loader.num_batches)

        for j in range(1000):

            state = self.sess.run(self.initial_state)
            self.data_loader.reset_batch_pointer()

            for i in range(self.data_loader.num_batches):
                
                x, y, profile_vec, _ = self.data_loader.random_batch()

                # we have to feed in the individual states of the MultiRNN cell
                feed = {self.in_ph: x, self.target_profile: profile_vec}
                for k, s in enumerate(self.initial_state):
                    feed[s] = state[k]

                ops = [self.optim, self.loss]
                ops.extend(list(self.final_state))

                # retval will have at least 3 entries:
                # 0 is None (triggered by the optim op)
                # 1 is the loss
                # 2+ are the new final states of the MultiRNN cell
                retval, loss_summary = self.sess.run([ops, self.loss_summary], feed_dict=feed)
                self.summary_writer.add_summary(loss_summary, (j * self.data_loader.num_batches) + i)

                lt = retval[1]
                state = retval[2:]

                if i%5==0:
                    print("%d %d\t%.4f" % ( j, i, lt ))
                    lts.append( lt )

            print("Completed epoch: {}, saving weights...".format(j))
            self.saver.save(self.sess, self.path + "/model.ckpt")

        self.summary_writer.close()
        # plot the loss
        plt.plot(range(len(lts)), lts)
        plt.show()



if __name__ == "__main__":
    bs = 1
    sl = 10

    profile_discriminator = ProfileDiscriminator(sample_batch_size=bs, sample_sequence_length=sl)
    # profile_discriminator.train()
    
    profile_discriminator.restore_weights(profile_discriminator.sess)
    dl = TextLoader(".", bs, sl)
    print(dl.num_batches)

    for i in range(10000):
        x, y, profile, _ = dl.next_batch()

    print(x)
    print(profile)
    output = profile_discriminator.compute_profile_from_without(x)
    print(output)
    print(np.mean((output - profile)**2))
