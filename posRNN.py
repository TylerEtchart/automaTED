import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.python.ops.rnn_cell import GRUCell, MultiRNNCell


class PosRNN():

	def __init__(self):

		self.batch_size = 50
		self.seq_length = 50
		self.num_layers = 2
		self.num_hidden = 128
		self.vocab_size = 35

		tf.reset_default_graph()
		self.createGraph()

		self.sess = tf.Session()
		self.path = "./pos_tf_logs"
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		if restore:
			self.saver.restore(self.sess, self.path + "/model.ckpt")


	def createGraph(self):

		self.input = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name='inputs')
		self.targs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name='targets')
		self.onehot = tf.one_hot(self.input, self.vocab_size, name='input_oh')

		inputs = tf.split(self.onehot, self.seq_length, 1)
		inputs = [tf.squeeze(i, [1]) for i in inputs]
		targets = tf.split(self.targs, self.seq_length, 1)
		
		with tf.variable_scope("posRNN"):

			cells = [GRUCell(self.num_hidden) for _ in range(self.num_layers)]

			stacked = MultiRNNCell(cells, state_is_tuple=True)
			self.zero_state = stacked_cells.zero_state(self.batch_size, tf.float32)

			output, self.last_state = seq2seq.rnn_decoder()

			w = tf.get_variable("w", [self.num_hidden, self.vocab_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
			b = tf.get_variable("b", [self.vocab_size], tf.constant_initializer(0.0))
			logits = [tf.matmul(o, w) + b for o in outputs]

			const_weights = [tf.ones([self.batch_size]) for _ in xrange(self.seq_length)]
			self.loss = seq2seq.sequence_length(logits, targets, const_weights)

			self.opt = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(loss)

		with tf.variable_scope("posRNN", reuse=True):

			batch_size = 1
			self.s_inputs = tf.placeholder(tf.int32, [batch_size], name='s_inputs')
			s_onehot = tf.one_hot(s_inputs, self.vocab_size, name='s_input_oh')

			self.s_zero_state = stacked_cells.zero_state(batch_size, tf.float32)
			s_output, self.s_last_state = seq2seq.rnn_decoder([s_onehot], self.s_zero_state, stacked)
			s_output = tf.reshape(s_output, [1, self.num_hidden])
			self.s_probs = tf.nn.softmax(tf.matmul(s_output, w) + b)


	def sample(self, num, prime):

		s_state = self.sess.run(self.s_zero_state)

		for word in prime.split(" "):
			# TODO : self.data_loader.vocab[word]
			x = np.ravel().astype('int32')
			d = {self.s_inputs:x}
			for i, s in enumerate(self.s_zero_state):
				d[s] = s_state[i]
			s_state = self.sess.run(self.s_last_state, feed_dict=d)

		total_probs = []
		for n in range(num):
			# TODO : self.data_loader.vocab[word]
			x = np.ravel().astype('int32')
			d = {self.s_inputs:x}
			for i, s in enumerate(self.s_zero_state):
				d[s] = s_state[i]
			ops = [self.s_probs]
			ops.extend(list(self.s_last_state))

			retval = self.sess.run(ops, feed_dict=d)

			s_probsv = retval[0]
			s_state = retval[1:]

			total_probs.append(s_probsv[0])

		return total_probs


	def train(self):

		lts = []

		for j in range(1000):

			state = self.sess.run(self.zero_state)
			# TODO : self.data_loader.reset_batch_pointer()

			for i in range(self.data_loader.num_batches):

				x, y = self.data_loader.next_batch()

				d = {self.input:x, self.targs:y}
				for k, s, in enumerate(self.zero_state):
					d[s] = state[k]

				ops = [self.opt, self.loss]
				ops.extend(list(self.last_state))

				retval = self.sess.run([ops], feed_dict=d)

				lt = retval[1]
				state = retval[2:]

				if i%100==0:
					print("%d %d\t %.4f" % (j, i, lt))
					lts.append(lt)

			print("Completed epoch: {}, saving weights...".format(j))
			self.saver.save(self.sess, self.path + "/model.ckpt")

		plt.plot(range(len(lts)), lts)
		plt.show()

if __name__ == "__main__":

	posRNN = PosRNN(restore=False)
	posRNN.train()