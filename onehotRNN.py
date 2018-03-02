import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from ohtextloader import TextLoader
from tensorflow.python.ops import rnn_cell
import tensorflow.contrib.legacy_seq2seq as seq2seq # I don't want to use legacy...
from goru.goru import GORUCell

#
# -------------------------------------------
#
# Global variables

batch_size = 10
sequence_length = 10

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 256

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )
profile = tf.placeholder( tf.float32, [ batch_size, 14 ], name="profile" )

inputs = tf.split( in_onehot, sequence_length, 1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, 1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

# create a BasicLSTMCell
#   use it to create a MultiRNNCell
#   use it to create an initial_state
#     note that initial_state will be a *list* of tensors!
with tf.variable_scope("RNN"):
    # cell = rnn_cell.BasicLSTMCell( state_dim )
    cells = [rnn_cell.GRUCell( state_dim ) for i in range(num_layers)]
    # cells = [GORUCell( state_dim, str(i) ) for i in range(num_layers)]

    stacked_cells = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    initial_state = stacked_cells.zero_state(batch_size, tf.float32)

    inputs = [tf.concat([i, profile], 1) for i in inputs]

    W_mix = tf.get_variable( "W_mix", [vocab_size + 14, vocab_size], tf.float32,
                                  tf.random_normal_initializer( stddev=0.02 ) )
    b_mix = tf.get_variable( "b_mix", [vocab_size],
                                 initializer=tf.constant_initializer( 0.0 ))
    mixed_inputs = [tf.nn.relu(tf.matmul( i, W_mix ) + b_mix) for i in inputs]

    # call seq2seq.rnn_decoder
    outputs, final_state = seq2seq.rnn_decoder(mixed_inputs, initial_state, stacked_cells)

    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    W = tf.get_variable( "W", [state_dim, vocab_size], tf.float32,
                                  tf.random_normal_initializer( stddev=0.02 ) )
    b = tf.get_variable( "b", [vocab_size],
                                 initializer=tf.constant_initializer( 0.0 ))
    logits = [tf.matmul( o, W ) + b for o in outputs]

    # call seq2seq.sequence_loss
    const_weights = [tf.ones([batch_size]) for i in xrange(sequence_length)]
    loss = seq2seq.sequence_loss(logits, targets, const_weights)

    # create a training op using the Adam optimizer
    optim = tf.train.AdamOptimizer( 0.001, beta1=0.5 ).minimize( loss )

# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.
with tf.variable_scope("RNN", reuse=True):
    batch_size = 1
    s_inputs = tf.placeholder( tf.int32, [ batch_size ], name='s_inputs' )
    s_onehot = tf.one_hot( s_inputs, vocab_size, name="s_input_onehot" )
    s_profile = tf.placeholder( tf.float32, [ batch_size, 14 ], name="s_profile" )
    #inputs_s = tf.split( 1, 1, s_onehot )
    #inputs_s = [ tf.squeeze(input_, [1]) for input_ in inputs_s ]

    s_initial_state = stacked_cells.zero_state(batch_size, tf.float32)

    s_onehot = tf.concat([s_onehot, s_profile], 1)
    s_mixed_inputs = tf.nn.relu(tf.matmul( s_onehot, W_mix ) + b_mix)

    # call seq2seq.rnn_decoder
    s_outputs, s_final_state = seq2seq.rnn_decoder([s_mixed_inputs], s_initial_state, stacked_cells)

    # transform the list of state outputs to a list of logits.
    # use a linear transformation.
    s_outputs = tf.reshape(s_outputs, [1, state_dim])
    s_probs = tf.nn.softmax(tf.matmul( s_outputs, W ) + b)

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='this is a ted talk' ):

    # prime the pump
    sample_profile = np.array([1.0] * 14)
    sample_profile[3] = 100.0
    sample_profile /= np.sum(sample_profile)
    sample_profile = sample_profile.reshape((1, 14))

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for word in prime.split(" "):
        x = np.ravel( data_loader.vocab[word] ).astype('int32')
        feed = { s_inputs:x, s_profile:sample_profile }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    for n in range(num):
        x = np.ravel( data_loader.vocab[word] ).astype('int32')

        # plug the most recent character in...
        feed = { s_inputs:x, s_profile:sample_profile }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        #sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        word = data_loader.vocab_list[sample]
        ret += " " + word

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./oh_tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y,profile_vec = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y, profile: profile_vec }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%100==0:
            print "%d %d\t%.4f" % ( j, i, lt ) 
            lts.append( lt )
            print sample( num=160, prime="this is a ted talk" )

    print sample( num=160, prime="this is a ted talk" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
