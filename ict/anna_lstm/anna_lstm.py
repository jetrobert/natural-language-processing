from __future__ import print_function
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf
import os

#os.environ['CUDA_VISIBLE_DEVICES']=''

# split mini-batch

def get_batches(arr, n_seqs, n_steps):
    '''
    arr: array
    n_seqs: sequence nums in a batch
    n_steps: word nums in a sequence
    '''
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n+n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

# build the model

def build_inputs(num_seqs, num_steps):
    '''
    num_seqs: sequence nums in a batch
    num_steps: word nums in a sequence
    '''
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')
    # add keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob
	
def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.NASCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' 
    num_layers: lstm hidden layers
    batch_size: batch_size
    '''
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size, 
	    keep_prob) for _ in range(num_layers)], state_is_tuple = True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    # tf.concat(concat_dim, values)
    seq_output = tf.concat(lstm_output, axis=1) 
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))  
    # calculate logits
    logits = tf.matmul(x, softmax_w) + softmax_b 
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits
	
# loss

def build_loss(logits, targets, lstm_size, num_classes):
    '''
    logits: fully_connected output
    targets: targets
    lstm_size
    num_classes: vocab_size    
    '''
    # One-hot encoding
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)    
    return loss
	
# optimizer

def build_optimizer(loss, learning_rate, grad_clip):
    # clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))   
    return optimizer
	
# rnn dynamic_run 

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128, 
	    num_layers=2, learning_rate=0.001, grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()      
        # input layer
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        # LSTM
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        # one-hot encoding
        x_one_hot = tf.one_hot(self.inputs, num_classes)        
        # run RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state        
        # predict
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)        
        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
		
# generate top words

def pick_top_n(preds, vocab_size, top_n=5):
    """
    predict top-5 words 
    preds: prediction results
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    # normailized prob
    p = p / np.sum(p)
    # randomly choose one word
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
	
# generate new words from checkpoints

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    """
    checkpoint: ckpts
    n_sample: word length
    lstm_size: hidden cells
    vocab_size
    prime: starting words
    """
    # transfers words to single string
    samples = [c for c in prime]
    # sampling=True --> batch size = 1 x 1
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # load ckpts and train
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            # input single word
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
        c = pick_top_n(preds, len(vocab))
        # add words into samples
        samples.append(int_to_vocab[c])
        # generate words 
        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
    return ''.join(samples)
	
# load data and preprocess

with open('anna.txt', 'r') as f:
    text=f.read()
text = text.strip().split()
vocab = set(text)
print (vocab)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
print(text[:100])
print(encoded[:100])
print(len(vocab))

# batch setting 

batches = get_batches(encoded, 10, 50)
x, y = next(batches)
print('x\n', x[:10, :10])
print('y\n', y[:10, :10])

# params setting

batch_size = 100         # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 20
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
        lstm_size=lstm_size, num_layers=num_layers, learning_rate=learning_rate)

# model save

saver = tf.train.Saver(max_to_keep=100)
ckpt_path = "checkpoints/"

if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)

# training

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                model.final_state, model.optimizer], feed_dict=feed)   
            end = time.time()
            # control the print lines
            if counter % 100 == 0:
                print('epoch: {}/{}... '.format(e+1, epochs),
                      'train step: {}... '.format(counter),
                      'loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))
            if (counter % save_every_n == 0):
                saver.save(sess, ckpt_path + "i{}_l{}.ckpt".format(counter, lstm_size))
    saver.save(sess, ckpt_path + "i{}_l{}.ckpt".format(counter, lstm_size))

# check checkpoints

tf.train.get_checkpoint_state('checkpoints')

# pick up the last ckpt to generate words

tf.train.latest_checkpoint('checkpoints')

checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="The")
print(samp)

checkpoint = 'checkpoints/i200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

checkpoint = 'checkpoints/i1000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

checkpoint = 'checkpoints/i2000_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)

