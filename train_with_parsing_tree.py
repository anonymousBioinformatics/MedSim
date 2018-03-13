# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import datetime
import os


class Settings(object):
	def __init__(self):
		self.vocab_size = 494725
		self.num_steps = 100
		self.num_epochs = 50
		self.num_classes = 26
		self.gru_size = 280
		self.keep_prob = 0.5
		self.num_layers = 1
		# the number of entity pairs of each batch during training or testing
		self.big_num = 100


		self.pos_size = 10
		self.pos_num = 123

		# ssc
		self.root_num = 4
		self.root_size = 3

		self.e1_num = 5
		self.e1_size = 5

		self.e2_num = 5
		self.e2_size = 5

		self.dep_num = 14+1
		self.dep_size = 10

		self.pospeech1_num = 27
		self.pospeech1_size = 10

		self.pospeech2_num = 100
		self.pospeech2_size = 10




		self.data_ratio =  0.3
		self.num_epochs = int(self.num_epochs / self.data_ratio)


class GRU:
	def __init__(self, is_training, word_embeddings, settings):

		self.num_steps = num_steps = settings.num_steps
		self.vocab_size = vocab_size = settings.vocab_size
		self.num_classes = num_classes = settings.num_classes
		self.gru_size = gru_size = settings.gru_size
		self.big_num = big_num = settings.big_num

		self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_word')
		self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos1')
		self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pos2')

		# ssc
		self.input_root = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_root')
		self.input_e1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_e1')
		self.input_e2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_e2')
		self.input_dep = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_dep')
		self.input_pospeech1 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pospeech1')
		self.input_pospeech2 = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name='input_pospeech2')


		self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
		self.total_shape = tf.placeholder(dtype=tf.int32, shape=[big_num + 1], name='total_shape')
		total_num = self.total_shape[-1]

		word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
		pos1_embedding = tf.get_variable('pos1_embedding', [settings.pos_num, settings.pos_size])
		pos2_embedding = tf.get_variable('pos2_embedding', [settings.pos_num, settings.pos_size])


		# ssc
		root_embedding = tf.get_variable('root_embedding', [settings.root_num, settings.root_size])
		e1_embedding = tf.get_variable('e1_embedding', [settings.e1_num, settings.e1_size])
		e2_embedding = tf.get_variable('e2_embedding', [settings.e2_num, settings.e2_size])
		dep_embedding = tf.get_variable('dep_embedding', [settings.dep_num, settings.dep_size])
		pospeech1_embedding = tf.get_variable('pospeech1_embedding', [settings.pospeech1_num, settings.pospeech1_size])
		pospeech2_embedding = tf.get_variable('pospeech2_embedding', [settings.pospeech2_num, settings.pospeech2_size])


		attention_w = tf.get_variable('attention_omega', [gru_size, 1])
		sen_a = tf.get_variable('attention_A', [gru_size])
		sen_r = tf.get_variable('query_r', [gru_size, 1])
		relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, gru_size])
		sen_d = tf.get_variable('bias_d', [self.num_classes])

		gru_cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
		gru_cell_backward = tf.nn.rnn_cell.GRUCell(gru_size)

		"""
        :param is_training: 是否要进行训练.如果is_training=False,则不会进行参数的修正。
        """
		if is_training and settings.keep_prob < 1:
			gru_cell_forward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_forward, output_keep_prob=settings.keep_prob)
			gru_cell_backward = tf.nn.rnn_cell.DropoutWrapper(gru_cell_backward, output_keep_prob=settings.keep_prob)

		cell_forward = tf.nn.rnn_cell.MultiRNNCell([gru_cell_forward] * settings.num_layers)
		cell_backward = tf.nn.rnn_cell.MultiRNNCell([gru_cell_backward] * settings.num_layers)

		sen_repre = []
		sen_alpha = []
		sen_s = []
		sen_out = []
		self.prob = []
		self.predictions = []
		self.loss = []
		self.accuracy = []
		self.total_loss = 0.0

		self._initial_state_forward = cell_forward.zero_state(total_num, tf.float32)
		self._initial_state_backward = cell_backward.zero_state(total_num, tf.float32)

		# embedding layer
		inputs_forward = tf.concat(2, [tf.nn.embedding_lookup(word_embedding, self.input_word),
		                               tf.nn.embedding_lookup(pos1_embedding, self.input_pos1),
		                               tf.nn.embedding_lookup(pos2_embedding, self.input_pos2),
		                               tf.nn.embedding_lookup(root_embedding, self.input_root),
		                               tf.nn.embedding_lookup(e1_embedding, self.input_e1),
		                               tf.nn.embedding_lookup(e2_embedding, self.input_e2),
		                               tf.nn.embedding_lookup(dep_embedding, self.input_dep),
		                               tf.nn.embedding_lookup(pospeech1_embedding, self.input_pospeech1),
		                               tf.nn.embedding_lookup(pospeech2_embedding, self.input_pospeech2)
		                               ])
		inputs_backward = tf.concat(2,
		                            [tf.nn.embedding_lookup(word_embedding, tf.reverse(self.input_word, [False, True])),
		                             tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos1, [False, True])),
		                             tf.nn.embedding_lookup(pos1_embedding, tf.reverse(self.input_pos2, [False, True])), # 是不是写错了？？？？？
		                             tf.nn.embedding_lookup(root_embedding, tf.reverse(self.input_root, [False, True])),
		                             tf.nn.embedding_lookup(e1_embedding, tf.reverse(self.input_e1, [False, True])),
		                             tf.nn.embedding_lookup(e2_embedding, tf.reverse(self.input_e2, [False, True])),
		                             tf.nn.embedding_lookup(dep_embedding, tf.reverse(self.input_dep, [False, True])),
		                             tf.nn.embedding_lookup(pospeech1_embedding, tf.reverse(self.input_pospeech1, [False, True])),
		                             tf.nn.embedding_lookup(pospeech2_embedding, tf.reverse(self.input_pospeech2, [False, True]))
		                             ])

		outputs_forward = []

		state_forward = self._initial_state_forward

		# Bi-GRU layer
		with tf.variable_scope('GRU_FORWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
				outputs_forward.append(cell_output_forward)

		outputs_backward = []

		state_backward = self._initial_state_backward
		with tf.variable_scope('GRU_BACKWARD'):
			for step in range(num_steps):
				if step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :], state_backward)
				outputs_backward.append(cell_output_backward)

		output_forward = tf.reshape(tf.concat(1, outputs_forward), [total_num, num_steps, gru_size])
		output_backward = tf.reverse(tf.reshape(tf.concat(1, outputs_backward), [total_num, num_steps, gru_size]),
		                             [False, True, False])

		# # DELETE word-level attention layer
		# output_h = tf.add(output_forward, output_backward)[::,-1,::]
		# # attention_r = tf.reshape(tf.batch_matmul(tf.reshape(tf.nn.softmax(
		# #     tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
		# #                [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])


		# word-level attention layer
		output_h = tf.add(output_forward, output_backward)
		attention_r = tf.reshape(tf.batch_matmul(tf.reshape(tf.nn.softmax(
			tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
			           [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])


		# sentence-level attention layer
		for i in range(big_num):

			sen_repre.append(tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]]))
			batch_size = self.total_shape[i + 1] - self.total_shape[i]

			sen_alpha.append(
				tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.mul(sen_repre[i], sen_a), sen_r), [batch_size])),
				           [1, batch_size]))

			sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [gru_size, 1]))
			sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [self.num_classes]), sen_d))

			self.prob.append(tf.nn.softmax(sen_out[i]))

			with tf.name_scope("output"):
				self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

			with tf.name_scope("loss"):
				self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(sen_out[i], self.input_y[i])))
				if i == 0:
					self.total_loss = self.loss[i]
				else:
					self.total_loss += self.loss[i]

			with tf.name_scope("accuracy"):
				self.accuracy.append(
					tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
					               name="accuracy"))


def main(_):
	save_path = './sample_model for parsing tree/'

	print 'reading wordembedding'
	wordembedding = np.load('./data for parsing tree/vec.npy')

	print 'reading training data'

	train_y = np.load('./data for parsing tree/train_y.npy')
	train_word = np.load('./data for parsing tree/train_word.npy')
	train_pos1 = np.load('./data for parsing tree/train_pos1.npy')
	train_pos2 = np.load('./data for parsing tree/train_pos2.npy')
	train_root = np.load('./data for parsing tree/train_root.npy')
	train_e1 = np.load('./data for parsing tree/train_e1.npy')
	train_e2 = np.load('./data for parsing tree/train_e2.npy')
	train_dep = np.load('./data for parsing tree/train_dep.npy')
	train_pospeech1 = np.load('./data for parsing tree/train_pospeech1.npy')
	train_pospeech2 = np.load('./data for parsing tree/train_pospeech2.npy')

	settings = Settings()
	settings.vocab_size = len(wordembedding)
	settings.num_classes = len(train_y[0])


	lenn = len(train_word) * settings.data_ratio
	lenn = int(lenn)
	train_word = train_word[:lenn]
	train_pos1 = train_pos1[:lenn]
	train_pos2 = train_pos2[:lenn]
	train_root = train_root[:lenn]
	train_e1 = train_e1[:lenn]
	train_e2 = train_e2[:lenn]
	train_dep = train_dep[:lenn]
	train_pospeech1 = train_pospeech1[:lenn]
	train_pospeech2 = train_pospeech2[:lenn]


	big_num = settings.big_num

	with tf.Graph().as_default():

		sess = tf.Session()
		with sess.as_default():

			initializer = tf.contrib.layers.xavier_initializer()
			with tf.variable_scope("model", reuse=None, initializer=initializer):
				m = GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(0.001)

			train_op = optimizer.minimize(m.total_loss, global_step=global_step)
			sess.run(tf.initialize_all_variables())
			saver = tf.train.Saver(max_to_keep=None)

			def train_step(word_batch, pos1_batch, pos2_batch, root_batch, e1_batch, e2_batch, dep_batch, pospeech1_batch, pospeech2_batch, y_batch, big_num):
				feed_dict = {}
				total_shape = []
				total_num = 0
				total_word = []
				total_pos1 = []
				total_pos2 = []
				total_root = []
				total_e1 = []
				total_e2 = []
				total_dep = []
				total_pospeech1 = []
				total_pospeech2 = []
				for i in range(len(word_batch)):
					total_shape.append(total_num)
					total_num += len(word_batch[i])
					for word in word_batch[i]:
						total_word.append(word)
					for pos1 in pos1_batch[i]:
						total_pos1.append(pos1)
					for pos2 in pos2_batch[i]:
						total_pos2.append(pos2)
					for root in root_batch[i]:
						total_root.append(root)
					for e1 in e1_batch[i]:
						total_e1.append(e1)
					for e2 in e2_batch[i]:
						total_e2.append(e2)
					for dep in dep_batch[i]:
						total_dep.append(dep)
					for pospeech1 in pospeech1_batch[i]:
						total_pospeech1.append(pospeech1)
					for pospeech2 in pospeech2_batch[i]:
						total_pospeech2.append(pospeech2)

				total_shape.append(total_num)
				total_shape = np.array(total_shape)
				total_word = np.array(total_word)
				total_pos1 = np.array(total_pos1)
				total_pos2 = np.array(total_pos2)
				total_root = np.array(total_root)
				total_e1 = np.array(total_e1)
				total_e2 = np.array(total_e2)
				total_dep = np.array(total_dep)
				total_pospeech1 = np.array(total_pospeech1)
				total_pospeech2 = np.array(total_pospeech2)

				feed_dict[m.total_shape] = total_shape
				feed_dict[m.input_word] = total_word
				feed_dict[m.input_pos1] = total_pos1
				feed_dict[m.input_pos2] = total_pos2
				feed_dict[m.input_root] = total_root
				feed_dict[m.input_e1] = total_e1
				feed_dict[m.input_e2] = total_e2
				feed_dict[m.input_dep] = total_dep
				feed_dict[m.input_pospeech1] = total_pospeech1
				feed_dict[m.input_pospeech2] = total_pospeech2

				feed_dict[m.input_y] = y_batch

				temp, step, loss, accuracy = sess.run([train_op, global_step, m.total_loss, m.accuracy], feed_dict)
				time_str = datetime.datetime.now().isoformat()
				accuracy = np.reshape(np.array(accuracy), (big_num))
				acc = np.mean(accuracy)

				if step % 100 == 0 or (step < 100 and step % 10 == 0):
					print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))

			for one_epoch in range(settings.num_epochs):
				temp_order = range(len(train_word))
				np.random.shuffle(temp_order)
				for i in range(int(len(temp_order) / float(settings.big_num))):

					temp_word = []
					temp_pos1 = []
					temp_pos2 = []
					temp_root = []
					temp_e1 = []
					temp_e2 = []
					temp_dep = []
					temp_pospeech1 = []
					temp_pospeech2 = []
					temp_y = []

					temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
					for k in temp_input:
						temp_word.append(train_word[k])
						temp_pos1.append(train_pos1[k])
						temp_pos2.append(train_pos2[k])
						temp_root.append(train_root[k])
						temp_e1.append(train_e1[k])
						temp_e2.append(train_e2[k])
						temp_dep.append(train_dep[k])
						temp_pospeech1.append(train_pospeech1[k])
						temp_pospeech2.append(train_pospeech2[k])
						temp_y.append(train_y[k])
					num = 0
					for single_word in temp_word:
						num += len(single_word)

					if num > 1500:
						print 'out of range'
						continue

					temp_word = np.array(temp_word)
					temp_pos1 = np.array(temp_pos1)
					temp_pos2 = np.array(temp_pos2)
					temp_root = np.array(temp_root)
					temp_e1 = np.array(temp_e1)
					temp_e2 = np.array(temp_e2)
					temp_dep = np.array(temp_dep)
					temp_pospeech1 = np.array(temp_pospeech1)
					temp_pospeech2 = np.array(temp_pospeech2)
					temp_y = np.array(temp_y)

					train_step(temp_word, temp_pos1, temp_pos2, temp_root, temp_e1, temp_e2, temp_dep, temp_pospeech1, temp_pospeech2, temp_y, settings.big_num)

					current_step = tf.train.global_step(sess, global_step)
					print current_step
					if (current_step <=1000 and current_step % 100 == 0) or (current_step < 100 and current_step % 20 == 0) or (current_step < 5000 and current_step % 500 == 0) or (current_step % 1000 == 0):
						print 'saving model'
						path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
						tempstr = 'have saved model to ' + path
						print tempstr

					if current_step >= 5010:
						return


if __name__ == "__main__":
	tf.app.run()

