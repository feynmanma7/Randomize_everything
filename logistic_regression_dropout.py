#encoding:utf-8
# version: tf1.9

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

def main():
	mnist = read_data_sets("MNIST-data", one_hot=True)

	X = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])

	X = tf.nn.dropout(X, keep_prob=0.5)

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	pred = tf.nn.softmax(tf.matmul(X, W) + b)

	#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)), axis=-1)
	cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))

	lr = 0.01
	optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

	init = tf.global_variables_initializer()

	num_epochs = 100
	batch_size = 100
	display_step = 1

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			total_cost = 0

			total_batch = int(mnist.train.num_examples/batch_size)

			for batch in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)

				_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, y:batch_ys})

				total_cost += c
				# avg_cost += c / total_batch

			avg_cost = total_cost / total_batch

			if (epoch + 1) % display_step == 0:
				#print('Epoch: %04d, cost={:.9f}' % (epoch+1, avg_cost))
				print('Epoch: %s, cost=%s' % (epoch+1, avg_cost))

		print("Optimization Finished!")

		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		print("Accuracy: %s" % accuracy.eval({X: mnist.test.images, y:mnist.test.labels}))


if __name__ == '__main__':
	main()