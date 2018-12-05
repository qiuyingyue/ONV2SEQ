#mnist_classification.py
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("./mnist")
import tensorflow as tf
from tensorflow.contrib import keras
import mnist
import numpy as np
import nn
#mnist = keras.datasets.mnist

def network(x):
	x = tf.reshape(x, [-1,28*28])
	print ("x", x)
	#fc1 = tf.layers.dense(x, 500, name="fc1")
	fc1=nn.my_fc_layer(x, "fc1", output_dim=500, apply_dropout=True)
	print ("fc1", fc1)
	#fc2 = tf.layers.dense(fc1, 25, name="fc2")
	fc2=nn.my_fc_layer(fc1, "fc2", output_dim=25, apply_dropout=True)
	print ("fc2", fc2)
	#fc3 = tf.layers.dense(fc2, 10, name="fc3")
	fc3=nn.my_fc_layer(fc2, "fc3", output_dim=10, apply_dropout=True)
	print ("fc3", fc3)
	return fc3

def convert_label(idx):
    arr = np.zeros((10))
    arr[idx] = 1
    return arr

#load data
def load_mnist_data():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    train_labels = [convert_label(idx) for idx in train_labels]
    train_labels = np.array(train_labels)
   
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    test_labels = [convert_label(idx) for idx in test_labels]
    test_labels = np.array(test_labels)
  
    
    
    train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels


batch_size = 100
img_size = 28

# Construct model
X = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
Y = tf.placeholder(tf.int32, [None,10])
logits = network(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def random_batch(batch_size, data, label):

	indices = np.random.randint(0, len(data), batch_size)
	return data[indices], label[indices]


train_images, train_labels, test_images, test_labels = load_mnist_data()
print (train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    num_steps = 100000
    display_step = 100
    for step in range(1, num_steps+1):

        batch_x, batch_y = random_batch(batch_size, train_images, train_labels)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and training accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            #print("prediction[0]", prediction[0], "logits[0]", logits[0])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            # Calculate batch loss and validation accuracy
            batch_x, batch_y = random_batch(batch_size, test_images, test_labels)
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", validation Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test onvs
    batch_x, batch_y = random_batch(batch_size, test_images, test_labels)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: batch_y}))
    for index in range(len(test_onvs)):
    	onv = test_onvs[index]
    	label = test_labels[index]
    	pred, accu = sess.run([correct_pred, accuracy], feed_dict={X:[onv], Y:[label]})
    	#print ("pred label:", pred, "accu", accu)
    	# if (accu==0):

    	# 	im = onv.fromarray(datasets[8][index])
    	# 	im.save(str(index)+".jpeg")