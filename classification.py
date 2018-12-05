# TensorFlow and tf.keras
import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import tensorflow as tf

# Helper libraries
import numpy as np
import nn

from sketch_rnn_train_image import load_dataset
import model_cnn_encoder as sketch_rnn_model
from PIL import Image

def network(batch):
	 #cnn layers
    print ("batch: ", batch)
    h = tf.layers.conv2d(batch, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
    print ("h_conv1: ", h)
    h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
    print ("h_conv2: ", h)
    h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
    print ("h_conv3: ", h)
    h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
    print ("h_conv4: ", h)
    h = tf.reshape(h, [-1, 2*2*256])
    print ("h", h)
    fc1 = tf.layers.dense(h, 288, name="enc_fc1")
    fc2 = tf.layers.dense(fc1, 3, name="enc_fc2")
    return fc2


batch_size = 100
image_size = 64

# Load dataset
data_dir = "/home/qyy/workspace/data"
model_params = sketch_rnn_model.get_default_hparams()
datasets = load_dataset(data_dir, model_params, do_filter=False, contain_labels=True)

print ("shape of datasets[6]", datasets[6].shape)
train_images = np.reshape(datasets[6], (datasets[6].shape[0], image_size, image_size, 1))
valid_images = np.reshape(datasets[7], (datasets[7].shape[0], image_size, image_size, 1))
test_images = np.reshape(datasets[8], (datasets[8].shape[0], image_size, image_size, 1))

train_labels = datasets[9]
valid_labels = datasets[10]
test_labels = datasets[11]


train_images = train_images / 255.0
valid_images = valid_images / 255.0
test_images = test_images / 255.0


# Construct model
X = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
Y = tf.placeholder(tf.int32, [None,3])
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
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    num_steps = 300000
    display_step = 1000
    for step in range(1, num_steps+1):

        batch_x, batch_y = random_batch(batch_size, train_images, train_labels)

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and training accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            # Calculate batch loss and validation accuracy
            batch_x, batch_y = random_batch(batch_size, valid_images, valid_labels)
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", validation Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    batch_x, batch_y = random_batch(batch_size, test_images, test_labels)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: batch_y}))
    for index in range(len(test_images)):
    	image = test_images[index]
    	label = test_labels[index]
    	pred, accu = sess.run([correct_pred, accuracy], feed_dict={X:[image], Y:[label]})
    	#print ("pred label:", pred, "accu", accu)
    	if (accu==0):

    		im = Image.fromarray(datasets[8][index])
    		im.save("error/"+str(index)+".jpeg")

