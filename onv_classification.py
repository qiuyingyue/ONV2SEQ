# TensorFlow and tf.keras
import sys
import os
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("./mnist")
import tensorflow as tf
import mnist
# Helper libraries
import numpy as np
import nn

from sketch_rnn_train_onv import load_dataset
import model_dnn_encoder as sketch_rnn_model
from PIL import Image
from onv_process import onv_convert_fromarr, show_onv

def network(x):
    print ("x", x)
    fc1 = tf.layers.dense(x, 1000, name="fc1")
    print ("fc1, ", fc1)
    fc2 = tf.layers.dense(fc1, 500, name="fc2")#nn.fc_layer(fc1, "fc2", output_dim=500, apply_dropout=True)
    print ("fc2, ", fc2)
    fc3 = tf.layers.dense(fc2, 250, name="fc3")#nn.fc_layer(fc2, "fc3", output_dim=250, apply_dropout=True)
    print ("fc3, ", fc3)
    fc4 = tf.layers.dense(fc3, 100, name="fc4")# nn.fc_layer(fc2, "fc4", output_dim=100, apply_dropout=True)
    print ("fc4, ", fc4)
    fc5 = tf.layers.dense(fc4, 50, name="fc5")#nn.fc_layer(fc4, "fc5", output_dim=50, apply_dropout=True)
    print ("fc5, ", fc5)
    fc6 = tf.layers.dense(fc5, 25, name="fc6")#nn.fc_layer(fc4, "fc6", output_dim=25, apply_dropout=True)
    print ("fc6, ", fc6)
    fc7 = tf.layers.dense(fc6, 4, name="fc7")#nn.fc_layer(fc6, "fc7", output_dim=10, apply_dropout=True)
    print ("fc7, ", fc7)
    return fc7
    
def network_dropout(x):
    fc1 = tf.layers.dense(x, 200, name="vector_rnn/fc1")#1000 200 256
    fc1 = tf.layers.dropout(fc1, rate=0.5, name="fc1_drop")
    #print ("fc1, ", fc1)
    fc2 = tf.layers.dense(fc1, 100, name="vector_rnn/fc2")#500 100 128
    fc2 = tf.layers.dropout(fc2, rate=0.5, name="fc2_drop")
    #print ("fc2, ", fc2)
    fc3 = tf.layers.dense(fc2, 50, name="vector_rnn/fc3")#250 50 64
    fc3 = tf.layers.dropout(fc3, rate=0.5, name="fc3_drop")
    #print ("fc3, ", fc3)
    fc4 = tf.layers.dense(fc3, 25, name="vector_rnn/fc4")#100 25 32
    fc4 = tf.layers.dropout(fc4, rate=0.5, name="fc4_drop")

    fc_last = tf.layers.dense(fc4, 5, name="fc_last")
    print ("fc_last, ", fc_last)
    return fc_last

batch_size = 100
onv_size = 9936

def load_data():
    # Load dataset
    data_dir = "/home/qyy/workspace/data"
    model_params = sketch_rnn_model.get_default_hparams()
    datasets = load_dataset(data_dir, model_params, contain_labels=True)

    train_onvs = datasets[6]
    valid_onvs = datasets[7]
    test_onvs = datasets[8]

    train_labels = datasets[9]
    valid_labels = datasets[10]
    test_labels = datasets[11]

    train_onvs = train_onvs / 255.0
    valid_onvs = valid_onvs / 255.0
    test_onvs = test_onvs / 255.0

    return train_onvs, train_labels, valid_onvs, valid_labels, test_onvs, test_labels

def convert_label(idx):
    arr = np.zeros((10))
    arr[idx] = 1
    return arr

def load_mnist_data():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    train_labels = [convert_label(idx) for idx in train_labels]
    train_labels = np.array(train_labels)
    train_onvs = [onv_convert_fromarr(I, resize=True) for I in train_images]
    train_onvs = np.array(train_onvs)
    print (train_images.shape, train_labels.shape, train_onvs.shape)

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    test_labels = [convert_label(idx) for idx in test_labels]
    test_labels = np.array(test_labels)
    test_onvs = [onv_convert_fromarr(I, resize=True) for I in test_images]
    test_onvs = np.array(test_onvs)

    train_onvs = train_onvs / 255.0
    test_onvs = test_onvs / 255.0

    return train_onvs, train_labels, test_onvs, test_labels



# Construct model
X = tf.placeholder(tf.float32, [None, onv_size])
Y = tf.placeholder(tf.int32, [None,5])
logits = network_dropout(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def random_batch(batch_size, data, label):

	indices = np.random.randint(0, len(data), batch_size)
	return data[indices], label[indices]



#create saver
saver = tf.train.Saver(tf.global_variables())
model_save_path = '../backup_models/onv_classification_model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
checkpoint_path = os.path.join(model_save_path, 'vector')

t_vars = tf.trainable_variables()
vars_list=[]
for var in t_vars:
    if ("fc1" in var.name or "fc2" in var.name or "fc3" in var.name or "fc4" in var.name):
        vars_list.append(var)
print (vars_list)
#train_onvs, train_labels, test_onvs, test_labels = load_mnist_data()
train_onvs, train_labels, valid_onvs, valid_labels, test_onvs, test_labels = load_data()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    num_steps = 1000000
    display_step = 1000
    for step in range(1, num_steps+1):

        batch_x, batch_y = random_batch(batch_size, train_onvs, train_labels)

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
            batch_x, batch_y = random_batch(batch_size, valid_onvs, valid_labels)
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", validation Accuracy= " + \
                  "{:.3f}".format(acc))


             # saving models
            if (step % 10  == 0):
                checkpoint_path_step = checkpoint_path + str(step)
                tf.logging.info('saving model %s.', checkpoint_path_step)
                tf.logging.info('global_step %i.', step)
                saver.save(sess, checkpoint_path, global_step=step)

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test onvs
    batch_x, batch_y = random_batch(batch_size, test_onvs, test_labels)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: batch_y}))



    '''for index in range(len(test_onvs)):
    	onv = test_onvs[index]
    	label = test_labels[index]
    	pred, accu = sess.run([correct_pred, accuracy], feed_dict={X:[onv], Y:[label]})'''
    	#print ("pred label:", pred, "accu", accu)
    	# if (accu==0):

    	# 	im = onv.fromarray(datasets[8][index])
    	# 	im.save(str(index)+".jpeg")

