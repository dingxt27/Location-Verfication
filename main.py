import addImages
import LoadData
import matplotlib.pyplot as plt
import cv2
import skimage.data
import numpy as np
import skimage.transform
import tensorflow as tf
from tensorflow.contrib.layers import flatten

training_data_dir = '/home/dingxt/PycharmProjects/BigdataProject/Location-Verfication/frames'
images,labels = LoadData.load_data(training_data_dir)
print(images[0].shape)
#print(labels)

newImgs, newLabels = addImages.equalize_samples_set(images,labels)
#dimension = newImgs[0].shape

#print(len(newLabels))
def display_image_labels(images,labels):
    unique_labels = set(labels)
    plt.figure(figsize = (15,15))
    i = 1
    for label in unique_labels:
        image = images[labels.index(label)]
        plt.subplot(8,8,i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i +=1
        #_ = plt.imshow(image)
        _=plt.imshow(image,cmap="gray")

    plt.show()

newImages_gray = []
#Histogram equalization & grayscale
for Im in newImgs:
    img_to_yuv = cv2.cvtColor(Im, cv2.COLOR_BGR2YUV)
    img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    Im = cv2.cvtColor(hist_equalization_result, cv2.COLOR_RGB2GRAY)[:, :, None]
    newImages_gray.append(Im)

dimension = newImages_gray[0].shape
print(dimension)

#resize img
images32 = [skimage.transform.resize(image,(32,32),mode = 'constant')
            for image in newImages_gray]

for img in images32[:5]:
    print("Shape: {0}, min: {1}, max: {2}".format(img.shape,img.min(),img.max()))
labels_a = np.array(newLabels)
images_a = np.array(images32)
#images_a = images_a.reshape(images_a.shape[0], images_a.shape[1], images_a.shape[2],1)
print("labels:", labels_a.shape, "\nimages: ",images_a.shape)

EPOCHS = 100
BATCH_SIZE = 50
image_depth = 1 #added by myself
n_classes =  5#added by myself


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 62.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))#1
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.003

logits = LeNet(x)

varss = tf.trainable_variables()
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in varss
                    if '_b' not in v.name ]) * 0.0001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy) + lossL2
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


from sklearn.utils import shuffle

cost_arr = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(images_a)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_processed, y_train = shuffle(images_a, labels_a)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_processed[offset:end], y_train[offset:end]
            to, cost = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})

        train_accuracy = evaluate(images_a, labels_a)
        print("EPOCH; {}; Train.Acc.; {:.3f}; Loss; {:.5f}".format(i + 1, train_accuracy, cost))
        cost_arr.append(cost)

    saver.save(sess, './lenet')
    print("Model saved")
