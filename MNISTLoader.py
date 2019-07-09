from mnist import MNIST
import numpy as np

mndata = MNIST('samples')

def load_training_data():
    training_images, training_rawlabels = mndata.load_training()
    training_labels = []
    for l in training_rawlabels:
        newlabel = np.zeros(10).astype('uint8')
        newlabel[l] = 1
        training_labels.append(newlabel)
    return np.array(training_images).astype('uint8'), np.array(training_labels).astype('uint8')


#test_images, test_rawlabels = mndata.load_testing()
#test_labels = []