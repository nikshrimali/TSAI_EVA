# Code that gets the images that are misclassified by our model and convert it into a single figure

import numpy as np 
import matplotlib.pyplot as plt
index = 0

def plot_misclassified(misclassified:list):
    '''Gets 25 misclassified images and plotts them'''
    plt.figure(figsize=(20,20))
    for plotIndex, badIndex in enumerate(misclassified[0:25]):
        plt.subplot(5, 5, plotIndex + 1)
        plt.imshow(np.reshape(test_img[badIndex], (28,28)), cmap=plt.cm.gray)
        # plt.title(‘Predicted: {}, Actual: {}’.format(predictions[badIndex], test_lbl[badIndex]), fontsize = 15)
