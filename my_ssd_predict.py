import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import os
from my_ssd_300 import ssd_300
from my_priors import prior_util


from data_icdar2015fst import GTUtility    #Dataset for testing the model
gt_util_train = GTUtility('..')
#gt_util_val = GTUtility('..', test=True)

weight_path=os.path.join('.','weights.061.h5')   #weight path for the model

model=ssd_300()
prior_util=prior_util(model)

model.load_weights(weight_path)    #Loading trained weights

print(model.summary())

import cv2
from my_BaseGTUtility import preprocess


def predict(img_name, thresh=0.60):
    img1 = cv2.imread(img_name)
    img = preprocess(img1, size=(300, 300))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, batch_size=1, verbose=1)
    res = prior_util.decode(preds[0], confidence_threshold=thresh, keep_top_k=100)

    plt.figure(figsize=[10] * 2)
    plt.imshow(img1)
    prior_util.plot_results(res, classes=gt_util_train.classes, show_labels=True, image=img1)  # gt_data=data[i]
    plt.show()


if __name__ == '__main__':

    img_list=os.listdir(os.path.join('.'))   #sending all images to predict function
    for i in img_list:
        if i.endswith('jpg'):
            image_path=os.path.join('.',i)
            predict(image_path)
