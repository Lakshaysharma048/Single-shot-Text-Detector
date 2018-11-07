import numpy as np
import cv2
import os


# Here the batch of 32 images are processed and send to returned to model.fit_generator.

# Also, IOU are also calculated and respective true mask values are send for training.

class inputgenerator:

    def __init__(self,gt_util,prior_util,batch_size,input_size):

        self.__dict__.update(locals())

        self.num_batches= gt_util.num_samples//batch_size

    def generate(self):
        h, w = self.input_size
        mean = np.array([104, 117, 123])
        gt_util = self.gt_util
        batch_size = self.batch_size
        num_batches = self.num_batches

        inputs, targets = [], []

        while True:
            idxs = np.arange(gt_util.num_samples)
            np.random.shuffle(idxs)
            idxs = idxs[:num_batches * batch_size]
            for j, i in enumerate(idxs):
                img_name = gt_util.image_names[i]
                img_path = os.path.join(gt_util.image_path, img_name)
                img = cv2.imread(img_path)
                y = np.copy(gt_util.data[i])

                img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
                img = img.astype(np.float32)


                img -= mean[np.newaxis, np.newaxis, :]

                inputs.append(img)
                targets.append(y)

                # if len(targets) == batch_size or j == len(idxs)-1: # last batch in epoch can be smaller then batch_size
                if len(targets) == batch_size:

                    targets = [self.prior_util.encode(y) for y in targets]
                    targets = np.array(targets, dtype=np.float32)
                    tmp_inputs = np.array(inputs, dtype=np.float32)
                    tmp_targets = targets
                    inputs, targets = [], []
                    yield tmp_inputs, tmp_targets
                elif j == len(idxs) - 1:
                    # forget last batch
                    inputs, targets = [], []
                    break

            print('NEW epoch')
        print('EXIT generator')

