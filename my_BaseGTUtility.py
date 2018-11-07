import numpy as np
import matplotlib.pyplot as plt
import cv2


class BaseGTUtility(object):
    """Base class for handling datasets.

    Derived classes should implement the following attributes and call the init methode:
        gt_path         str
        image_path      str
        classes         list of str, first class is normaly 'Background'
        image_names     list of str
        data            list of array (boxes, n * xy + one_hot_class)
    optional attributes are:
        text            list of list of str
    """

    def __init__(self):
        self.gt_path = ''
        self.image_path = ''
        self.classes = []
        self.image_names = []
        self.data = []

    def init(self):
        self.num_classes = len(self.classes)
        self.classes_lower = [s.lower() for s in self.classes]
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes) + 1)).tolist()

        # statistics
        stats = np.zeros(self.num_classes)
        num_without_annotation = 0
        for i in range(len(self.data)):
            # stats += np.sum(self.data[i][:,-self.num_classes:], axis=0)
            if len(self.data[i]) == 0:
                num_without_annotation += 1
            else:
                unique, counts = np.unique(self.data[i][:, -1].astype(np.int16), return_counts=True)
                stats[unique] += counts
        self.stats = stats
        self.num_without_annotation = num_without_annotation

        self.num_samples = len(self.image_names)
        self.num_images = len(self.data)
        self.num_objects = sum(self.stats)

    def split(self, split=0.8):
        gtu1 = BaseGTUtility()
        gtu1.gt_path = self.gt_path
        gtu1.image_path = self.image_path
        gtu1.classes = self.classes

        gtu2 = BaseGTUtility()
        gtu2.gt_path = self.gt_path
        gtu2.image_path = self.image_path
        gtu2.classes = self.classes

        n = int(round(split * len(self.image_names)))
        gtu1.image_names = self.image_names[:n]
        gtu2.image_names = self.image_names[n:]
        gtu1.data = self.data[:n]
        gtu2.data = self.data[n:]
        if hasattr(self, 'text'):
            gtu1.text = self.text[:n]
            gtu2.text = self.text[n:]

        gtu1.init()
        gtu2.init()
        return gtu1, gtu2


def preprocess(img, size):
    """Precprocess an image for ImageNet models.

    # Arguments
        img: Input Image
        size: Target image size (height, width).

    # Return
        Resized and mean subtracted BGR image, if input was also BGR.
    """
    h, w = size
    img = np.copy(img)
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    mean = np.array([104, 117, 123])
    img -= mean[np.newaxis, np.newaxis, :]
    return img
