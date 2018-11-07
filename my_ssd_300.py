
from keras.layers import Input,Conv2D,MaxPool2D,ZeroPadding2D,Flatten,concatenate
from keras.layers import Reshape, Activation
from keras.engine import Model
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import keras.backend as K
import numpy as np

def ssd_300_body(x):
    textbox_layers=[]
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    textbox_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    textbox_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, 3, strides=2, padding='valid', name='conv6_2', activation='relu')(x)
    textbox_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    textbox_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv8_2', activation='relu')(x)
    textbox_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='valid', name='conv9_2', activation='relu')(x)
    textbox_layers.append(x)

    return textbox_layers

def multibox(textbox_layers,num_priors,num_classes,normalizations=None,softmax=True):
    mbox_conf = []
    mbox_loc = []
    for i in range(len(textbox_layers)):
        x = textbox_layers[i]
        name = x.name.split('/')[0]

        #normalize
        if normalizations is not None and normalizations[i] > 0:
            name = name + '_norm'
            x = Normalize(normalizations[i], name=name)(x)

        # confidence
        name1 = name + '_mbox_conf'
        x1 = Conv2D(num_priors * num_classes, (1, 5), padding='same', name=name1)(x)
        x1 = Flatten(name=name1 + '_flat')(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + '_mbox_loc'
        x2 = Conv2D(num_priors * 4, (1, 5), padding='same', name=name2)(x)
        x2 = Flatten(name=name2 + '_flat')(x2)
        mbox_loc.append(x2)

    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    if softmax:
        mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    else:
        mbox_conf = Activation('sigmoid', name='mbox_conf_final')(mbox_conf)

    output_tensor = concatenate([mbox_loc, mbox_conf], axis=2, name='output_tensor')

    return output_tensor

class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.

    # Arguments
        scale: Default feature scale.

    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)

    # Output shape
        Same as input

    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf

    #TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name=self.name+'_gamma')
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


def ssd_300(input_shape=(300,300,3),num_classes=2,softmax=True):

    x=input_tensor=Input(shape=input_shape)
    textbox_layers=ssd_300_body(x)

    num_priors=12
    normalizations = [20, -1, -1, -1, -1, -1]
    output_tensor=multibox(textbox_layers,num_priors,num_classes,normalizations,softmax)
    model=Model(input_tensor,output_tensor)
    model.num_classes=num_classes

    num_maps=len(textbox_layers)
    model.image_size=input_shape[:2]
    model.textbox_layers=textbox_layers
    model.aspect_ratios=[[1,2,3,5,7,10]*2]*num_maps
    model.shifts=[[(0.0,-0.5)]*6 + [(0.0,0.5)]*6]*num_maps
    model.steps = [8, 16, 32, 64, 128, 256, 512]

    return model
