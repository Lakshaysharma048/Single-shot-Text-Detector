from my_ssd_300 import ssd_300
from my_priors import prior_util
from my_inputgenerator import inputgenerator
from my_ssd_loss import ssd_loss

import os
import pickle
import h5py

from keras import optimizers
from keras import regularizers
from keras import callbacks

from my_GTdataset import GTUtility
# with open('gt_util_synthtext.pkl', 'rb') as f:    #pickle file for GT data
#         gt_util = pickle.load(f)
#
#gt_util_train, gt_util_val = gt_util.split(split=0.8)

from data_icdar2015fst import GTUtility
gt_util_train = GTUtility('..')
gt_util_val = GTUtility('..', test=True)

model=ssd_300()        #Model Architecture

prior_util=prior_util(model)     #Generating priors

weight_path=os.path.join('..','textbox','vgg16_weights_tf_dim_ordering_tf_kernels.h5')

model.load_weights(weight_path,by_name=True)

freeze=['conv1_1','conv1_2',
        'conv2_1','conv2_2',
        'conv3_1','conv3_2','conv3_3',
        'conv4_1','conv4_2','conv4_3']

for l in model.layers:
    l.trainable=not l.name in freeze

#model parameters

epoch=20
batch_size=32

gen_train= inputgenerator(gt_util_train,prior_util,batch_size,model.image_size)

gen_val=inputgenerator(gt_util_val,prior_util,batch_size,model.image_size)

for l in model.layers:
    l.trainable= not l.name in freeze

checkdir = './checkpoints/tb300_epoch_files'
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

optim=optimizers.sgd(lr=1e-3,momentum=0.9,decay=0,nesterov=True)

regularizer=regularizers.l2(5e-2)

for l in model.layers:

    if l.__class__.__name__.startswith('conv'):
        l.kernal_regularizer=regularizer

#loss function

loss = ssd_loss(alpha=1.0, neg_pos_ratio=3.0)  #calculating the loss function

model.compile(optimizer=optim,loss=loss.compute,metrics=loss.metrics)

#model training starts here

history=model.fit_generator(gen_train.generate(),
                            steps_per_epoch=gen_train.num_batches,
                            epochs=20,
                            verbose=1,
                            callbacks=[callbacks.ModelCheckpoint('./checkpoints/tb300_epoch_files/weights.{epoch:03d}.h5',
                                                                verbose=1,save_weights_only=True)],
                            validation_data=gen_val.generate(),
                            validation_steps=gen_val.num_batches,
                            class_weight=None,
                            workers=1)






