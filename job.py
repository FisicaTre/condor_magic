import FIF
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import datetime
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the file with the data to read")
ap.add_argument("--id", required=True, help="batch id")
ap.add_argument("--logs", required=True, help="path to the folder that will contain logs")
args = vars(ap.parse_args())

data_file = args["data"]
l = args["id"]
logs_path = args["logs"]

# load dataset
test_dataset = tf.data.experimental.load(data_file)

mv_fif = FIF.MvFIF_v8

# L = 2  # should be 2848 for training and 311 for val test
# K = 2  # should be 1024
single_batch = []  # [0] * K
# new_dataset = [0] * L
IMFS_new = np.zeros((4, 120, 8))

# test_dataset.get_single_element()

# for l, (samples, target) in enumerate(
#        test_dataset.take(2)):  # indice membri del dataset .take(2848) per il train o .take(311) per val e test

samples, target = None, None
for i, element in test_dataset.as_numpy_iterator():
    if i == l:
        samples, target = element[0], element[1]

if samples is not None and target is not None:
    temp = np.zeros((1024, 120, 32))
    for k in range(1024):  # indice membri del batch k=0...1024-1
        # IMFS_provv = np.concatenate((samples[k, :, :], samples[k, :, :], samples[k, :, :], samples[k, :, :]), axis=1)
        # parametri per essere sicuri che estragga sempre almeno NIMFS = tot?
        print(f'\n\n analysis: k:{k} l:{l} \n\n')
        IMFS = mv_fif.FIF_run(samples[k, :, :], delta=0.001, alpha=30, NumSteps=1, ExtPoints=3, NIMFs=3, MaxInner=200,
                              Xi=1.6, MonotoneMaskLength=True)
        IMFS = IMFS[0]  # IMFS è un tuple, il primo elemento è il tensore (4, 8, 120) che ha le IMF

        if IMFS.shape[0] != 4:
            # IMFS=tf.stack([IMFS,IMFS,IMFS,IMFS],axis=1)
            for i in range(4):
                IMFS_new[k, :, :] = IMFS
        if IMFS.shape[0] != 4:
            IMFS = IMFS_new
            IMFS_new = np.zeros((4, 120, 8))  # re initialize tensor

        print(f'\n imfs shape:{IMFS.shape}\n')

        IMFS_reshaped = tf.unstack(IMFS, axis=0)

        # works only if you extract 3 imfs pluts trend EACH time
        IMFS_conc = np.concatenate(
            (IMFS_reshaped[0][:, :], IMFS_reshaped[1][:, :], IMFS_reshaped[2][:, :], IMFS_reshaped[3][:, :]), axis=1)

        temp[k, :, :] = IMFS_conc

        print(f'\n\n samples.shape :{samples.shape}\n\n')

        new_samples = np.append(samples, temp, 2)

        print(f' \n\n samples.shape:{new_samples.shape} should be (1024,120,32+8) for 3 IMFs+trend\n\n ')

        # here create tuple with new samples and targets (targets change based on FORECAST_value
        new_samples = tf.convert_to_tensor(new_samples)
        target = tf.convert_to_tensor(target)

        batch_member = (new_samples,
                        target)  # samples che includono le imf e rispettivi target. di batch_member ce ne sono 1024 per ogni batch (in tutto ci sono 3mila batch per training 311 per val test)

        single_batch.append(batch_member)

        print(f'k is:{k}')

        del IMFS, IMFS_reshaped, IMFS_conc
        # new_dataset[l] = single_batch

# save single_batch
tf.data.experimental.save(single_batch, "batch_{:d}".format(l))

# create dataset
# https://stackoverflow.com/questions/73881400/convert-list-of-tuples-to-tensorflow-dataset-tf-data-dataset
# x, y = zip(*new_dataset)
# dataset = tf.data.Dataset.from_tensor_slices((list(x), list(y)))

# dataset = tf.data.Dataset.from_tensor_slices(
#    new_dataset)  # da errore perchè le shape degli elementi del tuple sono diverse

# then dataset.batch to recreate batch dataset to feed to model.fit
# batched_new_dataset=dataset.batch(1024)