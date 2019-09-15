import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import keras
from keras import layers, models
from keras.models import Model
import tensorflow as tf


class NeurISTTOK:

    def __init__(self):
        self.model = models.Sequential()
        self.h = 15
        self.w = 15
        self.d = 15
        self.k = 5
        self.l = 1

        self.test_detectors = np.load("resources/testdetectors.npy")

        geo_matrix_np = np.load('resources/geo_mat60-newpin-ref-halfmm-px5000.npy')
        geo_matrix_tf = tf.transpose(tf.convert_to_tensor(geo_matrix_np, dtype=tf.float32))

        divertor_pos_seq = np.load('resources/divertorpos.npy')
        divertor_pos = divertor_pos_seq.reshape(60, 60, 1)
        divertor_pos_tf = tf.convert_to_tensor(divertor_pos, dtype=tf.float32)

        detector_sensitiv = np.array(
            [0.47526155, 0.55967664, 0.6198833, 0.64954575, 0.81633771, 0.84956775, 0.91388105, 0.99515464,
             1., 0.96469796, 0.8814758, 0.81815959, 0.76336376, 0.61661687, 0.60172786, 0.73707267,
             0.07322642, 0.18513981, 0.31756323, 0.44122716, 0.57898053, 0.74086678, 0.85592147, 0.94426889,
             0.98750972, 0.9700378, 0.89189378, 0.7681564, 0.61199248, 0.49246729, 0.39245764, 0.36008477])
        detector_sensitiv_tf = tf.convert_to_tensor(detector_sensitiv, dtype=tf.float32)

        n = self.h * self.w * self.d

        inputs = layers.Input(shape=(32,))
        x = layers.Dense(n, activation="relu")(inputs)
        x = layers.Dense(n, activation="relu")(x)
        x = layers.Reshape(target_shape=(15, 15, 15))(x)  # 15, 15, 30
        x = layers.Conv2DTranspose(filters=self.d, kernel_size=self.k, strides=(2, 2), padding='same',
                                   activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(x)
        x = layers.Conv2DTranspose(filters=self.d, kernel_size=self.k, strides=(2, 2), padding='same',
                                   activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(x)

        pre_tomo_out = layers.Conv2D(filters=1, kernel_size=self.l, strides=(1, 1), padding='same',
                                     activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
                                     name='pre_tomo_out')(x)
        tomo_out = layers.Lambda(lambda z: z * divertor_pos_tf, name='tomo_out')(pre_tomo_out)

        flat_tomo_out = layers.Reshape(target_shape=(3600,))(tomo_out)
        scaled_flat_tomo = layers.Lambda(lambda z: 2.8e2 * z)(flat_tomo_out)
        pre_detect_out = layers.Lambda(lambda z: (3.3e6 / 1.0) * keras.backend.dot(z, geo_matrix_tf))(scaled_flat_tomo)
        detect_out = layers.Lambda(lambda z: z * detector_sensitiv_tf, name='detect_out')(pre_detect_out)
        # detect_out = layers.Multiply()._merge_function(inputs=[detector_sensitiv_tf, pre_detect_out])

        self.model = Model(inputs=inputs, outputs=[tomo_out, detect_out])
        self.model.load_weights('resources/cnn_weights.hdf5')

    def plot_test_detectors(self, test_idx):
        font = {'family': 'normal',
                'size': 20}
        matplotlib.rc('font', **font)

        plt.subplot(121)
        plt.bar(list(range(16)), self.test_detectors[test_idx, :16])
        plt.xlabel("#detector")
        plt.ylabel("Measurements (V)")
        plt.title("Top camera")

        plt.subplot(122)
        plt.bar(list(range(16)), self.test_detectors[test_idx, 16:])
        plt.xlabel("#detector")
        plt.ylabel("Measurements (V)")
        plt.title("Outer camera")
        plt.show()

    def reconstruct_test_detector(self, test_idx):
        pred_tomogram, pred_detect = self.model.predict(x=self.test_detectors[test_idx].reshape((1, 32)), steps=None)
        self.plot_reconstructs(input_detectors=self.test_detectors[test_idx], pred_tomogram=pred_tomogram, pred_detect=pred_detect)
        return pred_tomogram, pred_detect

    def reconstruct_profile(self, real_detectors):
        real_detectors = np.array(real_detectors).reshape((1,32))
        pred_tomogram, pred_detect = self.model.predict(x=real_detectors, steps=None)
        self.plot_reconstructs(input_detectors=real_detectors, pred_tomogram=pred_tomogram, pred_detect=pred_detect)
        return pred_tomogram, pred_detect

    @staticmethod
    def plot_reconstructs(input_detectors, pred_tomogram, pred_detect):
        font = {'family': 'normal',
                'size': 20}
        matplotlib.rc('font', **font)

        plt.subplot(131)
        ax = plt.gca()
        plt.bar(list(range(32)), np.ravel(input_detectors))
        plt.xlabel("#detector")
        plt.ylabel("Measurements (V)")

        plt.subplot(132)
        plt.imshow(pred_tomogram.reshape(60, 60), cmap=matplotlib.cm.get_cmap('plasma'))
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.xticks(np.linspace(0, 59, num=3),
                   np.round(np.arange(start=-0.1, stop=0.1 + 0.1, step=0.1), 2))
        plt.yticks(np.linspace(0, 59, num=5),
                   np.flipud(np.round(np.arange(start=-0.1, stop=0.1 + 0.02, step=0.05), 2)))
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Emissivity (a.u.)', rotation=90)

        plt.subplot(133)
        plt.bar(list(range(32)), pred_detect[0])
        ax = plt.gca()
        taetd = np.round(np.sum(np.abs(pred_detect - input_detectors)), 2)
        plt.text(0.4, 0.85, "TAETD=" + str(taetd) + "V", transform=ax.transAxes)
        plt.xlabel("#detector")
        plt.ylabel("Measurements (V)")

        plt.subplots_adjust(wspace=0.6)
        plt.suptitle("Tomographic reconstruction")
        plt.show()
