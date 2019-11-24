import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
import ffmpeg

import keras
from keras import layers, models
from keras.models import Model
import tensorflow as tf

from sdas.core.client.SDASClient import SDASClient
from sdas.core.SDAStime import Date, Time, TimeStamp


class NeurISTTOK:

    def __init__(self):
        self.model = models.Sequential()
        self.h = 15
        self.w = 15
        self.d = 15
        self.k = 5
        self.l = 1

        # load relevant resources

        self.test_detectors = np.load("resources/testdetectors.npy")

        geo_matrix_np = np.load('resources/geo_mat60-newpin-ref-halfmm-px5000.npy')
        geo_matrix_tf = tf.transpose(tf.convert_to_tensor(geo_matrix_np, dtype=tf.float32))

        divertor_pos_seq = np.load('resources/divertorpos.npy')  # null emissivity after divertor
        divertor_pos = divertor_pos_seq.reshape(60, 60, 1)
        divertor_pos_tf = tf.convert_to_tensor(divertor_pos, dtype=tf.float32)

        detector_sensitiv = np.array(
            [0.47526155, 0.55967664, 0.6198833, 0.64954575, 0.81633771, 0.84956775, 0.91388105, 0.99515464,
             1., 0.96469796, 0.8814758, 0.81815959, 0.76336376, 0.61661687, 0.60172786, 0.73707267,
             0.07322642, 0.18513981, 0.31756323, 0.44122716, 0.57898053, 0.74086678, 0.85592147, 0.94426889,
             0.98750972, 0.9700378, 0.89189378, 0.7681564, 0.61199248, 0.49246729, 0.39245764, 0.36008477])
        detector_sensitiv_tf = tf.convert_to_tensor(detector_sensitiv, dtype=tf.float32)

        # neural network model

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

    def reconstruct_profile(self, real_detectors, plot=True):
        real_detectors = np.array(real_detectors).reshape((-1, 32))

        for idx in range(real_detectors.shape[0]):  # signal renormalization (also accounts for expected difference arising from different pinhole diameters)
            line = real_detectors[idx, :]
            top_sum = np.sum(line[:16])
            out_sum = np.sum(line[16:])
            line[16:] *= top_sum/out_sum
            line[16:] = line[16:]/np.array([2.617288080027149, 2.5232295415074155, 2.5561361125486584, 2.550394456494485,
                                            2.561285511054761, 2.673865476728384, 2.52714029717696, 2.575443760947923,
                                            2.5801198707348556, 2.555374296650839, 2.5661537557087817, 2.560182429249538,
                                            2.5183388085023664, 2.5186546200326667, 2.6248047782140977, 2.564414319267854])

        pred_tomogram, pred_detect = self.model.predict(x=real_detectors, steps=None)
        if plot:
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

    def load_sdas_data(self, channelID, shotnr):
        host = 'baco.ipfn.ist.utl.pt';
        port = 8888;
        self.client = SDASClient(host, port);
        dataStruct = self.client.getData(channelID, '0x0000', shotnr);
        dataArray = dataStruct[0].getData();
        len_d = len(dataArray);
        tstart = dataStruct[0].getTStart();
        tend = dataStruct[0].getTEnd();
        tbs = (tend.getTimeInMicros() - tstart.getTimeInMicros())*1.0/len_d;
        events = dataStruct[0].get('events')[0];
        tevent = TimeStamp(tstamp=events.get('tstamp'));
        delay = tstart.getTimeInMicros() - tevent.getTimeInMicros();
        timeVector = np.linspace(delay,delay+tbs*(len_d-1),len_d);
        return [dataArray, timeVector]

    def profile_evolv(self, shot_nr, init_time, end_time, video_name):
        data_top = []
        data_outer = []
        time_top, time_outer = [], []

        # list of acquisition channels for tomography (from detector 0 to 15)

        # for MARTE_NODE_IVO3.DataCollection.Channel_:
        # top_channel_order = ["182", "181", "184", "179", "178", "185", "183", "180", "004", "005", "003", "007", "000", "001", "002", "006"]
        # out_channel_order = ["190", "189", "192", "187", "186", "193", "191", "188", "012", "013", "011", "015", "008", "009", "010", "014"]

        # for PCIE_ATCA_ADC_16.BOARD_1.CHANNEL_:
        top_channel_order = ["013", "012", "015", "010", "009", "016", "014", "011", "005", "006", "004", "008", "001", "002", "003", "007"]
        out_channel_order = ["029", "028", "031", "026", "025", "032", "030", "027", "021", "022", "020", "024", "017", "018", "019", "023"]

        # downloading of tomography signal from database using sdas

        for top_channel_nr, out_channel_nr in zip(top_channel_order, out_channel_order):
            print("Downloading channel data...")
            # top_channelID = 'MARTE_NODE_IVO3.DataCollection.Channel_' + top_channel_nr
            # out_channelID = 'MARTE_NODE_IVO3.DataCollection.Channel_' + out_channel_nr
            top_channelID = 'PCIE_ATCA_ADC_16.BOARD_1.CHANNEL_' + top_channel_nr
            out_channelID = 'PCIE_ATCA_ADC_16.BOARD_1.CHANNEL_' + out_channel_nr
            add_data_top, add_time_top = self.load_sdas_data(channelID=top_channelID, shotnr=shot_nr)
            add_data_out, add_time_out = self.load_sdas_data(channelID=out_channelID, shotnr=shot_nr)

            data_top.append(add_data_top)
            data_outer.append(add_data_out)
            time_top.append(add_time_top)
            time_outer.append(add_time_out)

        # downloading of centroid position and plasma current data

        x_data, x_time = self.load_sdas_data(channelID='MARTE_NODE_IVO3.DataCollection.Channel_101', shotnr=shot_nr)
        z_data, z_time = self.load_sdas_data(channelID='MARTE_NODE_IVO3.DataCollection.Channel_102', shotnr=shot_nr)
        current_data, current_time = self.load_sdas_data(channelID='MARTE_NODE_IVO3.DataCollection.Channel_100', shotnr=shot_nr)

        top_signals = np.array(data_top)
        out_signals = np.array(data_outer)
        x_data = np.array(x_data)
        z_data = np.array(z_data)
        top_time = np.array(time_top[0])
        out_time = np.array(time_outer[0])

        # computation of DC components for the tomography signals

        dc_index = np.where(top_time > 2000)[0][0]
        top_dc, outer_dc = [], []
        for i in range(16):
            top_dc.append(np.mean(top_signals[i][:dc_index]))
            outer_dc.append(np.mean(out_signals[i][:dc_index]))
        top_dc = np.array(top_dc)
        outer_dc = np.array(outer_dc)

        # selection of desired time window for tomography signals

        top_idx = ( (top_time>init_time) & (top_time<end_time) )
        out_idx = ( (out_time>init_time) & (out_time<end_time) )

        top_time = top_time[top_idx]
        top_signals = top_signals[:, top_idx]
        out_signals = out_signals[:, out_idx]

        # down sample acquisition

        down_sample = list(range(0, top_signals.shape[1], 200))
        top_time = top_time[down_sample]
        max_len = top_signals.shape[1] - top_signals.shape[1]%200
        top_signals = top_signals[:, :max_len].reshape((16, -1, 200)).mean(-1)
        out_signals = out_signals[:, :max_len].reshape((16, -1, 200)).mean(-1)
        top_time = top_time[:top_signals.shape[1]]

        # selection of desired time window for current and centroid position, and handy renormalizations

        current_idx = ((current_time > init_time) & (current_time < end_time))

        x_data = x_data[current_idx]*300.0 + 30.0
        z_data = -z_data[current_idx]*300.0 + 30.0
        current_data = current_data[current_idx]/1000.0

        # subtraction of DC components from the tomography signals

        top_signals = top_signals.transpose() - top_dc
        out_signals = out_signals.transpose() - outer_dc

        # network forward propagation to compute predicted tomograms

        out_data = np.hstack((top_time.reshape((top_time.shape[0], 1)), top_signals, out_signals))
        tomo, det = self.reconstruct_profile(real_detectors=out_data[:, 1:], plot=False)

        # generation of video frames

        images = []
        f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [7, 1]})
        plt.suptitle("shot #" + str(shot_nr), fontsize=17)
        # plt_bar, = ax2.bar([1], [1])
        ax2.set_ylim(-6.1, 6.1)
        ax2.set_title("Current (kA)", fontsize=12)
        ax2.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        for idx, image in enumerate(tomo):
            plt_im = ax1.imshow(image.reshape((60, 60)), cmap=matplotlib.cm.get_cmap('plasma'), vmin=0, vmax=1, animated=True)
            plt_centr = ax1.scatter(x_data[idx], z_data[idx], s=100, c='red', marker='o')
            plt_txt = ax1.text(1, 5, str(round(out_data[idx, 0] / 1000, 3)) + " ms", color='white', fontsize=20, animated=True)
            if current_data[idx] >= 0:
                plt_bar, = ax2.bar([1], [current_data[idx]], color='r')
            else:
                plt_bar, = ax2.bar([1], [current_data[idx]], color='b')

            images.append([plt_im, plt_txt, plt_centr, plt_bar])

        # video conversion to .mp4, save and show

        ani = anim.ArtistAnimation(f, images, interval=35, blit=True, repeat_delay=1000)
        writer = anim.FFMpegWriter(fps=30, codec="h264")

        ani.save(video_name + ".mp4", writer=writer)
        plt.show()

    def reconstruct_lamp(self, lamp_radius, lamp_angle):
        real_data = []
        with open("resources/legless_july_data.csv") as f:
            content = f.readlines()
        for single_line in content[1:]:
            real_data.append(single_line.split(', ')[:-1])

        top_pixels = np.zeros(16)
        out_pixels = np.zeros(16)
        count = 0
        for line in real_data:
            if int(line[1]) == int(lamp_radius * 100) and float(line[2]) == lamp_angle:
                top_pixels += np.array([float(number) for number in line[3:19]])
                out_pixels += np.array([float(number) for number in line[19:35]])
                count += 1
            elif count != 0:
                break
        if count == 0:
            print("Requested radius and angle values not found.")
        else:
            integ_time = 0.007
            top_pixels = top_pixels / (count * integ_time)
            out_pixels = out_pixels / (count * integ_time)
            self.reconstruct_profile(real_detectors=np.concatenate((top_pixels, out_pixels)))