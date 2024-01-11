import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm

class EarlyStopping():

    def __init__(self):

        self.patience = 10
        self.patience_counter = 0
        self.threshold = 1e-5
        self.best_val_loss = np.inf

    def call(self, val_loss):

        if (self.best_val_loss - val_loss) > self.threshold:
            self.patience_counter = 0
            self.best_val_loss = val_loss
        else:
            self.patience_counter += 1

        if self.patience_counter == self.patience:
            return True

def check_data(root_path):
    for file in os.listdir(root_path):
        feat = np.load(os.path.join(root_path, file))
        print(feat.shape)
        print("MFCC mean: {}, MFCC std: {}".format(np.mean(feat), np.std(feat)))

def plot_mfcc(file):
    mfcc_data =np.load(file)
    fig, ax = plt.subplots()
    #mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    plt.savefig('mfcc.png')

if __name__ == "__main__":
    check_data('/data2/yun/aai_expr/Preprocessed_data_HY/F01/ema_final')
    #plot_mfcc('/data2/yun/aai_expr/Preprocessed_data_HY/F01/mfcc_final/F01_B10_S06_R01_N.npy')

