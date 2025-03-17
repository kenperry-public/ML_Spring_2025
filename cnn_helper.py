import matplotlib.pyplot as plt
import numpy as np

import time
import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

import pdb

class CNN_Helper():
    def __init__(self, **params):
        return

    def create_img(self):
        h, w = 8, 8
        img = np.zeros((h,w))
        img[2:-2,1] = 1
        img[-2,2:-2]= 1

        return img

        
    def create_filters(self):
        filt_horiz  = np.zeros( (3,3) )
        filt_horiz[0,:] = 1

        filt_vert = np.zeros( (3,3) )
        filt_vert[:,0] = 1

        filt_edge_h  = np.zeros( (3,3) )
        filt_edge_h[0,:] = -1
        filt_edge_h[1,:] =  1
        filt_edge_h[2,:] =  -1

        filt_edge_v  = filt_edge_h.T

        filt_edge_2 = np.zeros( (3,3))
        filt_edge_2[:, 0]  = 1
        filt_edge_2[-1, :] = 1

        filters = { "horiz, light to dark": filt_horiz,
                    "vert,  light to dark": filt_vert,
                    "horiz, light band":    filt_edge_h,
                    "vert, light band":     filt_edge_v,
                    "L"               :     filt_edge_2
                    }
            
        return filters

    def showmat(self,mat, ax, select=None, show_all=False):
        ax.matshow(mat, cmap="gray")

        if show_all:
            for row in range(0, mat.shape[0]):
                for col in range(0, mat.shape[1]):
                    ax.text(col, row, mat[row, col], color='black', backgroundcolor="white", ha='center', va='center')

        if select is not None:
            row_min, row_max, col_min, col_max = select
            for row in range(row_min, row_max):
                for col in range(col_min, col_max):
                    ax.text(col, row, mat[row, col], color='white', backgroundcolor="blue", ha='center', va='center')



    def pad(self, img, pad_size):
        padded_img = np.zeros( list(s+ 2*pad_size for s in img.shape) )
        padded_img[ pad_size:-pad_size, pad_size:-pad_size] = img
        return padded_img


    # Note: score[row,col] is result of applying filter CENTERED at img[row,col]
    def apply_filt_2d(self, img, filt):
        # Shape of the filter
        filt_rows, filt_cols = filt.shape

        # Pad the image
        pad_size = (filt_rows-1)// 2
        padded_img = self.pad(img, pad_size)

        score = np.zeros( img.shape )
        for row in range( img.shape[0] ):
            for col in range( img.shape[1] ):
                # window, centered on (row,col)of img
                # - is centered at (row+pad_size, col+pad_size) in padded_img
                # - so corners are (row+pad_size - pad_size, col+pad_size - pad_size)
                window = padded_img[ row:row+filt_rows, col:col+filt_cols]
                score[row,col] = (window *filt).sum()

        return score

    def plot_convs(self, img=None, filters=None):
        if filters is None:
            filters= self.create_filters()

        if img is None:
            img = self.create_img()
            
        fig, axs = plt.subplots( len(filters), 3, figsize=(12, 12 ),
                                 gridspec_kw={'width_ratios': [8, 1, 8]}
                                 )

        fig.subplots_adjust(hspace=10)

        i = 0
        for lab, filt in filters.items():

            img_filt = self.apply_filt_2d(img, filt)
            _= axs[i,0].matshow(img, cmap="gray")
            _= axs[i,0].xaxis.set_ticks_position('bottom')
            _= axs[i,0].set_title ("input")
            

            _= axs[i,1].matshow(filt, cmap="gray")
            _= axs[i,1].set_title ("filter:\n" + lab)
            _= axs[i,1].xaxis.set_ticks_position('bottom')
            
            _= axs[i,2].matshow(img_filt, cmap="gray")
            _= axs[i,2].xaxis.set_ticks_position('bottom')
            _= axs[i,2].set_title ("convolution output")
            
            i += 1

        fig.tight_layout()
        
        return fig, axs

class Charts_Helper():
    def __init__(self, save_dir="/tmp", visible=True, **params):
        """
        Class to produce charts (pre-compute rather than build on the fly) to include in notebook

        Parameters
        ----------
        save_dir: String.  Directory in which charts are created
        visible: Boolean.  Create charts but do/don't display immediately
        """
        self.X, self.y = None, None
        self.save_dir = save_dir

        self.visible = visible

        nnh = CNN_Helper()
        self.nnh = nnh

        return

    def create_receptive_field_1d_chart(self):
        nnh = self.nnh
        visible = self.visible
        

        # Set up the 1D input
        input_size = 28
        input_signal = np.random.rand(input_size)

        # Define the convolutional layers
        layers = [
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
        ]

        # Visualize the receptive field growth
        fig, axes = plt.subplots(2, 2, figsize=(10, 5))

        # Plot the input signal
        axes[0, 0].plot(input_signal)
        axes[0, 0].set_title('Input Signal')

        # Plot the receptive field for each layer
        for i, layer in enumerate(layers):
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            padding = layer['padding']

            # Calculate the receptive field size
            receptive_field_size = kernel_size + (kernel_size - 1) * (i)

            # Create a mask to highlight the receptive field
            mask = np.zeros_like(input_signal)
            mask[input_size // 2 - receptive_field_size // 2:input_size // 2 + receptive_field_size // 2 + 1] = 1

            # Plot the receptive field
            row = i // 2
            col = i % 2
            axes[row, col].plot(input_signal)
            axes[row, col].fill_between(np.arange(input_size), mask * input_signal.max(), alpha=0.5)
            axes[row, col].set_title(f'Layer {i + 1} Receptive Field: {receptive_field_size}')

        plt.tight_layout()
        
        if not visible:
            plt.close(fig)

        return fig, axes


    def create_receptive_field_2d_chart(self):
        nnh = self.nnh
        visible = self.visible

        # Set up the input image
        input_size = 28
        input_image = np.random.rand(input_size, input_size)

        # Define the convolutional layers
        layers = [
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'kernel_size': 3, 'stride': 1, 'padding': 1},
        ]

        # Visualize the receptive field growth
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the input image
        axes[0, 0].imshow(input_image, cmap='gray')
        axes[0, 0].set_title('Input Image')

        # Plot the receptive field for each layer
        for i, layer in enumerate(layers):
            kernel_size = layer['kernel_size']
            stride = layer['stride']
            padding = layer['padding']

            # Calculate the receptive field size
            receptive_field_size = kernel_size + (kernel_size - 1) * (i)

            # Create a mask to highlight the receptive field
            mask = np.zeros_like(input_image)
            mask[input_size // 2 - receptive_field_size // 2:input_size // 2 + receptive_field_size // 2 + 1,
                 input_size // 2 - receptive_field_size // 2:input_size // 2 + receptive_field_size // 2 + 1] = 1

            # Plot the receptive field
            row = i // 2
            col = i % 2
            axes[row, col].imshow(input_image, cmap='gray')
            axes[row, col].imshow(mask, cmap='Reds', alpha=0.5)
            axes[row, col].set_title(f'Layer {i + 1} Receptive Field: {receptive_field_size}x{receptive_field_size}')

        plt.tight_layout()

        if not visible:
            plt.close(fig)

        return fig, axes


    def create_charts(self):
        save_dir = self.save_dir

        print("Saving to directory: ", save_dir)
        
        print("Create receptive field 1d  chart")
        fig, ax = self.create_receptive_field_1d_chart()
        rfield_1d_file = os.path.join(save_dir, "receptive_field_1d.png")
        fig.savefig(rfield_1d_file)

        print("Create receptive field 2d  chart")
        fig, ax = self.create_receptive_field_2d_chart()
        rfield_2d_file = os.path.join(save_dir, "receptive_field_2d.png")
        fig.savefig(rfield_2d_file)

        print("Done")
        
        return { "receptive_field_1d": rfield_1d_file,
                 "receptive_field_2d": rfield_2d_file,
                 }
