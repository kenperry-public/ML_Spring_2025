import time
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pdb

class NN_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        return

    def sigmoid(self, x):
        x = 1/(1+np.exp(-x))
        return x

    def sigmoid_grad(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2
    

    def plot_activations(self, x):
        sigm   = self.sigmoid(x)
        d_sigm = self.sigmoid_grad(x)
        d_tanh = 1 - np.tanh(x)**2
        d_relu = np.zeros_like(x) +  (x >= 0)

        fig, axs = plt.subplots(3,2, figsize=(16, 8))
        _ = axs[0,0].plot(x, sigm)
        _ = axs[0,0].set_title("sigmoid")
        _ = axs[0,0].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[0,1].plot(x, d_sigm)
        _ = axs[0,1].set_title("derivative sigmoid")
        _ = axs[0,1].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[1,0].plot(x, np.tanh(x))
        _ = axs[1,0].set_title("tanh")
        _ = axs[1,0].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[1,1].plot(x, d_tanh)
        _ = axs[1,1].set_title("derivative tanh")
        _ = axs[1,1].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[2,0].plot(x, np.maximum(0.0, x))
        _ = axs[2,0].set_title("ReLU")
        _ = axs[2,0].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
        
        _ = axs[2,1].plot(x, d_relu)
        _ = axs[2,1].set_title("derivative ReLU")
        _ = axs[2,1].set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)

        _ = fig.tight_layout()
        return fig, axs

    
    def NN(self, W,b):
        """
        Create a "neuron" z = ReLu( W*x + b )
        Returns dict
        - key "x": range of input values x
        - key "y": y = W*x + b
        - Key "z": z = max(0, y)
        """
        x = np.linspace(-100, 100, 100)
        z = W*x + b
        
        y = np.maximum(0, z)
        return { "x":x,
                 "y":y,
                 "W":W,
                 "b":b
                 }


    def plot_steps(self, xypairs):
        fig, ax = plt.subplots(1,1, figsize=(10,6))
        for pair in xypairs:
            x, y, W, b = [ pair[l] for l in ["x", "y", "W", "b" ] ]
            _ = ax.plot(x, y, label="{w:d}x + {b:3.2f}".format(w=W, b=b))
            
            _ = ax.legend()
            _ = ax.set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)
            #_ = ax.set_xlabel("x")
            _ = ax.set_ylabel("activation")
            _ = ax.set_title("Binary Switch creation")

        _ = fig.tight_layout()
        return fig, ax

    def step_fn_plot(self, visible=True):
        slope = 1000
        start_offset = 0

        start_step = self.NN(slope, -start_offset)

        end_offset = start_offset + .0001

        end_step = self.NN(slope,- end_offset)

        step= {"x": start_step["x"], 
               "y": start_step["y"] - end_step["y"],
               "W": slope,
               "b": 0
              }
        fig, ax = self.plot_steps( [  step ] )

        if not visible:
            plt.close(fig)

        return fig, ax
            
    def sigmoid_fn_plot(self, visible=True):
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        x =np.arange(-5,5, 0.1)
        sigm   = self.sigmoid(x)
        _ = ax.plot(x, sigm)
        _= ax.set_title("sigmoid")
        _= ax.set_xlabel("$y_{(l-1)} \cdot W_{l,j}$", fontsize=14)

        if not visible:
            plt.close(fig)

        return fig, ax
    def plot_loss_fns(self):
        # prod = y * s(x)
        # Postive means correctly classified; negative means incorrectly classified
        prod  = np.linspace(-1, +2, 100)

        # Error if product is negative
        error_acc  =  prod < 0
        error_exp  =  np.exp( -prod )

        # Error is 0 when product is exactly 1 (i.e., s(x) = y = 1)
        error_sq    =  (prod -1 )** 2

        # Error is negative of product
        # Error unless product greater than margin of 1
        error_hinge =  (- (prod -1) ) * (prod -1 < 0)

        fig, ax = plt.subplots(1,1, figsize=(10,6))
        _ = ax.plot(prod, error_acc, label="accuracy")
        _ = ax.plot(prod, error_hinge, label="hinge")
        
        # Truncate the plot to keep y-axis small and comparable across traces
        _ = ax.plot(prod[ prod > -0.5], error_exp[ prod > -0.5], label="exponential")
        
        _ = ax.plot(prod[ prod > -0.5], error_sq[ prod > -0.5], label="square")
        _ = ax.legend()
        _ = ax.set_xlabel("error")
        _ = ax.set_ylabel("loss")
        _ = ax.set_title("Loss functions")



    def plot_cosine_lr(self):
        num_batches= 1000
        epochs = np.linspace(0, num_batches, 100)/num_batches
        coss = np.cos( np.pi * epochs )
        rates = 0.5 * (1 + coss)

        fig, ax = plt.subplots(1,1, figsize=(10,4))
        _ = ax.plot(epochs, rates)
        _  = ax.set_xlabel("Epoch")
        _  = ax.set_ylabel("Fraction of original rate")
        _  = ax.set_title("Cosine Learning Rate schedule")

        return fig, ax

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

        nnh = NN_Helper()
        self.nnh = nnh

        return

    def create_activation_functions_chart(self):
        nnh = self.nnh
        visible = self.visible
        
        fig, axs = nnh.plot_activations( np.arange(-5,5, 0.1) )
        
        if not visible:
            plt.close(fig)

        return fig, axs

    def create_sequential_arch_chart(self, visible=None):
        if visible is None:
            visible = self.visible
        
        # Define rectangle properties
        rect_width = 0.5
        rect_height = 1.5  # Adjusted height for longer rectangles
        spacing = 1.2  # Adjusted spacing for longer rectangles

        # Create figure and axis
        fig, ax = plt.subplots()

        # Draw rectangles and arrows
        for i in range(5):
            rect = plt.Rectangle((i*spacing, 0), rect_width, rect_height, color='lightgrey', edgecolor='black')
            ax.add_patch(rect)

            if i < 4:
                ax.annotate('', xy=((i+1)*spacing, rect_height/2), xytext=(i*spacing + rect_width, rect_height/2),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

        # Set axis limits and labels
        ax.set_xlim(-0.5, 6)  # Adjusted limit for longer rectangles
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        if not visible:
            plt.close(fig)

        return fig, ax

    def create_functional_arch_chart(self, visible=None):
        if visible is None:
            visible = self.visible

        # Define rectangle properties
        rect_width = 0.5
        rect_height = 1.5  # Adjusted height for longer rectangles
        spacing = 1.2  # Adjusted spacing for longer rectangles

        # Create figure and axis
        fig, ax = plt.subplots()

        # Draw rectangles and arrows
        for i in range(5):
            rect = plt.Rectangle((i*spacing, 0), rect_width, rect_height, color='lightgrey', edgecolor='black')
            ax.add_patch(rect)

            if i < 4:
                ax.annotate('', xy=((i+1)*spacing, rect_height/2), xytext=(i*spacing + rect_width, rect_height/2),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

                if i == 1:
                    ax.annotate('', xy=(3*spacing + rect_width, rect_height), xytext=(1*spacing, rect_height),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5', color='black'))

        # Set axis limits and labels
        ax.set_xlim(-0.5, 6)  # Adjusted limit for longer rectangles
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        if not visible:
            plt.close(fig)

        return fig, ax


    def draw_surface(self, visible=None):
        if visible is None:
            visible = self.visible


        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 10x10 grid of points
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 10)
        X, Y = np.meshgrid(x, y)

        # Define a simple quadratic function for Z
        #Z = np.sin(np.sqrt(X**2 + Y**2))

        Z = np.zeros_like(X) + 2
        Z += 0.1 * np.sin(.25*X)*np.cos(.25*Y)

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')


        # Add a color bar which maps values to colors
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_zlim(1.75,2.25)

        # Label the axes
        ax.set_xlabel("$\mathbf{x}_1$", fontsize=18)
        ax.set_ylabel("$\mathbf{x}_2$", fontsize=18)
        ax.set_zlabel("$\mathbf{y}$", fontsize=18)
        
        if not visible:
            plt.close(fig)

        return fig, ax

    def add_shaded(self, ax, xmin, xmax, ymin, ymax):
        # Define the vertices of the shaded area polygon
        zmin = ax.get_zlim()[0]
        verts = [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)]

        # Create a Poly3DCollection and add it to the plot
        poly = Poly3DCollection([verts], alpha=0.5, facecolors='grey')
        ax.add_collection3d(poly)

    def draw_ne(self, visible=None):
        if visible is None:
            visible = self.visible

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # Set axis labels
        ax.set_xlabel(r'$W_1$', fontsize=14)
        ax.set_ylabel(r'$W_2$', fontsize=14)

        # Set axis limits
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 5])

        # Draw the arrow
        start = np.array([1, 1])  # Base of the arrow (W(t))
        end = np.array([3, 3])    # Head of the arrow (W(t+1))
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Draw the arrow
        ax.arrow(start[0], start[1], dx, dy, head_width=0.3, head_length=0.5, fc='k', ec='k')

        ax.text(start[0] - 0.3, start[1] - 0.3, r'$\mathbf{W}_{(t)}$', fontsize=12)
        ax.text(end[0] + 0.3, end[1] + 0.5, r'$\mathbf{W}_{(t+1)}$', fontsize=12)

        # Show the plot
        plt.show()

        if not visible:
                    plt.close(fig)


        return fig, ax


    def draw_sw(self, visible=None):
        if visible is None:
            visible = self.visible

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # Set axis labels
        ax.set_xlabel(r'$W_1$', fontsize=14)
        ax.set_ylabel(r'$W_2$', fontsize=14)

        # Set axis limits
        ax.set_xlim([-1, 5])
        ax.set_ylim([-1, 5])

        # Draw the arrow
        end = np.array([1, 1])  # Base of the arrow (W(t))
        start = np.array([3, 3])    # Head of the arrow (W(t+1))
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Draw the arrow
        ax.arrow(start[0], start[1], dx, dy, head_width=0.3, head_length=0.5, fc='k', ec='k')

        # Add labels for the base and head of the arrow
        ax.text(end[0] - 0.5, end[1] - 0.8, r'$\mathbf{W}_{(t+1)}$', fontsize=12)
        ax.text(start[0] + 0.3, start[1] + 0.3, r'$\mathbf{W}_{(t)}$', fontsize=12)


        # Show the plot
        plt.show()

        if not visible:
            plt.close(fig)


        return fig, ax

    def draw_se(self, visible=None):
        if visible is None:
            visible = self.visible

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))

        # Set axis labels
        ax.set_xlabel(r'$W_1$', fontsize=14)
        ax.set_ylabel(r'$W_2$', fontsize=14)

        # Set axis limits
        ax.set_xlim([0, 4])
        ax.set_ylim([-1, 5])

        # Draw the arrow
        start = np.array([1, 1])  # Base of the arrow (W(t))
        end = np.array([3, 3])    # Head of the arrow (W(t+1))

        # From start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        ax.arrow(start[0], start[1], dx, dy, head_width=0.2, head_length=0.1, fc='k', ec='k')

        ax.text(start[0] - 0.3, start[1] - 0.3, r'$\mathbf{W}_{(t)}$', fontsize=12)
        ax.text(end[0] + 0.3, end[1] + 0.5, r'$\mathbf{W}_{(t+1)}$', fontsize=12)

        end2 = np.array([1.75, 0])


        # From  start to end2
        # allow extra space so the 2 arrow heads ending at end2 don't overlap
        extra_space = .15
        dx2, dy2 = end2[0] - start[0] - extra_space , end2[1] - start[1] - extra_space
        ax.arrow(start[0], start[1], dx2, dy2 , head_width=0.2, head_length=0.2, fc='k', ec='blue',
                 linestyle='dotted', linewidth=3)

        # From end to end2
        dx3, dy3 = end2[0] - end[0], end2[1] -end[1]
        ax.arrow(end[0], end[1], dx3, dy3, head_width=0.2, head_length=0.1, fc='k', ec='k')


        # Show the plot
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax


    def create_sigmoid_charts(self, visible=None):
        if visible is None:
            visible = self.visible

        # Define the sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Calculate the derivative of the sigmoid function
        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))

        # Calculate the second derivative of the sigmoid function
        def sigmoid_second_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x)) * (1 - 2 * sigmoid(x))

        # Create the x-axis values
        x = np.linspace(-5, 5, 100)

        # Calculate the sigmoid function values
        y = sigmoid(x)

        # Calculate the derivative values
        y_derivative = sigmoid_derivative(x)

        # Calculate the second derivative values
        y_second_derivative = sigmoid_second_derivative(x)

        # Plot the sigmoid function and its derivative
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-', linewidth=2, label='Sigmoid Function')
        ax.plot(x, y_derivative, 'r-', linewidth=2, label='Derivative')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper left')

        # Shade the regions where the absolute value of the second derivative is less than 0.02
        ax.fill_between(x, 1, where=np.abs(y_second_derivative) < 0.02, color='lightgray', alpha=0.5)

        # Set the plot title and axis labels
        ax.set_title('Sigmoid Function and its Derivative')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Show the plot
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax

    def draw_layer(visible=False):
    # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the figure height to 80% of the diagram height
        fig_height = 0.8
        fig_width = 8
        ax.set_position([ax.get_position().x0, 
                        (1 - fig_height) / 2, 
                        ax.get_position().width, 
                        fig_height])

        # Draw the vertical rectangle
        rect_width = 0.2
        rect_height = fig_height
        rect_x = 0.4
        rect_y = (1 - fig_height) / 2
        rect_linewidth = 2
        ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='black', linewidth=rect_linewidth))

        # Draw the 5 circles
        circle_radius = 0.06
        circle_spacing = 0.12
        circle_x = rect_x + rect_width / 2
        circle_y = np.linspace(rect_y + circle_radius + circle_spacing/2, 
                              rect_y + rect_height - circle_radius - circle_spacing/2, 5)
        circle_linewidth = 2

        for y in circle_y:
            ax.add_artist(plt.Circle((circle_x, y), circle_radius, fill=False, color='blue', linewidth=circle_linewidth))

        # Set the axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Remove the axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the figure
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax

    def draw_layer_select(self, visible=False):
        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the figure height to 80% of the diagram height
        fig_height = 0.8
        fig_width = 8
        ax.set_position([ax.get_position().x0, 
                        (1 - fig_height) / 2, 
                        ax.get_position().width, 
                        fig_height])

        # Draw the vertical rectangle
        rect_width = 0.2
        rect_height = fig_height
        rect_x = 0.4
        rect_y = (1 - fig_height) / 2
        rect_linewidth = 2
        ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='black', linewidth=rect_linewidth))

        # Draw the 5 circles
        circle_radius = 0.06
        circle_spacing = 0.12
        circle_x = rect_x + rect_width / 2
        circle_y = np.linspace(rect_y + circle_radius + circle_spacing/2, 
                              rect_y + rect_height - circle_radius - circle_spacing/2, 5)
        circle_linewidth = 2

        for i, y in enumerate(circle_y):
            if i == 3:  # Shade the second circle from the top
                ax.add_artist(plt.Circle((circle_x, y), circle_radius, fill=True, color='blue', linewidth=circle_linewidth))
            else:
                ax.add_artist(plt.Circle((circle_x, y), circle_radius, fill=False, color='blue', linewidth=circle_linewidth))

        # Set the axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Remove the axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the figure
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax

    def draw_layer_with_2d_elements(self, visible=False):
        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the figure height to 80% of the diagram height
        fig_height = 0.9
        fig_width = 8
        ax.set_position([ax.get_position().x0, 
                        (1 - fig_height) / 2, 
                        ax.get_position().width, 
                        fig_height])

        # Draw the vertical rectangle
        rect_width = 0.2
        rect_height = fig_height
        rect_x = 0.4
        rect_y = (1 - fig_height) / 2
        rect_linewidth = 2
        ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='black', linewidth=rect_linewidth))

        # Draw the 5 rectangles with internal lines and thicker borders
        rect_size = 0.12
        rect_spacing = 0.04
        rect_x = rect_x + rect_width / 2 - rect_size / 2
        rect_y_start = rect_y + rect_height / 10
        rect_y_step = (rect_height - 2 * rect_y_start) / 5
        rect_linewidth = 2

        for rect_idx in range(5):
            rect_y = rect_y_start + rect_idx * (rect_size + rect_spacing)
            ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_size, rect_size, fill=False, color='blue', linewidth=rect_linewidth))
            ax.add_line(plt.Line2D([rect_x, rect_x + rect_size], [rect_y + rect_size / 2, rect_y + rect_size / 2], color='black', linewidth=rect_linewidth))
            ax.add_line(plt.Line2D([rect_x + rect_size / 2, rect_x + rect_size / 2], [rect_y, rect_y + rect_size], color='black', linewidth=rect_linewidth))

        # Set the axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Remove the axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the figure
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax

    def draw_layer_with_2d_elements_select(self, visible=False):
        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the figure height to 80% of the diagram height
        fig_height = 0.8
        fig_width = 8
        ax.set_position([ax.get_position().x0, 
                        (1 - fig_height) / 2, 
                        ax.get_position().width, 
                        fig_height])

        # Draw the vertical rectangle
        rect_width = 0.2
        rect_height = fig_height
        rect_x = 0.4
        rect_y = (1 - fig_height) / 2
        rect_linewidth = 2
        ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='black', linewidth=rect_linewidth))

        # Draw the 5 squares
        square_size = 0.12
        square_spacing = 0.12
        square_x = rect_x + rect_width / 2 - square_size / 2
        square_y = np.linspace(rect_y + square_size / 2 + square_spacing/2, 
                              rect_y + rect_height - square_size / 2 - square_spacing/2, 5)
        square_linewidth = 2

        for i, y in enumerate(square_y):
            if i == 3:  # Shade the second square from the top
                ax.add_patch(plt.Rectangle((square_x, y - square_size / 2), square_size, square_size, fill=True, color='blue', linewidth=square_linewidth))
            else:
                ax.add_patch(plt.Rectangle((square_x, y - square_size / 2), square_size, square_size, fill=False, color='blue', linewidth=square_linewidth))

        # Set the axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Remove the axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the figure
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax

    def draw_layer_with_2d_elements_pool(self, visible=False):
        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the figure height to 80% of the diagram height
        fig_height = 0.8
        fig_width = 8
        ax.set_position([ax.get_position().x0, 
                        (1 - fig_height) / 2, 
                        ax.get_position().width, 
                        fig_height])

        # Draw the vertical rectangle
        rect_width = 0.2
        rect_height = fig_height
        rect_x = 0.4
        rect_y = (1 - fig_height) / 2
        rect_linewidth = 2
        ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color='black', linewidth=rect_linewidth))

        # Draw the 5 circles (twice as big)
        circle_radius = 0.02
        circle_spacing = 0.12
        circle_x = rect_x + rect_width / 2
        circle_y = np.linspace(rect_y + circle_radius + circle_spacing/2, 
                              rect_y + rect_height - circle_radius - circle_spacing/2, 5)
        circle_linewidth = 2

        for i, y in enumerate(circle_y):
            if i == 3:  # Shade the second circle from the top
                ax.add_artist(plt.Circle((circle_x, y), circle_radius, fill=True, color='blue', linewidth=circle_linewidth))
            else:
                ax.add_artist(plt.Circle((circle_x, y), circle_radius, fill=False, color='blue', linewidth=circle_linewidth))

        # Set the axis limits and aspect ratio
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Remove the axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the figure
        plt.show()

        if not visible:
            plt.close(fig)

        return fig, ax



    def create_charts(self):
        def create_and_save( method, fname ):
            # Invoke method to draw figure, save it in file, add entry to file_dict dictionary
             fig, ax = method()
             out_fname = os.path.join(save_dir, fname + ".png")
             fig.savefig(out_fname)

             file_dict[fname] = out_fname
             
             return out_fname
            
        save_dir = self.save_dir

        print("Saving to directory: ", save_dir)
        
        print("Create Activation function chart")
        fig, ax = self.create_activation_functions_chart()
        act_func_file = os.path.join(save_dir, "activation_functions.png")
        fig.savefig(act_func_file)

        fig, ax = self.create_sequential_arch_chart()
        seq_arch_file =  os.path.join(save_dir, "tf_sequential_arch.png")
        fig.savefig(seq_arch_file)

        fig, ax = self.create_functional_arch_chart()
        func_arch_file =  os.path.join(save_dir, "tf_functional_arch.png")
        fig.savefig(func_arch_file)

        fig, ax = self.draw_surface()
        surface_chart_file_0 = os.path.join(save_dir, "surface_chart_0.png")
        fig.savefig(surface_chart_file_0)

        fig, ax = self.draw_surface()
        _= self.add_shaded(ax, 2, 8, 2, 3)
        surface_chart_file_1 = os.path.join(save_dir, "surface_chart_1.png")
        fig.savefig(surface_chart_file_1)

        fig, ax = self.draw_surface()
        _= self.add_shaded(ax, 8, 9, 2, 8)
        surface_chart_file_2 = os.path.join(save_dir, "surface_chart_2.png")
        fig.savefig(surface_chart_file_2)

        fig, ax = self.draw_ne()
        grad_updt_ne_file = os.path.join(save_dir, "grad_updt_ne.png")
        fig.savefig(grad_updt_ne_file)

        fig, ax = self.draw_sw()
        grad_updt_sw_file = os.path.join(save_dir, "grad_updt_sw.png")
        fig.savefig(grad_updt_sw_file)

        fig, ax = self.draw_se()
        grad_updt_se_file = os.path.join(save_dir, "grad_updt_se.png")
        fig.savefig(grad_updt_se_file)

        fig, ax = self. create_sigmoid_charts()
        sigmoid_file = os.path.join(save_dir, "sigmoid_chart.png")
        fig.savefig(sigmoid_file)

        file_dict = {
            "activation functions": act_func_file,
            "TF Sequential arch" : seq_arch_file,
            "TF Function arch"   : func_arch_file,
            "surfaces": [ surface_chart_file_0, surface_chart_file_1, surface_chart_file_2 ],
            "gradient update NE": grad_updt_ne_file,
            "gradient update SW": grad_updt_sw_file,
            "gradient update SE": grad_updt_se_file,
            "sigmoid charts": sigmoid_file
            }
        

        # NEWER: use create_and_save convenience function to avoid repeated code for each diagram that we see above
        _ = create_and_save(self.draw_layer, "layer")

        _ = create_and_save(self.draw_layer_select, "layer_select")
        
        _=  create_and_save(self.draw_layer_with_2d_elements, "layer_w_2d_elements")

        _ = create_and_save(self.draw_layer_with_2d_elements_select, "layer_w_2d_elements_select")

        _ = create_and_save(self.draw_layer_with_2d_elements_pool, "layer_w_2d_elements_pool")
        
        print("Done")
        
        return file_dict

