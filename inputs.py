import scipy.io as sio
import numpy as np
import random
from helper_functions import rand_Initialize_Weights, put_y_in_matrix, unroll_list_matrices


m1=sio.loadmat('ex4weights.mat')
m2=sio.loadmat('ex4data1.mat')

#print(m1.get('Theta1').shape)
#Thetas=[m1.get('Theta1'), m1.get('Theta2')]
X=m2.get('X')
y=m2.get('y')
random.shuffle(y)
Lambda=1

#network-structure
input_layer_size=int(X.shape[1])
hiden_layer_sizes=[50]
output_layer_size=10

#initial_thetas=unroll_Gradients(Thetas)
epsilon_init=1
initial_thetas=rand_Initialize_Weights(input_layer_size,hiden_layer_sizes,output_layer_size,epsilon_init)
y_mat=put_y_in_matrix(y, output_layer_size)


