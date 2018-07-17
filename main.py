from inputs import X, initial_thetas, y, y_mat, Lambda, input_layer_size, hiden_layer_sizes, output_layer_size
from scipy.optimize import fmin_l_bfgs_b
from cost_functions import compute_cost, compute_grad, build_Thetas
from helper_functions import predict_values, get_accuracy




def minimize_cost(thetas,X,y_mat,Lambda,input_layer_size, hiden_layer_sizes, output_layer_size):

    return compute_cost(thetas,X,y_mat,Lambda,input_layer_size, hiden_layer_sizes, output_layer_size)

def minimize_grad(thetas,X,y_mat,Lambda,input_layer_size, hiden_layer_sizes, output_layer_size):

    return compute_grad(thetas,X,y_mat,Lambda,input_layer_size, hiden_layer_sizes, output_layer_size)


Nfeval = 1

def callbackF(thetas):
    global Nfeval
    print '{0:4d}  '.format(Nfeval)
    Nfeval += 1


[thetas_trained, cost,_]=fmin_l_bfgs_b(minimize_cost, initial_thetas,  maxiter=400, fprime=minimize_grad, args=(X,y_mat,Lambda,input_layer_size, hiden_layer_sizes, output_layer_size))


Thetas_trained=build_Thetas(thetas_trained,input_layer_size, hiden_layer_sizes, output_layer_size)

p=predict_values(Thetas_trained,X)
acc=get_accuracy(p,y)
print(acc)
#compute_cost(initial_thetas,X,y,la,input_layer_size, hiden_layer_sizes, output_layer_size)