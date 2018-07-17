import numpy as np

def rand_Initialize_Weights(i_size,h_sizes,o_size,epsilon):
    #collect the sizes of the layers
    n=0
    m=i_size+1
    for i in range(len(h_sizes)):
        n+=m*h_sizes[i]
        m=h_sizes[i]+1
    n+=m*o_size
    #randomly initialize the weights
    thetas=np.random.random((n,1))
    thetas=2*epsilon*thetas-epsilon

    return thetas


def put_y_in_matrix(y,o_size):

    yy=np.zeros((o_size,len(y)))
    for i in range(len(y)):
        yy[y[i]-1,i]=1

    return yy

def unroll_list_matrices(Matrix_list):
    n=0
    for i in range(len(Matrix_list)):
        n+=np.size(Matrix_list[i])
    

    theta_grads=np.zeros((n,1))

    start=0
    for i in range(len(Matrix_list)):
        tt=Matrix_list[i]
        end=start+tt.size
        theta_grads[start:end]=tt.reshape(tt.size,1,order='F')
        start=end
    return theta_grads


def sigmoid(X):
    #compute the sigmoid function
    s=1.0/(1.0 + np.exp(-X))
    return s


def sigmoidGradient(z):
    #compute the gradient of the sigmoid function
    zz=np.exp(-z)
    g=np.true_divide(zz,np.square(1.0+zz))
    return g


def predict_values(Thetas, X):
    zz=np.ones((X.shape[0],X.shape[1]+1))
    zz[:,1:]=X
    h=sigmoid(zz.dot(Thetas[0].T))
    for i in range(1,len(Thetas)):
        zz=np.ones((h.shape[0],h.shape[1]+1))
        zz[:,1:]=h
        h=sigmoid(zz.dot(Thetas[i].T))
    
    p=np.argmax(h, axis=1)+1

    return p


def get_accuracy(p,y):
 
    return (p == y.flatten()).sum()/float(len(y))
