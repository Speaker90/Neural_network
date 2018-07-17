import numpy as np
from helper_functions import unroll_list_matrices, sigmoid, sigmoidGradient

def compute_cost(thetas,X,y,Lambda,i_size, h_sizes, o_size):
   
    #run the feed-forward algorithm for the activation units A and the intermediate values Z
    [A,Z, Thetas]=feed_forward(X,thetas,i_size,h_sizes,o_size)
    
    #run the backpropagation algorithm to get the cost functions and the gradients
    J=cost_function(X,Thetas,y,Lambda,A,Z)   
 
    print(J)
    
    return J

  
def compute_grad(thetas,X,y,Lambda,i_size, h_sizes, o_size):
   
    #run the feed-forward algorithm for the activation units A and the intermediate values Z
    [A,Z, Thetas]=feed_forward(X,thetas,i_size,h_sizes,o_size)
    
    #run the backpropagation algorithm to get the cost functions and the gradients
    Theta_Grads=back_propagation_grad(X,Thetas,y,Lambda,A,Z)

      
    return Theta_Grads


def feed_forward(X,thetas,i_size,h_sizes,o_size):
    Thetas=build_Thetas(thetas,i_size,h_sizes,o_size)
    A=[]
    Z=[]
    a1=np.ones((X.shape[0],X.shape[1]+1))
    a1[:,1:]=X
    a1=np.transpose(a1)
    A.append(a1)
    for i in range(len(Thetas)-1):
        theta=Thetas[i]
        z=np.dot(theta,A[-1])
        Z.append(z)
        a=np.vstack((np.ones((1,z.shape[1])),sigmoid(z)))
        A.append(a)
    z_last=np.dot(Thetas[-1],A[-1])
    #Z.append(z_last)
    A.append(sigmoid(z_last))
    return A, Z, Thetas


def cost_function(X,Thetas,yy,la,A,Z):

    #initialize the cost function and the gradients
    J=0
    #get the    ber ouf ouptput units
    m=yy.shape[1]
    yy=np.reshape(yy,(1,yy.size),order='F')
    #calculate the delta for the output units
    h=A[-1]
    h=np.reshape(h,(h.size,1),order='F')

    #get the unregularized cost function
    J+=1.0/m*(-yy.dot(np.log(h))-(1-yy).dot(np.log(1-h)))
    J=addRegularization_cost(J,m,la,Thetas)
 
    return J.flatten()


def back_propagation_grad(X,Thetas,yy,la,A,Z):

    #initialize the cost function and the gradients
    Theta_Grads=[0]*len(Thetas)
    #initalize the deltas
    Deltas=[None]*len(Thetas)
    #get the    ber ouf ouptput units
    m=yy.shape[1]
    #calculate the delta for the output units
    h=A[-1]
  
    #h=np.reshape(h,(h.size,1),order='F')
    Deltas[-1]=np.subtract(h,yy)

    hh=A[-2]

    

    Theta_Grads[-1]=np.dot(Deltas[-1],hh.T)

       
      
        #loop over all layers
    for d in range(len(Deltas)-2,-1,-1):
        
        #get the deltas and the thetas for all layers
        theta=Thetas[d+1][:,1:]
        hh=A[d]
        z=Z[d]
        Deltas[d]=np.multiply(np.dot(np.transpose(theta),Deltas[d+1]),sigmoidGradient(z))
        Theta_Grads[d]=Theta_Grads[d]+np.dot(Deltas[d],hh.T)

    Theta_Grads=addRegularization_grad(Theta_Grads,m,la,Thetas)
    Theta_Grads=unroll_list_matrices(Theta_Grads)
    
    return Theta_Grads.flatten()


def addRegularization_cost(J,m,la,Thetas):
    
    # calculate regularized gradients and cost
    for j in range(len(Thetas)):        
        #square the thetas for the cost function
        ts=np.square(Thetas[j])
        J=J+la/(2.0*m)*np.sum(ts[:,1:])
    
    return J


def addRegularization_grad(Theta_Grads, m,la,Thetas):
    
    # calculate regularized gradients and cost
    for j in range(len(Theta_Grads)):
        regularization=Thetas[j]
        #do not regularize first term
        regularization[:,0]=0.0
        regularization=(la/float(m)) * regularization
        Theta_Grads[j]=Theta_Grads[j]/m+regularization
    
    return Theta_Grads


def build_Thetas(thetas, i_size, h_sizes, o_size):
    Thetas=[]
    ss=h_sizes[:]
    ss.insert(0, i_size)
    ss.append(o_size)
    start=0
    
    for i in range(len(ss)-1):
        end=start+ss[i+1]*(ss[i]+1)
        tt=thetas[start:end]
        theta=tt.reshape(ss[i+1], ss[i]+1, order='F')
        Thetas.append(theta)
        start=end
    return Thetas


