# BOLTZMANN MACHINE

# Importing Necessary Libraries

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data

# Importing the Dataset

movies = pd.read_csv('ml-1m/movies.dat',sep='::',header = None,engine='python' ,encoding='latin-1', names=['ID','Movie Name(Year)','Genre'] )
users = pd.read_csv('ml-1m/users.dat',sep='::',header = None,engine='python' ,encoding='latin-1', names=['ID','Gender','Genre'] )
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::',header = None,engine='python' ,encoding='latin-1' )

# Preparing the Training Set and the Test Set

# Note: delimiter for \t and sep for '::',',' and so on

training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')

# To Array
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')

# To Array
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies

#Type casting is encouraged

nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns

def convert(data):
    new_data = []
    for id_users in range(1,nb_users+1): # Index from 1 to nb_users
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies -1] = id_ratings #For Index to start at zero
        new_data.append(list(ratings)) #Torch expects List of Lists
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch Tensors

training_set = torch.FloatTensor(training_set) #FloatTensor expects a List of Lists
test_set = torch.FloatTensor(test_set)

# Converting the Ratings into Binary ratings 1 (Liked) or 0 (Not Liked)
# For both Training and Test sets

training_set[training_set == 0] = -1  # Replacing Originally Zero values to -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

class RBM():
    # nv - number of Visible nodes
    # nh - number of Hidden Nodes
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        # Probabilities of Visible Nodes given Hidden Nodes
        # randn - Standard Normal Distribution
        # Size is nh * nv
        
        # Tensor functions expect 2Dimensions. 1st is Batch size
        self.a = torch.randn(1, nh)
        # Bias for Hidden Nodes
        self.b = torch.randn(1, nv)
        # Bias for Visisble Nides
        
        # Hidden Node: W_trans X + a
        # Visible Node: W_trans X + b
        
    # Sampling the hidden nodes according to Probablities P[h/v]: Sigmoid Activation Function
    # x - corresponds to Visible Node v
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)  # Expand as: Shape of wx
        p_h_given_v = torch.sigmoid(activation)
        
        # Bernoulli Samples
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    # Sampling the visible nodes according to Probablities P[v/h]: Sigmoid Activation Function
    # y - corresponds to Hidden Node v
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)  # Expand as: Shape of wx
        p_v_given_h = torch.sigmoid(activation)
        
        # Bernoulli Samples
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # Training through Contrastive Divergence v0 -> h0 -> v1 -> ... -> h(k-1) -> vk
    # v0 - Input vector consisting of all ratings of 1 user
    # vk - Visible node obtained after k-steps in ContrastiveDivergence Algorithm 
    # ph0- Vector of Probabilities at iteration 0 that Hidden Nodes = 1 given V = v0
    # phk- Vector of Probabilities at iteration k that Hidden Nodes = 1 given V = vk
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0) # Preserves the Dimensions
        self.a += torch.sum((ph0 - phk), 0)
    
nv = len(training_set[0])
nh = 100 # Hyperparameter-Number of features
batch_size = 100 # Another Hyperparameter

rbm = RBM(nv, nh) # Creating an Object

# Training the RBM

nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.0
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user : id_user + batch_size]
        v0 = training_set[id_user : id_user + batch_size]
        ph0,_ = rbm.sample_h(v0)
        # k-steps in Contrastive Divergence
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]  # Freezing the originally -1 datapoints
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.0
    train_loss /= s
    print(f"Epoch: {epoch}; Loss: {train_loss}")
    
# Testing the RBM

test_loss = 0
s = 0.0
for id_user in range(nb_users):
    v = training_set[id_user : id_user + 1] # Crucial Point 1: Needed to activate the Neurons
    vt = test_set[id_user : id_user + 1]
    # ph0,_ = rbm.sample_h(v0) => Needed only for training
    
    # for k in range(10): You have been blind-folded trained to take 10 1-steps and remain in the straight line
                         #All you need to do now is take one step (High prob that you will stay on the line due to training)
    if len(vt[vt>=0]) > 0: 
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.0
test_loss /= s
print(f"Loss: {test_loss}")