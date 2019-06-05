"""
Simple neural network using pytorch
"""
import torch
import torch.nn as nn

# Prepare the data

# X represents the amount of hours studied and how much time students spent sleeping
X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
# y represent grades. 
y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
# xPredicted is a single input for which we want to predict a grade using 
# the parameters learned by the neural network.
xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor

# Scale units
breakpoint()
X_max, index1 = torch.max(X, 0)
xPredicted_max, index2 = torch.max(xPredicted, 0)

X = torch.div(X, X_max)
xPredicted = torch.div(xPredicted, xPredicted_max)
y = y / 100  # max test score is 100

print("X_max:", X_max)
print("xPredicted_max:", xPredicted_max)
print("X:", X)
print("y:", y)
print("xPredicted:", xPredicted)

class Neural_Network(nn.Module):
    """Neural network class"""
    def __init__(self, input_size=2, output_size=1, hidden_size=3):
        super(Neural_Network, self).__init__()
        # parameters

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # weights
        self.W1 = torch.randn(self.input_size, self.hidden_size) # 3 X 2 tensor
        self.W2 = torch.randn(self.hidden_size, self.output_size) # 3 X 1 tensor
        
    def forward(self, X):
        """forward calculation"""
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3) # final activation function
        return o

    def backward(self, X, y, o):
        """backward calculation"""
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoid_prime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def sigmoid(self, s):
        """calculate sigmoid"""
        return 1 / (1 + torch.exp(-s))
    
    def sigmoid_prime(self, s):
        """calculate derivative of sigmoid"""
        return s * (1 - s)
    
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def save_weights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        """predict"""
        # @TODO: should be passed in as argument
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))
        

NN = Neural_Network()
epoch = 1000
for i in range(epoch):  # trains the NN epoch times
    #print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.save_weights(NN)
NN.predict()