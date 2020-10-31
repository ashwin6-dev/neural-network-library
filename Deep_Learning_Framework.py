

# SIMPLE NEURAL NETWORK

import framework as fw
import framework.nn as nn
import framework.optim as optim
model = nn.Model([
   nn.Linear(75, input_shape=[4,2]), 
   nn.ReLu(),
   nn.Linear(50, input_shape=[75]), 
   nn.ReLu(),
   nn.Linear(25 ,input_shape=[50]),
   nn.Sigmoid(),
   nn.Linear(10,input_shape=[25]),
   nn.Sigmoid(),
   nn.Linear(1 ,input_shape=[10]),
   nn.Sigmoid()
])

#Solving for XOR problem

x = fw.Tensor([[1, 0], [0, 1], [0, 0], [1, 1]])
y = fw.Tensor([[1], [1], [0], [0]])

model.train(x, y, epochs=400, optim=optim.Momentum(learning_rate=0.2))

print (model.predict(x))
