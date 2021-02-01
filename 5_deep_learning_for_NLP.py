#### Natural Language Generation in Python

### Write like Shakespeare

Simple network using Keras

#By now you have an intuitive understanding of how the gradient values become lesser and lesser as we back-propagate. 
#In this exercise, you'll work on an example to demonstrate this vanishing gradient problem. 
#You'll create a simple network of Dense layers using Keras and checkout the gradient values of the weights for one iteration of back-propagation.

#The Sequential model and the Dense and Activation layers are already imported from Keras. The Keras module backend is also imported. 
#This has a method .gradients() that can be used to get the gradient values of the weights.

# Create a sequential model
model = Sequential()

# Create a dense layer of 12 units
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

# Create a dense layer of 8 units
model.add(Dense(8, init='uniform', activation='relu'))

# Create a dense layer of 1 unit
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile the model and get gradients
model.compile(loss='binary_crossentropy', optimizer='adam')
gradients = backend.gradients(model.output, model.trainable_weights)
