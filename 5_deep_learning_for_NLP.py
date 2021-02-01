#### Natural Language Generation in Python

### Write like Shakespeare

#Simple network using Keras

#By now you have an intuitive understanding of how the gradient values become lesser and lesser as we back-propagate. 
#In this exercise, you'll work on an example to demonstrate this vanishing gradient problem. 
#You'll create a simple network of Dense layers using Keras and checkout the gradient values of the weights for one iteration of back-propagation.

#The Sequential model and the Dense and Activation layers are already imported from Keras. The Keras module backend is also imported. 
#This has a method .gradients() that can be used to get the gradient values of the weights.

from keras.layers import Activation, Dense 

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



# Vanishing Gradients

#Before you start working on more robust applications of language generation, it's best to learn how to identify when you are suffering from the vanishing gradient problem. 
#In this exercise, you will train the network you created in the last exercise using random training data.

#To run a model in Tensorflow, you need to initialize a tensorflow session first by using the InteractiveSession() function. 
#Then you will initialize all variables using the global_variables_initializer() function.

#You will execute gradients node which you created in the last exercise inside this tensorflow session. 
#You will also check some of the gradient values from different layers. 
#Note that tensorflow has been imported as tf. 
#To know how to run a basic model in Tensorflow, check out this post.

# Create a dummy input vector
input_vector = np.random.random((1,8))

# Create a tensorflow session to run the network
sess = tf.InteractiveSession()

# Initialize all the variables
sess.run(tf.global_variables_initializer())

# Evaluate the gradients using the training examples
evaluated_gradients = sess.run(gradients,feed_dict={model.input:input_vector})

# Print gradient values from third layer and two nodes of the second layer
print(evaluated_gradients[4])
print(evaluated_gradients[2][4])



#Vocabulary and character to integer mapping

#Suppose you're working as a Data Scientist in a company that is creating an automatic content generation system to assist human writers and make the writing process 
#more efficient and effective. This system needs to generate text imitating a human writer. 
#Throughout the remainder of this chapter, you will be working on creating a system that generates text in the style of Shakespeare.

#To do this, you will need to first find the vocabulary from the dataset and then create two dictionaries to contain mappings of characters to integers and integers 
#to characters.

#A small collection of Shakespeare's literary works is saved in a string variable named text. All characters in text are in lowercase.

# Find the vocabulary
vocabulary = sorted(set(text))

# Print the vocabulary size
print('Vocabulary size:', len(vocabulary))

# Dictionary to save the mapping from char to integer
char_to_idx = { char : idx for idx, char in enumerate(vocabulary) }

# Dictionary to save the mapping from integer to char
idx_to_char = { idx : char for idx, char in enumerate(vocabulary) }

# Print char_to_idx and idx_to_char
print(char_to_idx)
print(idx_to_char)
