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



#Input and target dataset

#In the previous exercise, you created the vocabulary, the character to integer and the reverse mapping. 
#Now, you'll create your input and target datasets. The problem here is to generate the next character given a sequence of characters as input. 
#So, the input data will be a sequence of characters and the target data will be the next character in the sequence. 
#To achieve this, you'll divide the text into sequences of fixed length and for each sequence find out what is the next character.

#The full text is available in text. The vocabulary, character to integer and integer to character mappings are available in variables vocabulary, 
#char_to_idx and idx_to_char respectively. The length of each sequence is set to 40 and it is available in a variable named maxlen.

# Create empty lists for input and target dataset
input_data = []
target_data = []

# Iterate to get all substrings of length maxlen
for i in range(0, len(text) - maxlen):
    # Find the sequence of length maxlen starting at i
    input_data.append(text[i : i+maxlen])
    
    # Find the next char after this sequence 
    target_data.append(text[i+maxlen])

# Print number of sequences in input data
print('No of Sequences:', len(input_data))



#Create and initialize the input and target vectors

#In this exercise, you'll encode the text input and target data to numeric values. 
#You'll create two tensors x and y to contain these encodings. 
#Input data is a set of sequences and so the tensor x will be three-dimensional. 
#The first dimension is the number of samples, the second and third being the number of time-steps and the size of the vocabulary. 
#Target data is a set of single characters and so y will be two dimensional. 
#The first dimension will be the number of samples and the second will be the size of the vocabulary. 
#You'll first define these tensors and then fill them up with data.

#The vocabulary, length of each sequence, input data, target data, the character to integer mapping are saved in vocabulary, maxlen, input_data, 
#target_data, char_to_idx respectively.

# Create a 3-D zero vector to contain the encoded input sequences
x = np.zeros((len(input_data), maxlen, len(vocabulary)), dtype='float32')

# Create a 2-D zero vector to contain the encoded target characters
y = np.zeros((len(target_data), len(vocabulary)), dtype='float32')

# Iterate over the sequences
for s_idx, sequence in enumerate(input_data):
    # Iterate over all characters in the sequence
    for idx, char in enumerate(sequence):
        # Fill up vector x
        x[s_idx, idx, char_to_idx[char]] = 1    
    # Fill up vector y
    y[s_idx, char_to_idx[target_data[s_idx]]] = 1

    
    
#Create LSTM model in keras
#In the previous exercise, you completed all the preprocessing of the dataset needed to train a neural network. 
#You have the input and target vectors created. Now let's build the LSTM network which can be trained using these input and target vectors.

#The Sequential model is already imported from keras.models. 
#The Dense and LSTM layers are also imported from keras.layers. The vocabulary and the length of each sequence are saved in vocabulary and maxlen respectively.

# Create Sequential model 
model = Sequential()

# Add an LSTM layer of 128 units
model.add(LSTM(128, input_shape=(maxlen, len(vocabulary))))

# Add a Dense output layer
model.add(Dense(len(vocabulary), activation='softmax'))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Print model summary
model.summary()



#Train LSTM model

#In the last lesson, you pre-processed a dataset of selected literary works of Shakespeare so that it can be used to train an LSTM model to generate texts that 
#imitate Shakespeare's unique writing style. 
#You created the input and target vectors. You also built and compiled the LSTM network.

#Now, you'll train this model using the input and target vectors. In this exercise, you'll train the model for only 1 epoch to save time. 
#For a better prediction performance, you should train the model for more epochs with a bigger dataset. 
#A more complex model with more hidden layers and more nodes in each layer would perform way better than this basic model.

#The compiled model is available in the variable model. The input and target vectors are saved in x and y respectively.

# Create Sequential model 
model = Sequential()

# Add an LSTM layer of 128 units
model.add(LSTM(128, input_shape=(maxlen, len(vocabulary))))

# Add a Dense output layer
model.add(Dense(len(vocabulary), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model
model.fit(x, y, batch_size=64, epochs=1, validation_split=0.2)



#Predict next character given a sequence

#Now that your LSTM model is trained, it can be used for prediction. 
#In this exercise, you'll predict the next character given a sequence as input. 
#You are given a sequence of length 40 and you need to find out the next character following the sequence using the trained model.

#The trained model is available in model. 
#The vocabulary, length of each sequence, the character to integer and the integer to character mappings are saved in vocabulary, maxlen, char_to_idx, 
#idx_to_char respectively.

# Input sequence
sentence = "that, poor contempt, or claim'd thou sle"

# Create a 3-D zero vector to contain the encoding of sentence.
X_test = np.zeros((1, maxlen, len(vocabulary)))

# Iterate over each character and convert them to one-hot encoded vector.
for s_idx, char in enumerate(sentence):
    X_test[0, s_idx, char_to_idx[char]] = 1
