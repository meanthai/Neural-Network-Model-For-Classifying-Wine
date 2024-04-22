import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

np.random.seed(2)

# PCA step by step explaination by Fredtou
def PCA(X , num_components):
    #Step-1: Apply normalization method
    # Scaling data using Z-score normalization
    scaler = StandardScaler()
    X_meaned = scaler.fit_transform(X)
    
    #Step-2: Creating covariance matrix
    cov_mat = np.cov(X_meaned, rowvar = False)
     
    #Step-3: Calculating eigen values and eigen vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4: Sorting the eigen vectors in descending order based on the eigen values
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    total = sum(eigen_values)
    var_exp = [( i /total ) * 100 for i in sorted_eigenvalue]
    cum_var_exp = np.cumsum(var_exp)
    print("percentage of cummulative variance per eigenvector in order: ", cum_var_exp)
         
    #Step-5: Extracting the final dataset after applying dimensionality reduction
    eigenvector_subset = sorted_eigenvectors[:, : num_components]
     
    #Step-6:Transforming
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced


def xavier_init(input_size, output_size):
    """Xavier initialization for weights"""
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(input_size, output_size))


# Applying one-hot Encoding to transform the categories into categorical binary vectors suitable for machine learning
def one_hot_encode(labels, num_classes):
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels  
      

# Decoding categorical binary vectors back to the original labels
def one_hot_decode(y):
    return np.argmax(y, axis = 1)

# Using RelU activation (non-linear function) for each neuron in layers
def relu(value):
    return np.maximum(0, value)


# Derivative of relU activation
def relu_derivative(value):
    return np.where(value > 0, 1, 0)


# Using softmax to calculate the probs of each label for multi-categories classification
def softmax(x):
    # Numerically stable softmax
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True)) # Compress the number by using log function and minus by a maximum number of the array
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


# Calculating the cost of the model using the cross_entrophy formula
def cross_entropy(y_pred, y_true):
    eps = 1e-15 # Avoid numerical errors by adding the eps
    loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=0))
    return loss


def shuffle(arr):
    shuffled_indices = np.random.permutation(len(arr))
    shuffled_data = arr[shuffled_indices]
    return shuffled_data


class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, batch, learning_rate=0.1, epochs = 6500):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.batch = batch
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # initialize weights using Xavier Weight Initialization
        self.weights1 = xavier_init(self.input_size, self.hidden_size1)
        self.weights2 = xavier_init(self.hidden_size1, self.hidden_size2)
        self.weights3 = xavier_init(self.hidden_size2, self.hidden_size3)
        self.weights4 = xavier_init(self.hidden_size3, self.output_size)
        
        # initialize biases to 0
        self.bias1 = np.zeros((1, self.hidden_size1))
        self.bias2 = np.zeros((1, self.hidden_size2))
        self.bias3 = np.zeros((1, self.hidden_size3))
        self.bias4 = np.zeros((1, self.output_size))
    
    def fit(self, X, y):
        for epoch in range(self.epochs):
            # feedforward algorithm
            layer1 = X.dot(self.weights1) + self.bias1
            activation1 = relu(layer1)
            layer2 = activation1.dot(self.weights2) + self.bias2
            activation2 = relu(layer2)
            layer3 = activation2.dot(self.weights3) + self.bias3
            activation3 = relu(layer3)
            layer4 = activation3.dot(self.weights4) + self.bias4
            activation4 = softmax(layer4) # Using softmax at the output layer
            
            
            # backpropagation
            error = (activation4 - y) / self.batch
            d_weights4 = activation3.T.dot(error * relu_derivative(layer4)) / self.batch
            d_bias4 = np.sum(error * relu_derivative(layer4), axis = 0, keepdims = True) / self.batch 
            
            error_hidden3 = error.dot(self.weights4.T) * relu_derivative(layer3) / self.batch
            d_weights3 = activation2.T.dot(error_hidden3) / self.batch
            d_bias3 = np.sum(error_hidden3, axis = 0, keepdims = True) / self.batch
            
            error_hidden2 = error_hidden3.dot(self.weights3.T) * relu_derivative(layer2) / self.batch
            d_weights2 = activation1.T.dot(error_hidden2) / self.batch
            d_bias2 = np.sum(error_hidden2, axis = 0, keepdims = True) / self.batch
            
            error_hidden1 = error_hidden2.dot(self.weights2.T) * relu_derivative(layer1) / self.batch
            d_weights1 = X.T.dot(error_hidden1) / self.batch
            d_bias1 = np.sum(error_hidden1, axis = 0, keepdims = True) / self.batch
            
            
            # update weights and biases
            self.weights4 -= self.learning_rate * d_weights4
            self.bias4 -= self.learning_rate * d_bias4
            self.weights3 -= self.learning_rate * d_weights3
            self.bias3 -= self.learning_rate * d_bias3
            self.weights2 -= self.learning_rate * d_weights2
            self.bias2 -= self.learning_rate * d_bias2
            self.weights1 -= self.learning_rate * d_weights1
            self.bias1 -= self.learning_rate * d_bias1
            
            
            # Computing the loss function for every 100 epochs
            if epoch % 100 == 0:
                y_predict = self.predict(X)
                loss_val = cross_entropy(one_hot_decode(y), y_predict)
                print(f"The loss value of the model after {epoch}th training is: ", loss_val)
                # Check convergence to prevent overlifting
                if(np.mean(one_hot_decode(y) == y_predict) >= 0.98):
                    return
    
    # Multi-categories Classification
    def predict(self, X):
        layer1 = X.dot(self.weights1) + self.bias1
        activation1 = relu(layer1)
        layer2 = activation1.dot(self.weights2) + self.bias2
        activation2 = relu(layer2)
        layer3 = activation2.dot(self.weights3) + self.bias3
        activation3 = relu(layer3)
        layer4 = activation3.dot(self.weights4) + self.bias4
        
        # Softmax function for probability calculation
        exp_scores = np.exp(layer4)
        probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-15)
        return np.argmax(probs, axis=1) # Return the class with the highest probability for each sample


# Load the Wine dataset
url = "c:\\Users\\acer\\Downloads\\wine.csv"
raw_Data = pd.read_csv(url)
data = raw_Data.iloc[:, :].values
data = shuffle(data) # Shuffle the dataset

# Slicing the dataset to seperate the labels array and the features array
X = np.array(data[:, 1:])
y = np.array(data[:, 0], dtype=int) - 1

# Applying PCA to scale the dataset and reduce dimensionality
num_features = 10
X = PCA(X, num_features)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Onehot encoding the labels
num_classes = len(set(y))
y_train_encoded = one_hot_encode(y_train, num_classes)
y_test_encoded = one_hot_encode(y_test, num_classes)

# Define and train the model
batch_size = 10
mlp = MLP(input_size=X.shape[1], hidden_size1=10, hidden_size2=10, hidden_size3=10, output_size=num_classes, batch = batch_size)
mlp.fit(X_train, y_train_encoded)

# Prediction and evaluation
y_predict_train = mlp.predict(X_train)
y_predict_test = mlp.predict(X_test)
train_accuracy = np.mean(y_predict_train == y_train)
test_accuracy = np.mean(y_predict_test == y_test)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

