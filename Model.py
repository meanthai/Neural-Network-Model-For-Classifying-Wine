import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

np.random.seed(2)

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

# Applying one-hot Encoding to transform the categories into categorical binary vectors suitable for machine learning
def one_hot_encode(labels, num_classes):
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels  
      
# Decoding categorical binary vectors back to the original labels
def one_hot_decode(y):
    return np.argmax(y, axis = 1)


def relu(value):
    return np.maximum(0, value)


def relu_derivative(value):
    return np.where(value > 0, 1, 0)

# Calculating the cost of the model using the cross_entrophy formula
def cross_entropy(y_pred, y_true):
    # Avoid numerical errors by adding a very small value to probabilities
    eps = 1e-15
    loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=0))
    return loss


def shuffle(arr):
    shuffled_indices = np.random.permutation(len(arr))
    shuffled_data = arr[shuffled_indices]
    return shuffled_data


class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, learning_rate=0.00001, epochs = 3000):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # initialize weights randomly
        self.weights1 = np.random.randn(self.input_size, self.hidden_size1)
        self.weights2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.weights3 = np.random.randn(self.hidden_size2, self.hidden_size3)
        self.weights4 = np.random.randn(self.hidden_size3, self.output_size)
        
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
            activation4 = relu(layer4)
            
            
            # backpropagation
            error = activation4 - y
            d_weights4 = activation3.T.dot(error * relu_derivative(layer4))
            d_bias4 = np.sum(error * relu_derivative(layer4), axis = 0, keepdims = True)
            
            error_hidden3 = error.dot(self.weights4.T) * relu_derivative(layer3)
            d_weights3 = activation2.T.dot(error_hidden3)
            d_bias3 = np.sum(error_hidden3, axis = 0, keepdims = True)
            
            error_hidden2 = error_hidden3.dot(self.weights3.T) * relu_derivative(layer2)
            d_weights2 = activation1.T.dot(error_hidden2)
            d_bias2 = np.sum(error_hidden2, axis = 0, keepdims = True)
            
            error_hidden1 = error_hidden2.dot(self.weights2.T) * relu_derivative(layer1)
            d_weights1 = X.T.dot(error_hidden1)
            d_bias1 = np.sum(error_hidden1, axis = 0, keepdims = True)
            
            
            # update weights and biases
            self.weights4 -= self.learning_rate * d_weights4
            self.bias4 -= self.learning_rate * d_bias4
            self.weights3 -= self.learning_rate * d_weights3
            self.bias3 -= self.learning_rate * d_bias3
            self.weights2 -= self.learning_rate * d_weights2
            self.bias2 -= self.learning_rate * d_bias2
            self.weights1 -= self.learning_rate * d_weights1
            self.bias1 -= self.learning_rate * d_bias1
            
            
            # Computing the loss function for every 10 epochs
            if epoch % 10 == 0:
                y_predict = self.predict(X)
                loss_val = cross_entropy(one_hot_decode(y), y_predict)
                print(f"The loss value of the model after {epoch}th training is: ", loss_val)
                # Check convergence to prevent overlifting
                if(np.mean(one_hot_decode(y) == y_predict) >= 0.99):
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
        # Return the class with the highest probability for each sample
        return np.argmax(probs, axis=1)


# Get the Wine dataset using pandas library
url = "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
data = pd.read_csv(url)

# Slice the dataset to get one array for features only and one for categories
data_value = shuffle(data.values)
X = np.array(data_value[:, 1:])
y = np.array(data_value[:, 0], dtype = int) - 1

# Convert categorical labels to an one-hot encoding array
y_one_hot = one_hot_encode(y.flatten(), num_classes=3)

# Take features accounting for most variance out of all using PCA (to reduce dimensionality)
num_classes = 3
num_features = 8
X = PCA(X, num_features)

# initialize the input size, hidden size, and output size and train the model
mlp = MLP(input_size=num_features, hidden_size1=15, hidden_size2=15, hidden_size3=20, output_size=num_classes)
mlp.fit(X, y_one_hot)

y_predict = mlp.predict(X)
print("The predicted results of the model after the training process: ", y_predict)

# Accuracy calculation
accuracy = np.mean(y_predict == one_hot_decode(y_one_hot))
print("Accuracy of the model after training is: ", accuracy)

