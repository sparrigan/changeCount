from __future__ import division, print_function
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

# Get training data
data = datasets.load_digits()
inp_data = data.data
targets = data.target

# Define training and test sets
training_set = slice(1000)
test_set = slice(1000,None)
X = inp_data[training_set]
y = targets[training_set]
# Train MLP classifier
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5,2))#, random_state=1)
clf.fit(X, y)

predictions = clf.predict(inp_data[test_set])

print(predictions)
print(clf.coefs_)
print("APPROX ACCURACY: ", sum(predictions == targets[test_set]) / len(predictions))
