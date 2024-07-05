import numpy as np 
import matplotlib.pyplot as plt
from mlp import MLP


# ---- SINE ----
X = []
y = []
for i in range(500):
    vec = []
    for i in range(4):
        vec.append([np.random.rand() * 2 - 1])
    y.append(np.sin(vec[0][0] - vec[1][0] + vec[2][0] - vec[3][0]))
    X.append(vec) 

X_train = X[:400]
y_train = y[:400]
X_test = X[400:]
y_test = y[400:]

net = MLP(input_dim = 4, hidden_dim = 7, output_dim = 1, hidden_activation="sigmoid", output_activation="linear", debug=False)
net.initialize(mode="random")
net.train(X_train, y_train, num_epochs=1000, lr=0.01, X_test=X_test, y_test=y_test, visualize=True, batch_size=1)
acc, err = net.evaluate(X_test, y_test)
acc2, err2 = net.evaluate(X_train, y_train)
print("Accuracy:", acc, "Average Test Error:", err )
plt.show()


"""
RESULTS
- Sine seems to work very well with 4, 7, 1, ha = sigmoid, oa = linear, 1000 epochs, lr = 0.05
"""
