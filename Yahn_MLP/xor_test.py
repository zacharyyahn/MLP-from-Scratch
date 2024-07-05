import numpy as np 
import matplotlib.pyplot as plt
from mlp import MLP

X = [
    [[0], [0]],
    [[0], [1]],
    [[1], [0]],
    [[1], [1]]
]
y = [
    [[0]],
    [[1]],
    [[1]],
    [[0]]
]

net = MLP(input_dim = 2, hidden_dim = 3, output_dim = 1, hidden_activation="sigmoid", output_activation="sigmoid", debug=False, eval_type="one_val")
net.initialize(mode="random")
net.train(X, y, num_epochs=3000, lr=1, X_test=X, y_test=y, visualize=True)
acc, err = net.evaluate(X, y)
print("Accuracy:", acc, "Average Error:", err )
plt.show()

"""
RESULTS
- To get XOR to work, do 2, 4, 1, ha = sigmoid, oa = sigmoid, 3000 epochs, lr = 1
"""