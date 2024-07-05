import numpy as np 
import matplotlib.pyplot as plt
from mlp import MLP

X = []
y = []
with open("letters.txt", "r") as f:
    for i in range(20000):
        print("Reading and converting data: (", i+1, "/ 20000 )", end="\r")
        line = f.readline()
        line_arr = line.split(",")
        
        label = line_arr.pop(0)
        label = int(ord(label) - 65) #convert an A to a 0, Z to a 26, etc
        label_arr = np.zeros((26,1))
        label_arr[label][0] = 1 #one-hot encode the letters
        
        values_arr = [[int(x)] for x in line_arr]

        X.append(values_arr)
        y.append(label_arr)

X_train = X[:16000]
y_train = y[:16000]
X_test = X[16000:]
y_test = y[16000:]

hidden_sizes = [200]
lrs = [0.0005]
batch_sizes = [1]
activations = ["softmax"]
dropout = [0.2]
for dropout in dropout:
    for act in activations:
        for hidden_size in hidden_sizes:
            for lr in lrs:
                for batch_size in batch_sizes:
                    print("---------- TRYING LEARNING RATE:", lr, ", HIDDEN SIZE:", hidden_size, ", ACTIVATION:", act, "DROPOUT:", dropout , "----------")
                    net = MLP(input_dim = 16, hidden_dim = hidden_size, output_dim = 26, hidden_activation="sigmoid", output_activation=act, debug=False, eval_type="one_hot")
                    net.initialize(mode="random")
                    net.train(X_train, y_train, num_epochs=2000, lr=lr, X_test=X_test, y_test=y_test, visualize=True, batch_size=batch_size, dropout=dropout)
                    acc, err = net.evaluate(X_test, y_test)
                    print("Test Accuracy:", acc, "Average Test Error:", err )
                    plt.savefig("lr"+str(lr)+","+"hs"+str(hidden_size)+","+act+","+str(batch_size)+"dropout_d="+ str(dropout)+".png")
