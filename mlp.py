import numpy as np 
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, input_dim = 2, hidden_dim = 3, output_dim = 2, hidden_activation="sigmoid", output_activation="sigmoid", debug=False, eval_type="one_val"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w = []
        self.b = []
        self.v = []
        self.y = []
        self.z = []
        self.deltas = []
        self.grad = []

        self.train_progress = []
        self.train_error = []
        self.test_error = []
        self.train_acc = []
        self.test_acc = []
        self.debug = debug
        self.eval_type = eval_type
        self.activation_type = output_activation
        
        activation_select = {
            "sigmoid":[self.__sigmoid,self.__sigmoid_prime],
            "tanh":[self.__tanh, self.__tanh_prime],
            "linear":[self.__linear, self.__linear_prime],
            "softmax":[self.__softmax, self.__softmax_prime]
        }
        self.output_activation = activation_select[output_activation][0]
        self.output_activation_prime = activation_select[output_activation][1]
        self.hidden_activation = activation_select[hidden_activation][0]
        self.hidden_activation_prime = activation_select[hidden_activation][1]
    
    def __linear(self, vec):
        return vec.copy()
    
    def __linear_prime(self, vec):
        vec = vec.copy()
        return np.multiply(vec, 1 / vec)

    def __tanh(self, vec):
        vec = vec.copy()
        return np.tanh(vec)
    
    def __tanh_prime(self, vec):
        vec = vec.copy()
        return np.multiply(1 / np.cosh(vec), 1 / np.cosh(vec))

    def __sigmoid(self, vec):
        vec = vec.copy()
        return 1 / (1 + np.exp(-1 * vec))
    
    def __sigmoid_prime(self, vec):
        return np.multiply(self.__sigmoid(vec), 1 - self.__sigmoid(vec))
    
    def __softmax(self, vec):
        vec = vec.copy()
        sum_e = np.sum(np.exp(vec))
        return np.exp(vec) / sum_e
    
    def __softmax_prime(self, vec):
        #When we use softmax the f' goes away, so we return a vector of ones for the vector product
        return np.ones(vec.shape)
    
    def clip_gradients(self, vec, clip):
        vec = vec.copy()
        for i in range(vec.shape[0]):
            if vec[i][0] > 0:
                vec[i][0] = min(vec[i][0], clip)
            if vec[i][0] < 0:
                vec[i][0] = max(vec[i][0], -clip)
        return vec
    
    def norm_gradients(self, vec):
        vec = vec.copy()
        sum_vec = np.sum(vec) #sum of vecs squared
        print(sum_vec)
        vec = vec / sum_vec
        return vec

    #update the weights with one backward pass of the network
    def backprop(self, target, output, lr, current_point, batch_size=1):
        
        #Accumulate gradients
        this_delta = np.multiply(target - output, self.output_activation_prime(self.z[2])) #get the delta for this sample
        self.deltas[1] += this_delta #add this delta to the collective delta for use when we're calculating the grad for the next layer
        self.grad[0] += np.matmul(this_delta, np.transpose(self.y[1])) #also accumulate a gradient using the output from the hidden layer

        #Once we've accumulated equal to the batch, update
        if current_point % batch_size == 0:
            self.w[1] = self.w[1] + (lr * self.grad[0])
            #self.b[1] = self.b[1] + (lr * deltas[1])
            self.deltas[0] += np.multiply(np.matmul(np.transpose(self.w[1]), self.deltas[1]), self.hidden_activation_prime(self.z[1]))
            self.w[0] = self.w[0] + (lr * self.deltas[0] * np.transpose(self.y[0]))
            #self.b[0] = self.b[0] + (lr * deltas[0])
            self.deltas[1] = 0 * self.deltas[1]
            self.deltas[0] = 0 * self.deltas[0]
            self.grad[0] *= 0
            if self.debug == True:
                print("Got output", output, "for target", target, "(z is", self.z[2], " and sigmoid is ", self.output_activation_prime(self.z[2]))
                print("output layer deltas are", self.deltas[1], "output from previous layer is", self.y[1])
                print("hidden layer deltas are", self.deltas[0], "output from previous layer is", self.y[0])
                print("NEW WEIGHTS:\n", self.w)
    
    #produce an output with one forward pass of the network
    def feedforward(self, input_vec, dropout=0.0):
        #First do dropout
        if dropout > 0:
            weightsave = [[],[]]
            weightsave[1] = self.w[1].copy()
            weightsave[0] = self.w[0].copy()
            dropout1 = np.random.choice(a=[0,1], size=self.w[1].shape, p=[dropout, 1-dropout])
            dropout0 = np.random.choice(a=[0,1], size=self.w[0].shape, p=[dropout, 1-dropout])
            self.w[1] = np.multiply(self.w[1], dropout1)
            self.w[0] = np.multiply(self.w[0], dropout0)

        #Now Propagate
        self.y = []
        self.z = []
        self.y.append(input_vec)
        self.z.append(input_vec)
        self.z.append(np.matmul(self.w[0], self.y[0])) #+ self.b[0])
        self.y.append(self.hidden_activation(self.z[1]))
        self.z.append(np.matmul(self.w[1], self.y[1]))# + self.b[1])
        self.y.append(self.output_activation(self.z[2]))

        #Now restore weights and undo dropout
        if dropout > 0:
            self.w[1] = weightsave[1]
            self.w[0] = weightsave[0]
        return self.y[2]

    #choose initial values for the weights and biases
    def initialize(self, mode="random"):
        self.w = []
        self.b = []
        if mode == "random":
            self.w.append((np.random.rand(self.hidden_dim, self.input_dim) - 0.5) / 2) #input layer -> hidden layer weights
            self.w.append((np.random.rand(self.output_dim, self.hidden_dim) - 0.5) / 2) #hidden layer -> output layer weights
        if mode == "ones":
            self.w.append(np.ones(self.hidden_dim, self.input_dim))
            self.w.append(np.ones(self.output_dim, self.hidden_dim))
        self.b.append((np.zeros((self.hidden_dim,1))))
        self.b.append((np.zeros((self.output_dim,1))))

        #Initialize the deltas to 0
        self.deltas.append(np.zeros((self.hidden_dim,1)))
        self.deltas.append(np.zeros((self.output_dim,1)))
        
        self.grad.append(np.zeros((self.output_dim, self.hidden_dim)))

    #train for num_epochs on the data. After each epoch, print out the training error and test error
    def train(self, X, y, num_epochs, lr, X_test=None, y_test=None, visualize=False, batch_size=1, dropout=0.0):
        error = 0
        num_correct = 0
        size = len(X)
        for epoch in range(1, num_epochs+1):
            for i in range(len(X)):
                output = self.feedforward(X[i], dropout)
                self.backprop(y[i], output, lr, i, batch_size)
                if np.argmax(y[i]) == np.argmax(output):
                    num_correct += 1
                error += np.sum(0.5 * np.abs(output - y[i]) * np.abs(output - y[i])) #sum of all values in the vector is the error for that vector, now sum for all vectors
            
            #Record error and print out the number of epochs that have passed
            if num_epochs <= 100:
                if visualize:
                    self.train_progress.append(epoch)
                    self.train_error.append(error / size)
                    acc, err = self.evaluate(X_test, y_test)
                    self.test_error.append(err)
                    self.train_acc.append(num_correct / size)
                    self.test_acc.append(acc)
                print("Completed epoch (", epoch, "/", num_epochs, "), train error =", error / size, ", test acc =", acc, end="\n")
                error = 0
                num_correct = 0
            elif num_epochs <= 1000 and num_epochs > 100:
                if epoch % 10 == 0:
                    if visualize:
                        self.train_progress.append(epoch)
                        self.train_error.append(error / 10 / size)
                        acc, err = self.evaluate(X_test, y_test)
                        self.test_error.append(err)
                        self.train_acc.append(num_correct / 10 / size)
                        self.test_acc.append(acc)
                    print("Completed epoch (", epoch, "/", num_epochs, "), train error =", error / size / 10, ", test acc =", acc, end="\n")
                    error = 0
                    num_correct = 0
            elif num_epochs > 1000:
                if epoch % 100 == 0:
                    if visualize:
                        self.train_progress.append(epoch)
                        self.train_error.append(error / 100 / size)
                        acc, err = self.evaluate(X_test, y_test)
                        self.test_error.append(err)
                        self.train_acc.append(num_correct / 100 / size)
                        self.test_acc.append(acc)
                    print("Completed epoch (", epoch, "/", num_epochs, "), train error =", error / size / 100, ", test acc =", acc, end="\n")
                    error = 0
                    num_correct = 0
        if visualize:
            plt.clf()
            if self.eval_type == "one_hot":
                plt.plot(self.train_progress, self.train_acc, label="Train Accuracy")
                plt.plot(self.train_progress, self.test_acc, label="Test Accuracy")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.title("Training Progress\n(lr=" + str(lr) + ", hidden_num=" + str(self.hidden_dim) + ", act=" + str(self.activation_type) + "\nbatch_sz=" +str(batch_size) + ", dropout=" + str(dropout) + ")")
                plt.legend()
            else:
                plt.plot(self.train_progress, self.train_error, label="Train Error")
                plt.plot(self.train_progress, self.test_error, label="Test Error")
                plt.xlabel("Epochs")
                plt.ylabel("Error")
                plt.title("Training Progress\n(lr=" + str(lr) + ", hidden_num=" + str(self.hidden_dim) + ", act=" + str(self.activation_type) + "\nbatch_sz=" +str(batch_size) + ", dropout=" + str(dropout) + ")")
                plt.legend()
        
        print("\n")
    #generate an output for input data
    def predict(self, the_input):
        return self.feedforward(the_input)

    def evaluate(self, the_input, the_output):
        num_correct = 0
        total_error = 0
        if self.eval_type == "one_val":
            for i in range(len(the_input)):
                pred = self.feedforward(the_input[i], dropout=0.0)
                error = np.sum(0.5 * np.abs(pred - the_output[i]) * np.abs(pred - the_output[i]))
                if error <= 0.1:
                    num_correct += 1
                total_error += error
            return float(num_correct) / float(len(the_input)), total_error / len(the_input)
        if self.eval_type == "one_hot":
            for i in range(len(the_input)):
                pred = self.feedforward(the_input[i])
                error = np.sum(np.abs(pred - the_output[i]))
                if np.argmax(pred) == np.argmax(the_output[i]):
                    num_correct += 1
                total_error += error
            return float(num_correct) / float(len(the_input)), total_error / len(the_input)
