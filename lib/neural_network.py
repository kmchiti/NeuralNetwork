import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='svhn.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=100,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            low = -np.sqrt(6/(all_dims[layer_n - 1] + all_dims[layer_n]))
            high = np.sqrt(6/(all_dims[layer_n - 1] + all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(low, high, size = (all_dims[layer_n-1], all_dims[layer_n]))
        # return self.weights

    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            return (x > 0) * 1
        # WRITE CODE HERE
        else:
          return (x > 0) * x
        return 0

    def sigmoid(self, x, grad=False):
        f = 1 / (1 + np.exp(-x))
        if grad:
            # WRITE CODE HERE
            return f * (1 - f)
        # WRITE CODE HERE
        else:
          return f
        return 0

    def tanh(self, x, grad=False):
        t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        if grad:
            # WRITE CODE HERE
            return 1-t**2
        # WRITE CODE HERE
        else:
          return t
        return 0

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad:
            # WRITE CODE HERE
            return np.where(x > 0, 1, alpha)
        # WRITE CODE HERE
        else:
          return np.where(x > 0, x, x * alpha)
        return 0

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            # WRITE CODE HERE
            return self.leakyrelu(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        shifted_x = x - np.max(x)
        if (len(x.shape)>1):
            return np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis = 1, keepdims = True)
        return np.exp(shifted_x) / np.sum(np.exp(shifted_x))

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        for layer_n in range(1, self.n_hidden + 2):
          cache[f"A{layer_n}"] = cache[f"Z{layer_n - 1}"] @ self.weights[f"W{layer_n}"]+ self.weights[f"b{layer_n}"]
          if (layer_n == self.n_hidden + 1):
            cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])  
          else:
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        for layer_n in range(self.n_hidden + 1,0,-1):
          if (layer_n == self.n_hidden + 1): 
            grads[f"dA{layer_n}"] = output - labels #graient of cross entropy loss w.r.t input feature map
            grads[f"dW{layer_n}"] = (cache[f"Z{layer_n - 1}"].T @ grads[f"dA{layer_n}"]) / labels.shape[0]
            grads[f"db{layer_n}"] = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True) / labels.shape[0]
          else: 
            grads[f"dZ{layer_n}"] = grads[f"dA{layer_n + 1}"] @ self.weights[f"W{layer_n+1}"].T
            grads[f"dA{layer_n}"] = grads[f"dZ{layer_n}"] * self.activation(cache[f"A{layer_n}"],grad=True)
            grads[f"dW{layer_n}"] = (cache[f"Z{layer_n - 1}"].T @ grads[f"dA{layer_n}"]) / labels.shape[0]
            grads[f"db{layer_n}"] = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True) / labels.shape[0]
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"] 

    def one_hot(self, y):
        # WRITE CODE HERE
        one_y = np.zeros((y.size, y.max()+1))
        one_y[np.arange(y.size),y] = 1
        return one_y.astype(int)


    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return -np.sum(labels * np.log(prediction)) / labels.shape[0]

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs, loaded = False):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]

        if (loaded == False):
          self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)
                
            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            print("Epoch:",epoch," train_accuracy=",train_accuracy, " validation_accuracy=",valid_accuracy)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

    def normalize(self):
        # WRITE CODE HERE
        # compute mean and std along the first axis
        mu = np.mean(self.train[0], axis=0)
        std = np.std(self.train[0], axis=0)
        self.train = (self.train[0] - mu) / (std + self.epsilon) , self.train[1]
        self.valid = (self.valid[0] - mu) / (std + self.epsilon) , self.valid[1]
        self.test = (self.test[0] - mu) / (std + self.epsilon) , self.test[1]

    def summery(self):
      param = 0
      for key, value in self.weights.items():
        param += value.shape[0]*value.shape[1]
        print (key, value.shape)
      print("Total parameters:", param)
    
    def save_model(self):
      a_file = open("model.pkl", "wb")
      model = self.weights
      pickle.dump(model, a_file)
      a_file.close()

    def load_model(self,model_path="data.pkl"):
      a_file = open(model_path, "rb")
      Model = pickle.load(a_file)
      self.weights = Model
      a_file.close()


    def shuffel_evaluate(self,p=0.1):
      X_train, y_train = self.train
      X_test, y_test = self.test
      y_train_shuffel = y_train
      y_test_shuffel = y_test
      np.random.shuffle(y_train_shuffel[0:int(len(y_train_shuffel)*p)])
      np.random.shuffle(y_test_shuffel[0:int(len(y_test_shuffel)*p)])
      train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train_shuffel)
      test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test_shuffel)
      print("train_accuracy=",train_accuracy, " test_accuracy=",test_accuracy)

    def noisy_evaluate(self,sigma=0.01):
      X_train, y_train = self.train
      X_test, y_test = self.test
      nosiy_X_train = X_train + np.random.normal(np.mean(X_train),sigma,size=X_train.shape)
      nosiy_X_test = X_test + np.random.normal(np.mean(X_train),sigma,size=X_test.shape)
      train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(nosiy_X_train, y_train)
      test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(nosiy_X_test, y_test)
      print("train_accuracy=",train_accuracy, " test_accuracy=",test_accuracy)

    def plot_accuracy(self, n_epochs, log):
      plt.figure(figsize=(8, 6))
      plt.plot(np.arange(n_epochs), log[f"train_accuracy"], label="training")
      plt.plot(np.arange(n_epochs), log[f"validation_accuracy"], label="validation")
      plt.legend()
      plt.ylim(top=1)
      plt.ylim(bottom=0)
      plt.xlabel("epoch")
      plt.ylabel("accuracy")

    def plot_loss(self, n_epochs, log):
      plt.figure(figsize=(8, 6))
      plt.plot(np.arange(n_epochs), log[f"train_loss"], label="training")
      plt.plot(np.arange(n_epochs), log[f"validation_loss"], label="validation")
      plt.legend()
      plt.ylim(top=2.4) 
      plt.ylim(bottom=0)
      plt.xlabel("epoch")
      plt.ylabel("loss")

    def save_logs(self, log):
      a_file = open("log.pkl", "wb")
      pickle.dump(log, a_file)
      a_file.close()
