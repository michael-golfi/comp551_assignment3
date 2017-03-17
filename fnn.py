import random
import numpy as np
from random import shuffle


def main():
    number_of_layers = len(sizes)
    sizes = sizes
    biases = [np.random.randn(y, 1) for y in sizes[1:]]
    weights = [np.random.randn(y, x)
                for x, y in zip(sizes[:-1], sizes[1:])]

def ff_process(a):
    for b, w in zip(biases, weights):
        a = sig(np.dot(w, a)+b)
    return a

def update_mb(mini_b, eta):
    set_b = [np.zeros(b.shape) for b in biases]
    set_w = [np.zeros(w.shape) for w in weights]
    for x, y in mini_b:
        delta_set_b, delta_set_w = backprop(x, y)
        set_b = [nb+dnb for nb, dnb in zip(set_b, delta_set_b)]
        set_w = [nw+dnw for nw, dnw in zip(set_w, delta_set_w)]
    weights = [w-(eta/len(mini_b))*nw
                    for w, nw in zip(weights, set_w)]
    biases = [b-(eta/len(mini_b))*nb
                   for b, nb in zip(biases, set_b)]

def backprop(x, y):
    set_b = [np.zeros(b.shape) for b in biases]
    set_w = [np.zeros(w.shape) for w in weights]
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sig(z)
        activations.append(activation)
    delta = deriv_cost(activations[-1], y) * sig_prime(zs[-1])
    set_b[-1] = delta
    set_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in xrange(2, num_layers):
        z = zs[-l]
        sp = sig_prime(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        set_b[-l] = delta
        set_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (set_b, set_w)

def gd(training_set, epochs, mini_b_size, eta):
    n = len(training_set)
    for j in xrange(epochs):
        random.shuffle(training_set)
        mini_bes = [
            training_set[k:k+mini_b_size]
            for k in xrange(0, n, mini_b_size)]
        for mini_b in mini_bes:
            update_mb(mini_b, eta)
        if test_set:
            print "{0}: {1} / {2}".format(
                j, evaluate(test_set), n_test)
        else:
            print "{0} complete".format(j)

#returns a vector of partial derivatives for the activations of the output
def deriv_cost(output_act, y):
    return (output_act-y)

#returns sigmoid
def sig(z):
    return 1.0/(1.0+np.exp(-z))

#takes derivative of sig
def sig_prime(z):
    return sig(z)*(1-sig(z))

def load_data():
    training_x = np.load('tinyX.npy')
    training_y = np.load('tinyY.npy')
    test_data = np.load('tinyX_test.npy')
    return (training_x, training_y, test_data)

def load_data_wrapper():
    training_x, training_y, test_data = load_data()
    training_data = []
    for values in training_x:
        values = values.flatten()
        training_data.append(values)

    testing_data = []
    for values in test_data:
        values = values.flatten()
        testing_data.append(values)

    training_results=[]
    for values in training_y:
        e = np.zeros((40, 1))
        e[values] = 1.0
        training_results.append(e)

    training = zip(training_data,training_results)

    shuffle(training)
    length = len(training)

    training_final = training[:int(length*0.75)]
    validation_final = training[int(length*0.75):]

    return (training_data, validation_final)
