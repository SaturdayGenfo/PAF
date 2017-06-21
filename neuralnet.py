# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:58:18 2017

@author: leello
"""
import numpy as np

class Caliente(object):
    
    def __init__(self, nombre_par_couche):
        self.nlayers = len(nombre_par_couche)
        self.nperlayer = nombre_par_couche
        
        #initialise les biais, un biais par neurone
        self.biases = [np.random.randn(k, 1) for k in nombre_par_couche[1:]]
        
        self.weights  = [np.random.randn(n_sortie, n_entree) for n_entree, n_sortie in zip(nombre_par_couche[:-1], nombre_par_couche[1:])]
    
    def sigmoid(self, alpha):
        return 1.0/(1.0+np.exp(-1*alpha))
        
    def sigmoidprime(self, alpha):
        return self.sigmoid(alpha)*(1-self.sigmoid(alpha))
        
    def cost_derivative(self, a, y):
        return (a.reshape((10, 1)) - y.reshape((10, 1)))
    
    
    def passthrough(self, input_vect):
        nxt = input_vect
        for b, w in zip(self.biases, self.weights):
            dprod = np.dot(w, nxt)
            dprod = dprod.reshape(b.shape)
            nxt = self.sigmoid(dprod + b)
        return nxt
    
    def gradient_descent(self, training_data, n_rounds, batch_size, learning_coef):
        
        n = len(training_data)
        for k in range(n_rounds):
            np.random.shuffle(training_data)
            batches = [training_data[i: i+ batch_size] for i in range(0, n, batch_size)]
            for batch in batches:
                self.update(batch, learning_coef)
            print("Muy Caliente {0}".format(k+1))
    
    def update(self, batch, learning_coef):
        
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]
        
        m=len(batch)
        
        for (x, y) in batch:
            local_grad_b, local_grad_w = self.backprop(x, y)
            grad_b = [l_b + b for l_b, b in zip(local_grad_b, grad_b)]
            grad_w = [l_w + w for l_w, w in zip(local_grad_w, grad_w)]
            
        self.weights = [w - learning_coef*g_w/m for g_w, w in zip(grad_w, self.weights)]
        self.biases = [b - learning_coef*g_b/m for g_b, b in zip(grad_b, self.biases) ]
    
    def backprop(self, x, y):
        
        #forward feeding
        a = [np.zeros(k) for k in self.nperlayer]
        z = [np.zeros(k) for k in self.nperlayer]
        a[0] = np.array(x)
        a[0] = a[0].reshape((a[0].shape[0], 1))
        l = 0
        for b, w in zip(self.biases, self.weights):
            #print(w.shape, a[l].shape, b.shape)
            dprod = np.dot(w, a[l])
            dprod = dprod.reshape(b.shape)
            znxt =  dprod + b
            #print(znxt.shape)
            z[l+1] = znxt
            a[l+1] = self.sigmoid(znxt)
            l += 1
        
        #init of errors

        nablaC = self.cost_derivative(a[-1], y)
        
        delta_l  = np.multiply(nablaC,self.sigmoidprime(z[-1]))
        #print(nablaC.shape, delta_l.shape)
        grad_b = [np.zeros_like(b) for b in self.biases]
        grad_w = [np.zeros_like(w) for w in self.weights]
        
        #backprop and fill in
        l = self.nlayers - 2
        grad_b[-1] = delta_l
        grad_w[-1] = np.dot(delta_l, np.transpose(a[l]))
        while l > 0:
            delta_l = np.multiply(np.dot(np.transpose(self.weights[l]), delta_l), self.sigmoidprime(z[l]))
            grad_b[l-1] = delta_l
            grad_w[l-1] = np.dot(delta_l, np.transpose(a[l-1]))
            l -= 1
        
        return grad_b, grad_w
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.passthrough(x)), np.argmax(y))
                        for (x, y) in test_data]
                            
        return sum(int(x == y) for (x, y) in test_results)
            
            
        
        
        
            
        
        
        