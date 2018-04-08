# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:42:44 2018

@author: scott
"""

import tensorflow as tf
import numpy as np

class SemiSupervisedBinaryClassificationNet(object):
    
    def __init__(self, 
                 layers, 
                 initialiser=tf.keras.initializers.glorot_uniform(), 
                 l2_reg=0.0, 
                 optimiser=tf.train.AdamOptimizer(),
                 lambda_u=1.0,
                 checkpoint_path='/tmp/my_model_final.ckpt'):
        '''
        Class initialiser
        
        Inputs:
            layers - (dict) Dictionary of dictionaries specifying properties of each hidden layer
            e.g. {1: {'dim': 20, 'activation': tf.nn.relu, 'dropout_rate': 0.5, 'batch_norm': True},
                  2: {'dim': 20, 'activation': tf.nn.relu, 'dropout_rate': 0.5, 'batch_norm': True}}
            initialiser - (tf op) tensorflow initialiser
                          default=tf.keras.initializers.glorot_uniform()
            l2_reg - (float >= 0) l2 regularisation parameter
                     default=0.0
            optimiser - (tf op) tensorflow optimiser
                        default=tf.train.AdamOptimizer()
            lambda_u - (float >= 0) weight of unlabelled examples in cost function 
                       default=1.0
            checkpoint_path - (string) path for saving model checkpoint
                              default='/tmp/my_model_final.ckpt'
        '''
        
        # add linear output layer with a single neuron to layer dictionary
        layers[max(layers, key=int) + 1] = {'dim': 1}
        
        self.layers = layers
        self.initialiser = initialiser
        self.l2_reg = l2_reg
        self.optimiser = optimiser
        self.lambda_u = lambda_u 
        self.checkpoint_path = checkpoint_path
        self.weights = {}
        self.biases = {}
        self.tf_ops = {}
        self.first_fit = True
        
    
    def fit(self, 
            X_l_train, 
            y_l_train, 
            X_u_train, 
            X_l_val=None,
            y_l_val=None,
            batch_size=32, 
            num_epochs=100, 
            patience=10,
            verbose=True):
        '''
        Trains the network
        
        Inputs:
            X_l_train - (np array) labelled training features
            y_l_train - (np vector) training labels (0/1)
            X_u_train - (np array) unlabelled training features
            X_l_val - (np array) validation training features
                      default=None
            y_l_val - (np vector) validation labels (0/1) 
                      default=None
            batch_size - (int > 0) training batch size
                         default=32
            num_epochs - (int > 0) number of training epochs
                         default=100
            patience - (int > 0) if validation data provided, 
                       number of epochs without improvement to wait before early stopping
                       default=10
            verbose - (bool) whether or not to print updates after every epoch
                      default=True
        '''
        
        # get data dimensions
        num_l, input_dim = X_l_train.shape
        num_u = X_u_train.shape[0]
        
        # build computational graph
        self.build_graph(input_dim)
                    
        # train network               
        with tf.Session() as sess:
            
            if self.first_fit:                
                # initialise variables
                sess.run(self.tf_ops['init'])
                self.first_fit = False                                                
            else:                
                # restore variables
                self.tf_ops['saver'].restore(sess, self.checkpoint_path)
            
            if (X_l_val is not None) & (y_l_val is not None):
                # compute initial validation loss
                best_loss = self.tf_ops['loss_l'].eval(feed_dict={self.tf_ops['X_l']: X_l_val,
                                                                    self.tf_ops['y_l']: y_l_val})
            
            # train for num_epochs
            num_batches = int(np.ceil(float(num_l) / batch_size))
            for epoch in np.arange(1, num_epochs + 1):

                if verbose:
                    print('+-----------------------------------------------------------+')
                    print('Running epoch', epoch, 'of', num_epochs)

                # shuffle examples
                if epoch == 1:
                    shuffle_l = np.random.choice(num_l, num_l, replace=False)
                    shuffle_u = np.random.choice(num_u, num_u, replace=False)
                else:
                    shuffle_l = shuffle_l[np.random.choice(num_l, num_l, replace=False)]
                    shuffle_u = shuffle_u[np.random.choice(num_u, num_u, replace=False)]
                X_l_train = X_l_train[shuffle_l]
                y_l_train = y_l_train[shuffle_l]
                X_u_train = X_u_train[shuffle_u]
                
                # train in batches
                for batch in np.arange(num_batches):

                    # get labelled data in this batch
                    i_first = batch * batch_size
                    i_last = (1 + batch) * batch_size
                    i_last = min(num_l, i_last)
                    X_l_batch = X_l_train[i_first:i_last]
                    y_l_batch = y_l_train[i_first:i_last]

                    # sample a batch of unlabelled examples of the same size
                    X_u_batch = X_u_train[np.random.choice(num_u, X_l_batch.shape[0], replace=False)]
                    
                    # run training operation
                    sess.run(self.tf_ops['training_op'], 
                             feed_dict={self.tf_ops['X_l']: X_l_batch,
                                        self.tf_ops['y_l']: y_l_batch,
                                        self.tf_ops['X_u']: X_u_batch,
                                        self.tf_ops['is_training']: True})
                    
                # compute cross entropy of training examples in batches
                _, xentropy_train = self.apply_in_batches(sess, 
                                                          X_l_train, 
                                                          y_l_train, 
                                                          batch_size=batch_size)
                
                # compute mean cross entropy
                loss_train = xentropy_train.mean()
                
                if verbose:
                    print('Training loss =', loss_train)
                
                if (X_l_val is not None) & (y_l_val is not None): 
                    
                    # compute cross entropy of validation examples in batches
                    _, xentropy_val = self.apply_in_batches(sess, 
                                                            X_l_val, 
                                                            y_l_val, 
                                                            batch_size=batch_size)

                    # compute mean cross entropy
                    loss_val = xentropy_val.mean()                  
                
                    if verbose:
                        print('Validation loss =', loss_val)
                    
                    # loss improved?
                    if loss_val < best_loss:                        
                        best_loss -= best_loss - loss_val
                        epochs_since_improvement = 0
                        
                        # save model checkpoint
                        self.tf_ops['saver'].save(sess, self.checkpoint_path)
                        
                    else:                        
                        epochs_since_improvement += 1
                        
                        # early stopping?
                        if epochs_since_improvement >= patience:                               
                            if verbose:
                                print('Early stopping. Best epoch =', epoch - patience)                            
                            break
                            
            if (X_l_val is None) or (y_l_val is None):                
                # save final model
                self.tf_ops['saver'].save(sess, self.checkpoint_path)
        
    
    def build_graph(self, input_dim):
        '''
        Builds the tensorflow computational graph
        
        Inputs:
            input_dim - (int > 0) number of input features        
        '''
        
        tf.reset_default_graph()
        
        # create placeholder to specify whether we are training or predicting
        is_training = tf.placeholder_with_default(False, shape=(), name='training')

        # define the network weights and biases
        self.init_weights(input_dim)

        if self.l2_reg > 0:
            # add regularisation loss
            reg_loss = tf.add_n([tf.nn.l2_loss(W) 
                                      for _, W in self.weights.items()]) * self.l2_reg
        else:
            reg_loss = 0.0

        # define placeholders for input data
        X_l = tf.placeholder(tf.float32, shape=(None, input_dim), name='X_l')
        y_l = tf.placeholder(tf.float32, shape=(None), name='y_l')
        X_u = tf.placeholder(tf.float32, shape=(None, input_dim), name='X_u')

        # labelled forward pass
        logits_l = self.forward_pass(X_l, is_training)
        p_l = tf.sigmoid(logits_l)

        # unlabelled forward pass
        logits_u = self.forward_pass(X_u, is_training)

        with tf.name_scope('loss'):

            # labelled loss
            xentropy_l = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_l, logits=logits_l)) 
            loss_l = tf.reduce_mean(xentropy_l, name='labelled_loss')

            # unlabelled loss
            p_u = tf.sigmoid(logits_u)
            p_u = p_u / tf.reduce_max(p_u)   
            xentropy_u = tf.squeeze(- (p_u * tf.log(p_u + 1e-15) + (1 - p_u) * tf.log(1 - p_u + 1e-15)))
            loss_u = tf.reduce_mean(xentropy_u * self.lambda_u, name='unlabelled_loss')

            # sum losses, including regularisation loss
            loss = loss_l + loss_u + reg_loss

        # optimise
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = self.optimiser.minimize(loss)

        # define initialiser and saver nodes
        init = tf.global_variables_initializer() 
        saver = tf.train.Saver()
        
        # store tf operations we will need to access from other methods
        self.tf_ops['is_training'] = is_training
        self.tf_ops['X_l'] = X_l
        self.tf_ops['y_l'] = y_l
        self.tf_ops['X_u'] = X_u
        self.tf_ops['logits_l'] = logits_l
        self.tf_ops['logits_u'] = logits_u
        self.tf_ops['p'] = p_l
        self.tf_ops['xentropy_l'] = xentropy_l
        self.tf_ops['loss_l'] = loss_l
        self.tf_ops['loss_u'] = loss_u
        self.tf_ops['loss'] = loss
        self.tf_ops['training_op'] = training_op
        self.tf_ops['init'] = init
        self.tf_ops['saver'] = saver
         
       
    def init_weights(self, input_dim):
        '''
        Defines the network weights and biases
        
        Inputs:
            input_dim - (int > 1) number of input features                      
        '''
         
        # loop through layers
        for i, layer in self.layers.items():
                                       
            # define weights
            output_dim = layer['dim']
            self.weights[i] = tf.Variable(self.initialiser((input_dim, output_dim)), 
                                          name='kernel')  
                                     
            # define biases
            self.biases[i] = tf.Variable(tf.zeros([output_dim]), 
                                         name='bias')
            
            # output dimension becomes new input dimension
            input_dim = int(self.weights[i].get_shape()[1])
                    
            
    def forward_pass(self, X, training=False):
        '''
        Performs forward pass through network to compute logits
        
        Inputs:
            X - (np array) features
            training - (bool) whether or not we are in training mode
                       default=False
        
        Outputs:
            Z - (tensor) logits
        '''
        
        # loop through layers
        for i, layer in self.layers.items():
            
            # get weights and biases for this layer
            W = self.weights[i]
            b = self.biases[i]
        
            # compute linear transformation
            Z = tf.matmul(X, W) + b
            
            if 'batch_norm' in layer:
                if layer['batch_norm'] == True:
                    # perfom batch normalisation
                    Z = tf.layers.batch_normalization(Z, training=training)
                                       
            if 'activation' in layer:
                # compute activations 
                Z = layer['activation'](Z)

            if 'dropout_rate' in layer:
                # dropout
                Z = tf.layers.dropout(Z, rate=layer['dropout_rate'], training=training)
            
            # layer output becomes input for next layer
            X = Z
        
        return tf.squeeze(Z)
    
    
    def apply_in_batches(self, sess, X, y=None, batch_size=32):
        '''
        Applys the network to the input data in batches.
        If y is provided returns the probabilities of the poitives class
        as well as the cross entropy of each examples.
        If y is not provided returns only the probabilities.
        
        Inputs:
            sess - (tf Session) current tensorflow session
            X - (np array) features
            y - (np vector) class labels (0/1)
                default=None
            batch_size - (int > 0) batch size
                         default = 32
                         
        Outputs:
            p - (np vector) probability of positive class for each input
            xentropy - (np vector) cross entropy for each input (only return if y provided)            
        '''
        
        # define variables for storing outputs
        num_x = X.shape[0]        
        p = np.zeros(num_x)
        if y is not None:        
            xentropy = np.zeros(num_x)
        
        # loop through batches
        num_batches = int(np.ceil(float(num_x) / batch_size))
        for batch in np.arange(num_batches):

            # get data in this batch
            i_first = batch * batch_size
            i_last = (1 + batch) * batch_size
            i_last = min(num_x, i_last)
            X_batch = X[i_first:i_last]
            
            if y is None:
                # compute probabilities
                p[i_first:i_last] = self.tf_ops['p'].eval(feed_dict={self.tf_ops['X_l']: X_batch})
            else:
                # compute probabilities and cross entropy
                y_batch = y[i_first:i_last]
                p[i_first:i_last], xentropy[i_first:i_last] = sess.run([self.tf_ops['p'], 
                                                                        self.tf_ops['xentropy_l']],
                                                                        feed_dict={
                                                                        self.tf_ops['X_l']: X_batch,
                                                                        self.tf_ops['y_l']: y_batch})
                
        if y is not None:
            return p, xentropy
        else:
            return p
                                                                   
    def predict(self, X, batch_size=32):
        '''
        Predicts probability of positive class for each input
        
        Inputs:
            X - (np array) features
            batch_size - (int > 0) batch size for computing probabilities
                         default = 32
            
        Outputs:
            p - (np vector) probability of positive class for each input
        
        '''
        
        # build computational graph
        num_x, input_dim = X.shape
        self.build_graph(input_dim)
        
        with tf.Session() as sess:
        
            # restore variables
            self.tf_ops['saver'].restore(sess, self.checkpoint_path)
            
            # compute probabilities in batches
            p = self.apply_in_batches(sess, X, batch_size=batch_size)
              
        return p                    
