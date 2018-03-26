# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 22:42:44 2018

@author: scott
"""

class SemiSupervisedBinaryClassificationNet(object):
    
    def __init__(self, 
                 layers, 
                 initialiser=tf.truncated_normal, 
                 l2_reg=0.0, 
                 optimiser=tf.train.AdamOptimizer(),
                 lambda_u=1,
                 checkpoint_path='/tmp/my_model_final.ckpt'):
        '''
        
        Inputs:
            layers - (dict) Dictionary of dictionaries specifying properties of each hidden layer
            e.g. {1: {'dim': 100, 'activation': tf.nn.relu, 'dropout_rate': 0.5}
            
            
            'regulariser': tf.contrib.layers.l2_regularizer(1)}
        
        '''
        
        # add output layer
        layers[max(layers, key=int) + 1] = {'dim': 1}
        
        self.first_fit = True
        self.layers = layers
        self.l2_reg = l2_reg
        self.initialiser = initialiser
        self.optimiser = optimiser
        self.lambda_u = lambda_u 
        self.checkpoint_path = checkpoint_path
        self.weights = {}
        self.biases = {}
        self.tf_ops = {}
        
    
    def build_graph(self, input_dim):
        '''
        
        
        '''
        
        
        tf.reset_default_graph()
        
        is_training = tf.placeholder_with_default(False, shape=(), name='training')

        # define the network weights and biases
        self.init_weights(input_dim)

        if self.l2_reg > 0:
            # add regularisation loss
            reg_loss = tf.add_n([tf.nn.l2_loss(W) 
                                      for _, W in self.weights.items()]) * self.l2_reg
        else:
            reg_loss = 0.0

        # define placeholder for input data
        X_l = tf.placeholder(tf.float32, shape=(None, input_dim), name='X_l')
        y_l = tf.placeholder(tf.float32, shape=(None), name='y_l')
        X_u = tf.placeholder(tf.float32, shape=(None, input_dim), name='X_u')

        # labelled forward pass
        logits_l = self.forward_pass(X_l, is_training)
        p = tf.sigmoid(logits_l)

        # unlabelled forward pass
        logits_u = self.forward_pass(X_u, is_training)

        with tf.name_scope('loss'):

            # labelled loss
            xentropy_l = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_l, logits=logits_l) 
            loss_l = tf.reduce_mean(xentropy_l, name='labelled_loss')

            # unlabelled loss
            p_u = tf.sigmoid(logits_u)
            p_u = p_u / tf.reduce_max(p_u)   
            xentropy_u = - (p_u * tf.log(p_u + 1e-15) + (1 - p_u) * tf.log(1 - p_u + 1e-15))
            loss_u = tf.reduce_mean(xentropy_u * self.lambda_u, name='unlabelled_loss')

            # sum losses, including regularisation loss
            loss = loss_l + loss_u + reg_loss

        # optimise
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = self.optimiser.minimize(loss)

        init = tf.global_variables_initializer() 
        saver = tf.train.Saver()
        
        # store tf operations we will need
        self.tf_ops['is_training'] = is_training
        self.tf_ops['X_l'] = X_l
        self.tf_ops['y_l'] = y_l
        self.tf_ops['X_u'] = X_u
        self.tf_ops['logits_l'] = logits_l
        self.tf_ops['logits_u'] = logits_u
        self.tf_ops['p'] = p
        self.tf_ops['loss_l'] = loss_l
        self.tf_ops['loss_u'] = loss_u
        self.tf_ops['loss'] = loss
        self.tf_ops['training_op'] = training_op
        self.tf_ops['init'] = init
        self.tf_ops['saver'] = saver
         
       
    def init_weights(self, input_dim):
        '''
        
        '''
               
        for i, layer in self.layers.items():
                                       
            # use weight initialiser
            output_dim = layer['dim']
            self.weights[i] = tf.Variable(self.initialiser((input_dim, output_dim)), 
                                          name='kernel')  
                                     
            # initialise with zeros
            self.biases[i] = tf.Variable(tf.zeros([output_dim]), 
                                         name='bias')
            
            # output dimension becomes new input dimension
            input_dim = int(self.weights[i].get_shape()[1])
        
        
            
            
    def forward_pass(self, X, training=False):
        '''
        
        '''
        
        # hidden layers
        for i, layer in self.layers.items():
            
            # get weights and biases for this layer
            W = self.weights[i]
            b = self.biases[i]
        
            # compute linear transformation
            Z = tf.matmul(X, W) + b
            
            # batch normalisation?
            if 'batch_norm' in layer:
                if layer['batch_norm'] == True:
                    Z = tf.layers.batch_normalization(Z, training=training)
                            
            # compute activations            
            if 'activation' in layer:
                Z = layer['activation'](Z)

            # dropout?
            if 'dropout_rate' in layer:
                Z = tf.layers.dropout(Z, rate=layer['dropout_rate'], training=training)
                
            X = Z
        
        return Z

                
    
        
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
                    
                # compute training loss
                loss_train = self.tf_ops['loss_l'].eval(feed_dict={self.tf_ops['X_l']: X_l_train,
                                                                    self.tf_ops['y_l']: y_l_train})
                
                if verbose:
                    print('Training loss =', loss_train)
                
                if (X_l_val is not None) & (y_l_val is not None):
                    
                    # compute validation loss
                    loss_val = self.tf_ops['loss_l'].eval(feed_dict={self.tf_ops['X_l']: X_l_val,
                                                                    self.tf_ops['y_l']: y_l_val})
                    
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
                    
                                               
    
    def predict(self, X):
        '''
        
        
        '''
        
        # build computational graph
        input_dim = X.shape[1]
        self.build_graph(input_dim)
        
        with tf.Session() as sess:
        
            # restore variables
            self.tf_ops['saver'].restore(sess, self.checkpoint_path)
            
            # return probabilities
            return self.tf_ops['p'].eval(feed_dict={self.tf_ops['X_l']: X}).squeeze()
                    