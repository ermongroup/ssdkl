import numpy as np
import lasagne


class FNN(object):

    def __init__(self):
        raise NotImplementedError()

    def forward_train(self, input):
        return self.forward(input, train=True)

    def forward_test(self, input):
        return self.forward(input, train=False)

    def forward(self,input,train=True):
        return input


class THEANO_CNN_MNIST(FNN):
    def __init__(self, l2loss=0.0005, input_dim=100):
        self.dim = int(np.sqrt(input_dim))
        self.l2loss = l2loss

        out = lasagne.layers.InputLayer((None, self.dim*self.dim))
        out = lasagne.layers.ReshapeLayer(out, ([0], self.dim, self.dim, 1))
        out = lasagne.layers.DimshuffleLayer(out, (0, 3, 1, 2))

        # conv1
        conv1 = lasagne.layers.Conv2DLayer(out, 32, (5, 5), stride=(1, 1),
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.GlorotUniform(),
                                           b=lasagne.init.Constant(0.1))
        pool1 = lasagne.layers.Pool2DLayer(conv1, (2, 2), stride=2, mode='max')

        # conv2
        conv2 = lasagne.layers.Conv2DLayer(pool1, 64, (5, 5), stride=(1, 1),
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.GlorotUniform(),
                                           b=lasagne.init.Constant(0.1))
        pool2 = lasagne.layers.Pool2DLayer(conv2, (2, 2), stride=2, mode='max')

        # conv3
        conv3 = lasagne.layers.Conv2DLayer(pool2, 64, (3, 3), stride=(1, 1),
                                           nonlinearity=lasagne.nonlinearities.rectify,
                                           W=lasagne.init.GlorotUniform(),
                                           b=lasagne.init.Constant(0.1))
        pool3 = lasagne.layers.Pool2DLayer(conv3, (2, 2), stride=2, mode='max')

    	# conv4
        conv4 = lasagne.layers.Conv2DLayer(pool3, 64, (3, 3), stride=(1, 1),
                             nonlinearity=lasagne.nonlinearities.rectify,
                             W=lasagne.init.GlorotUniform(),
                             b=lasagne.init.Constant(0.1))
        pool4 = lasagne.layers.Pool2DLayer(conv4, (3, 3), stride=2, mode='max')

        # dense
        fc1 = lasagne.layers.DenseLayer(pool4,
                            256, nonlinearity=lasagne.nonlinearities.rectify,
                            W=lasagne.init.GlorotUniform(),
                            b=lasagne.init.Constant(0.1))

        fc2 = lasagne.layers.DenseLayer(fc1,
                            1, nonlinearity=lasagne.nonlinearities.rectify,
                            W=lasagne.init.GlorotUniform(),
                            b=lasagne.init.Constant(0.1))
        self.network = fc2
        self.regularization_layers = [conv1, conv2, conv3, conv4, fc1, fc2]
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

    def forward_for_finetuning_batch_stat(self,input):
        return self.forward(input,finetune=True)

    def forward_no_update_batch_stat(self,input,train=True):
        return self.forward(input,train,False)

    def forward(self,input,train=True,update_batch_stat=True,finetune=False):
    	return lasagne.layers.get_output(self.network, inputs=input)


class FNN_SIMPLE2(FNN):
    def __init__(self,layer_sizes):
    	self.network = lasagne.layers.InputLayer((None, layer_sizes[0]))
    	self.regularization_layers = []
        for n in layer_sizes[1:-1]:
        	self.network = lasagne.layers.DenseLayer(self.network,
        				           n, nonlinearity=lasagne.nonlinearities.rectify,
        				           W=lasagne.init.GlorotUniform(),
        				           b=lasagne.init.Constant(0.1))
        	self.regularization_layers.append(self.network)
    	self.network = lasagne.layers.DenseLayer(self.network,
    				           layer_sizes[-1], nonlinearity=lasagne.nonlinearities.identity,
    				           W=lasagne.init.GlorotUniform(),
    				           b=lasagne.init.Constant(0.1))
    	self.regularization_layers.append(self.network)
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

    def forward_for_finetuning_batch_stat(self,input):
        return self.forward(input,finetune=True)

    def forward_no_update_batch_stat(self,input,train=True):
        return self.forward(input,train,False)

    def forward(self,input,train=True,update_batch_stat=True,finetune=False):
    	return lasagne.layers.get_output(self.network, inputs=input)
