#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""

Defines an abstract class BaseModel, that wraps the CNN model construction
process.

It defined, configures, trains and evaluates the CNN model. Also supports
prediction.

"""

import numpy as np
import os
from keras.models import Sequential
from keras.callbacks import History, ModelCheckpoint

class BaseModel(object):
    def __init__(self,config,dataset):

        """
        Constructor to initailize the CNN parameters

        :param config: the JSON config namespace
        :param dataset: the training and testing dataset
        :return none
        :raises none
        """

        #Configuration parameters
        self.config = config

        #train and test dataset
        self.dataset = dataset

        #CNN model
        self.cnn_model = Sequential()

        #History object
        self.history = History()

        #save model path
        self.saved_model_path = os.path.join(self.config.config_namespace.saved_model_dir, "xyz")

        #checkpoint for model
        self.checkpoint = ModelCheckpoint(self.saved_model_path,
                                         monitor = 'val_acc',
                                         verbose = self.config.config_namespace.checkpoint_verbose,
                                         save_best_only = True,
                                         mode = 'max')

        #callbacks list
        self.callbacks_list = [self.checkpoint]

        #score evaluation
        self.scores = []

        #training time
        self.train_time = 0

        #Predicted class labels
        self.predictions = np.array([])

        #validate the stride size
        self.validate_stride()

        #construct the CNN model
        self.define_model()

        #configure the CNN model
        self.compile_model()

        #Train the CNN model using testing dataset
        self.fit_model()

        #evaluate the CNN model using testing dataset
        self.evaluate_model()

        #Predict the class labels of testing dataset
        self.predict()

        return

    def calculate_number_of_filters(self):
        """
        Calculates the filter size for a given layer

        :param none
        :return none
        :raises NotImplementedError
        """
        #Implement this method in the inherited class to calculate filter size
        raise NotImplementError

    def validate_stride(self):
        """
        Validate the stride based on the input data's size, filter's size and
        padding specified.

        :param none
        :return none
        :raises Exception: Invalid stride size.
        """

        valid_stride_width = (
                              self.config.config_namespace.image_width - self.config.config_namespace.kernel_row +
                              2_self.config.config_namespace.padding_size
                             ) / self.config.config_namespace.stride_size + 1


        valid_stride_height = (
                              self.config.config_namespace.image_height - self.config.config_namespace.kernel_coloumn +
                              2_self.config.config_namespace.padding_size
                             ) / self.config.config_namespace.stride_size + 1

        if(not float(valid_stride_width).is_integer()
               and
           not float(valid_stride_height).is_integer()
          ):

            print("Invalid stride size specified, model does not fit into \\
                  specification. !")
            raise Exception

        else:
            return


    def define_model(self):
        """
        Constructs the CNN model

        :param none]
        :return none
        :raises NotImplementedError
        """

        #Implement this method in the inherited class to calculate filter size
        raise NotImplementedError


    def compile_model(self):
        """
        Complies the CNN model

        :param none
        :return none
        :raises NotImplementedError
        """

        #Implement this method in the inherited class to calculate filter size
        raise NotImplementedError


    def fit_model(self):
        """
        Trains the CNN model

        :param none
        :return none
        :raises NotImplementedError
        """

        #Implement this method in the inherited class to calculate filter size
        raise NotImplementedError


    def evaluate_model(self):
        """
        Evaluates the CNN model

        :param none
        :return none
        :raises NotImplementedError
        """

        #Implement this method in the inherited class to calculate filter size
        raise NotImplementedError


    def predict(self):
        """
        predicts class labels for the CNN model

        :param none
        :return none
        :raises NotImplementedError
        """

        #Implement this method in the inherited class to calculate filter size
        raise NotImplementedError


    def save_model(self):
        """
        saves the CNN model in h5 format

        :param none
        :return none
        """

        if(self.cnn_model is None):
            raise Exception("CNN model not configured and trained")

        self.cnn_model.save(self.saved_model_path)
        print("CNN model save at path:", self.saved_model_path, "\n")

        return


    def load_cnn_model(self):
        """
        Loads the saved model from the disk.

        :param none
        :return none
        """

        if( self.cnn_model is None ):
            raise Exception("ConvNet model not configured and trained !")

        self.cnn_model.load_weights( self.saved_model_path )
        print("ConvNet model loaded from the path: ", self.saved_model_path, "\n")

        return
