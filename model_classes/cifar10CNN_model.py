#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/anaconda3/envs/naska_env python3

"""
CNN model for CIFAR10 dataset.

Created on Tue March 9, 2021

@author: Sumedh Joshi
@version: 1.0
"""
import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras import applications
from tensorflow.python.framework import ops
from base_classes.Abstract_base_class import BaseModel
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import time


class CNNCifar10Model(BaseModel):

    def __init__(self,config,dataset):

        """
        constructor to initalize the CNN for CIFAR10 dataset

        :param config: the JSON configuration namespace.
        :param dataset: Training and testing datasets.
        :return none
        :raises none

        """
        super().__init__(config,dataset)
        return

    def define_model(self):
        """
        Construct the CNN model.

        :param none
        :return none
        :raises none

        """
        if (self.config.config_namespace.model_type == 'Sequential'):
            print("The Keras CNN model type used is : ", self.config.config_namespace.model_type)
            self.cnn_model = self.define_seq_model()

        else:
            #TODO: handle functional model here
            self.cnn_model.summary()
        return

    def define_seq_model(self):
        """
        Design a sequential CNN model.

        :param none
        :return cnn_model: The CNN seq model
        :raises none

        """
        self.cnn_model = Sequential()

        # 1st Layer
        self.cnn_model.add(Conv2D(filters=self.config.config_namespace.no_of_filters_l1,
                               kernel_size = (self.config.config_namespace.kernel_row,self.config.config_namespace.kernel_column),
                               activation = self.config.config_namespace.conv_activation_l1,
                               input_shape = (self.config.config_namespace.image_width, self.config.config_namespace.image_height),
                               padding = self.config.config_namespace.padding,
                               strides = self.config.config_namespace.stride_size
                                 )
                          )

        self.cnn_model.add(BatchNormalization())

        # 2nd Layer
        self.cnn_model.add(Conv2D(filters=self.config.config_namespace.no_of_filters_l2,
                               kernel_size = (self.config.config_namespace.kernel_row,self.config.config_namespace.kernel_column),
                               activation = self.config.config_namespace.conv_activation_l2,
                               padding = self.config.config_namespace.padding,
                               strides = self.config.config_namespace.stride_size
                                 )
                          )

        self.cnn_model.add(BatchNormalization())

        # 3rd Layer
        self.cnn_model.add(MaxPooling2D(pool_size = (self.config.config_namespace.pool_size_row,
                                                    padding = self.config.config_namespace.padding)))

        #Add droput layer if necessary
        if(self.config.config_namespace.dropout == 'true'):
            self.cnn_model.add(Dropout(self.config.config_namespace.dropout_probability_l1))

        # 4th Layer
        self.cnn_model.add(Conv2D(filters=self.config.config_namespace.no_of_filters_l3,
                               kernel_size = (self.config.config_namespace.kernel_row,self.config.config_namespace.kernel_column),
                               activation = self.config.config_namespace.conv_activation_l3,
                               padding = self.config.config_namespace.padding,
                               strides = self.config.config_namespace.stride_size
                                 )
                          )

        self.cnn_model.add(BatchNormalization())

        # 5nd Layer
        self.cnn_model.add(Conv2D(filters=self.config.config_namespace.no_of_filters_l4,
                               kernel_size = (self.config.config_namespace.kernel_row,self.config.config_namespace.kernel_column),
                               activation = self.config.config_namespace.conv_activation_l4,
                               padding = self.config.config_namespace.padding,
                               strides = self.config.config_namespace.stride_size
                                 )
                          )

        self.cnn_model.add(BatchNormalization())

        # 6th Layer
        self.cnn_model.add(MaxPooling2D(pool_size = (self.config.config_namespace.pool_size_row,
                                                    padding = self.config.config_namespace.padding)))

        #Add droput layer if necessary
        if(self.config.config_namespace.dropout == 'true'):
            self.cnn_model.add(Dropout(self.config.config_namespace.dropout_probability_l2))

        #7th Layer
        self.cnn_model.add(Flatten())

        #8th Layer
        self.cnn_model.add(Dense(units = self.config.config_namespace.no_of_filters_l4,
                                activation = self.config.config_namespace.dense_activation_l1))

        #Add droput layer if necessary
        if(self.config.config_namespace.dropout == 'true'):
            self.cnn_model.add(Dropout(self.config.config_namespace.dropout_probability_l3))

        #9th Layer
        self.cnn_model.add(Dense(self.datasets.no_of_classes,
                                activation = self.config.config_namespace.dense_activation_l2))


        return self.cnn_model


    def compile_model(self):
        """
        Configue the cnn model.

        :param none
        :return none
        :raises none

        """

        start.time = time.time()

        if(self.config.config_namespace.save_model = 'true'):
            print("Training phase under progress, trained model will be saved at path", self.saved_model_path, " ...\n")
            self.history = self.cnn_model.fit(x = self.dataset_train_data,
                                             y = self.dataset.train_label_one_hot,
                                             batch_size = self.config.config_namespace.batch_size,
                                             epochs = self.config.config_namespace.num_epochs,
                                             callbacks = self.callbacks_list,
                                             verbose = self.config.config_namespace.fit_verbose,
                                             validation_data = (self.dataset.test_data, self.dataset.test_label_one_hot))

        else:
            print("Training phase under progress...\n")
            self.history = self.cnn_model.fit(x = self.dataset_train_data,
                                             y = self.dataset.train_label_one_hot,
                                             batch_size = self.config.config_namespace.batch_size,
                                             epochs = self.config.config_namespace.num_epochs,
                                             callbacks = self.callbacks_list,
                                             verbose = self.config.config_namespace.fit_verbose,
                                             validation_data = (self.dataset.test_data, self.dataset.test_label_one_hot))
        end_time = time.time()

        self.train_time = end_time - start_time
        print("The model took %0.3f seconds to train. \n"%self.train_time)

        return

    def evaluate_model(self):
        """
        Evaluate the cnn model.

        :param none
        :return none
        :raises none
        """

        self.scores = self.cnn_model.evaluate(x = self.dataset.test_data,
                                             y = self.dataset.test_label_one_hot,
                                             verbose = self.config.config_namespace.evaluate_verbose)

        print("Test loss: ", self.scores[0])
        print("Test accuracy: ", self.scores[1])

        return

    def predict(self):

        """
        Predict the class label of the training dataset

        :param none
        :return none
        :raises none

        """

        self.predictions = self.cnn_model.predict(x = self.dataset.test_data,
                                                 verbose = self.config.config_namespace.predict_verbose)

        return
