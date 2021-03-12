#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
implements the CIFAR10 data load class by inheriting from the LoadData
avstract class

Provides the CIFAR10 dataset
"""

from base_classes.Abstract_loaddata_class import LoadData
import math
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class LoadDatacifar10(LoadData):
    
    def __init__(self,config):
        """
        Constructor to initialize the trainig and testing datasets
        
        :param config: the JSON configuration namespace
        :return none
        :raises none
        """
        
        super().__init__(config)
        return
    
    def load_dataset(self):
        """
        Loads the cifar10 dataset and updates the respective class
        members
        
        :param none
        :return none
        :raises none
        """
        
        #load the dataset from the keras library
        print("loading the dataset from the keras library...")
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = cifar10.load_data()
        
        print("Dataset loaded successfully from the keras library", self.config.config_namespace.exp_name, "\n")
                
        #Reshape the training and testing data
        self.train_data = self.train_data.reshape(-1,
                                                 self.config.config_namespace.image_width,
                                                 self.config.config_namespace.image_height,
                                                 self.config.config_namespace.image_channel)
        
        self.test_data = self.test_data.reshape(-1,
                                                 self.config.config_namespace.image_width,
                                                 self.config.config_namespace.image_height,
                                                 self.config.config_namespace.image_channel)
        
        print("Trainig and testing are reshaped to size: ",
             self.config.config_namespace.image_width,
             self.config.config_namespace.image_height,
             self.config.config_namespace.image_channel, "\n")
        
        return
        
    
    def display_data_elements(self,which_data,label,index):
        """
        Displays data from the cifar10 dataset
        
        :param whichdata: Specifies the dataset to be used (train or test)
        :param index: specified the index to be displayed
        :returns none
        :raises none
        """
        
        #Display a training data element
        if(which_data == "train_data"):
            fig,axes = plt.subplots(10,10,figsize=(20,20))
            axes = axes.ravel()
            for i in range (0,100):
                index = np.random.randint(0,len(self.train_data))
                axes[i].imshow(self.train_data[index,:])
                label_index = int(self.train_labels[index])
                axes[i].set_title(label[label_index], fontsize = 8)
                axes[i].axis("off")
                
            plt.subplots_adjust(hspace=0.4)
            
        elif(which_data = 'test_data'):
            fig,axes = plt.subplots(10,10,figsize=(20,20))
            axes = axes.ravel()
            for i in range (0,100):
                index = np.random.randint(0,len(self.test_data))
                axes[i].imshow(self.test_data[index,:])
                axes[i].axis("off")
                
            plt.subplots_adjust(hspace=0.4)
            
        return
    
    
    def preprocess_data(self):
        """
        Preprocess the data, one-hot encoding, split validation set, normalization
        
        :param none
        :raises none
        :return none
        """
        
        #convert integer data to float
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        
        #Normalize the data
        self.train_data = self.train_data / self.config.config_namespace.image_pixel_size
        self.test_data = self.test_data / self.config.config_namespace.image_pixel_size
        
        #Split the training set into train and validation dataset. 
        self.train_data, self.train_data_val, self.train_labels, self.train_labels_val= train_test_split(self.train_data, self.train_labels, test_size = 0.1, 
                                                                                                           stratify = y_train, random_state = 27)
        
        #Convert the labels to one hot encoding
        self.train_label_one_hot = to_categorical(self.train_labels,10)
        self.train_label_val_one_hot = to_categorical(self.train_labels_val,10)
        self.test_label_one_hot = to_categorical(self.test_labels,10)
        
        print("Training and testing datasets are normalized and their respective class labels are converted to one-hot encoded vector. \n")
        
        return

