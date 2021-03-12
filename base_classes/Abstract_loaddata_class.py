#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Defines an abstract class LoadData, that wraps the data loading process.
Dataset can either be present in the library or on the disk.

It also supports dataset printing options.
"""

import numpy as np

class LoadData(object):

    def __init__(self,config):
        """
        Constructor to initialize the training, testing datasets and their
        properties.

        :param config: the JSON configuration namespace.
        :return none
        :raises none
        """

        #Configuration parameters.
        self.config = config

        #Training dataset data and labels
        self.train_data = np.array([])
        self.train_labels = np.array([])

        self.train_data_val = np.array([])
        self.train_labels_val = np.array([])

        #Testing dataset's data and labels
        self.test_data = np.array([])
        self.test_labels = np.array([])

        #Total number of class labels
        self.no_of_classes = 0

        #class label list
        self.list_of_classes = []

        #One-hot encoded label vector
        self.train_label_one_hot = np.array([])
        self.train_label_val_one_hot = np.array([])
        self.test_label_one_hot = np.array([])

        #Load the dataset from disk/library
        self.load_dataset()

        #calculate the number of class labels and list them
        self.calculate_class_label_size()

        #print the details of the dataset
        self.print_dataset_details()

        #Preprocess the dataset
        self.preprocess_dataset()
        return

    def load_dataset(self):
        """
        Loads the dataset

        :param none
        :return none
        :raises NotImplementedError
        """

        raise NotImplementedError


    def print_dataset_details(self):
        """
        Prints the dataset

        :param none
        :return none
        :raises none
        """

        #Number of samples in dataset
        print("Training dataset size (Data,Labels) is: ", \\
               self.train_data.shape, self.train_labels.shape)
        print("Testing dataset size (Data, Labels) is: ", \\
               self.test_data.shape, self.test_labels.shape)

        #Number of class labels and their list
        print("Total number of classes n the dataset: ", self.no_of_classes)
        print("The ", self.no_of_classes, "Classes of the dataset are: ", \\
               self.list_of_classes)

        return


    def calculate_class_label_size(self):
        """
        Calculates the total number of classes in dataset

        :param none
        :return none
        """
        self.list_of_classes = np.unique(self.train_labels)
        self.no_of_classes = len(self.list_of_classes)
        print("Number of classes and list from the loaded dataset")

        return


    def display_data_elements(self, which_data, label, index):
        """
        Displays a data element from a particular dataset (training/testing)

        :param none
        :return none
        :raises NotImplementedError
        """

        # Implement this method in the inherited class to display a given
        # data element.
        raise NotImplementedError


    def preprocess_dataset(self, which_data, index):
        """
        DPreprocess the dataset

        :param none
        :return none
        :raises NotImplementedError
        """

        # Implement this method in the inherited class to preprocess the dataset.
        raise NotImplementedError
