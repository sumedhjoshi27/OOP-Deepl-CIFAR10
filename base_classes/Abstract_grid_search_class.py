#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Creates an abstract gridsearch class which defines the hyperparameter optimizatio process

The optimal set of hyperparameters are found for the CNN model
"""

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from base_classes.Abstract_base_class import BaseModel

class GridSearchBase(BaseModel):

    def __init__(self,config,dataset):
        """
        Initalizes the grid search parameters

        :param config: the Json configuration file
        :return none
        :raise none
        """

        self.config = config

        #Dataset
        self.dataset = dataset

        #Scikit learn wrapper
        self.model_wrapper = 0

        #Dictionary of hyperparameters
        self.param_grid = {}

        #Parallel processing option, -1 for true and 1 for false
        self.n_jobs = 1

        #Grid search model
        self.grid = GridSearchCV(self.model_wrapper, self.param_grid)

        #Grid search results
        self.grid_result = 0

        super().__init__(config)
        return

    def create_model(self):
        """
        creates and compile the CNN model
        :param none
        :return none
        :raises NotImplementedError
        """
        #Implement this method in the inherited class
        raise NotImplementedError
