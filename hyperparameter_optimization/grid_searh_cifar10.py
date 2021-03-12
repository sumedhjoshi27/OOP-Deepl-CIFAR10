"""
Implements grid search for a optimal set of hyperparameters to train the CNN
model
"""

from base_classes.Abstract_grid_search_class import GridSearchBase
from model_classes.cifar10CNN_model import CNNCifar10Model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from dataset_classes.LoadDataCifar10 import LoadDataCifar10
from utils.process_argument import get_args
from utils.process_configuration import ConfigurationParameters

def create_model(self):
    """
    creates and compiles the CNN model_classes
    :param ConfigurationParameters
    :return CNN_model
    :raises none
    """

    model = CNNCifar10Model(self.config, self.dataset)
    cnn_cifar10 = model.define()
    return cnn_cifar10

def main():

    try:

        args = get_args()

        #Parse the configuration ConfigurationParameters
        config = ConfigurationParameters(args)

    except:

        print('Missing or invalid arguments !')
        exit(0)

    dataset = LoadDataCifar10(config)

    g_search = GridSearchBase(config,dataset)

    #create a Scikit learn wrapper
    g_search.model_wrapper = KerasClassifier(build_fn = g_search.create_model, verbose=0)

    #define the grid search parameters
    batch_size = [1,2]
    epochs = [30,50]
    g_search.param_grid = dict(batch_size = batch_size,
                                   epochs = epochs)

    g_search.grid = GridSearchCV(estimator=g_search.model_wrapper,\\
                                     param_grid = g_search.param_grid,\\
                                     n_jobs = g_search.n_jobs)

    g_search.grid_result = g_search.grid.fit(dataset.train_data,\\
                                                 dataset.train_label_one_hot)

    #summarize results
    print("best :%f using %s" %(g_search.grid_result.best_score_,\\
                                    g_search.grid_result.best_params_))
    means = g_search.grid_result.cv_results_['mean_test_score']
    stds = g_search.grid_result.cv_results_['std_test_score']
    params = g_search.grid_result.cv_results_['params']

    for mean,stdev,param in zip(means, stds, params):
        print("%f (%f) with: %r" %(mean, stdev, param))


if __name__ == '__main__':
    main()
