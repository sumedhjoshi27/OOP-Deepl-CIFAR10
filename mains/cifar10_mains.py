"""
Execution flow for the CIFAR10 model.

"""

#Reproduce results by seeding the random number generator.
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from utils.process_configuration import ConfigurationParameters
from dataset_classes.LoadDataCifar10 import LoadDataCifar10
from model_classes.cifar10CNN_model import CNNCifar10Model
from utils.model_utils import Report
from utils.process_argument import get_args

def main():

    try:

        #capture the command line arguments from the interface script.
        args = get_args()

        #parse the configuration parameters for the cnn model
        config = ConfigurationParameters(args)

    except:
        print('Missing or invalid arguments !')
        exit(0)

    #load the dataset from the library and print the details
    dataset = LoadDataCifar10(config,dataset)

    #construct , build, compile and train the cnn model
    model = CNNCifar10Model(config, dataset)

    #save the model to the disk
    #model.save_model()

    #generate graphs classfication report, confusion matrix
    report = Report(config, model)
    report.plot()
    report.model_classification_report()
    report.plot_confusion_matrix()

if __name__ == '__main__':
    main()
