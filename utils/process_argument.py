"""
Process the json configuration files
configuration file holds the parameters to initialize the cnn model.
These fiels are located in the configuration file folder

"""

import argparse

def get_args():
    """
    Get arguments from the command line

    :param none
    :return none
    :raises none
    """

    parser = argparse.ArgumentParser(description = __doc__ )

    #Configuration file path argument
    parser.add_argument('-c, --config',
                         metvar = 'C',
                         help = 'The configuration file',
                         default = './configuration_files/cifar10.json',
                         required = False)


    #Epoch size argument
    parser.add_argument('-e, --epoch',
                         metvar = 'E',
                         help = 'Number of epochs for training the model',
                         default = 1,
                         required = False)


    #convert to dictionary
    args = vars(parser.parse_args())

    if(args['config'] == './configuration_files/fashion_config.json'):
        print('Using default configuration file.')

    else:
        print('Using configuration from file:', args['config'])

    if(args['epoch'] == 1):
        print('Using default epoch size of 1')

    else:
        print('Using epoch size: ' args['epoch'])

    return args 
