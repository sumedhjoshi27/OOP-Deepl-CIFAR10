"""
Parse the JSON configuration file of the experiment

configuation file holds the parameters to initalize the cnn model
These files are located in the configuration files folder
"""

import json
import os
from bunch import Bunch

class ConfigurationParameters:

    def __init__(self, args):
        """
        Initialize the data members

        :param  json_file Path to the JSON configuration files
        :return none
        :raises none
        """

        self.args = args

        json_file = self.args['config']

        #Parse the configuration from the config json provided
        with open(json_file, 'r') as config_file:
            self.config_dictionary = json.load(config_file)

        #Convert the dictionary to a namespace using bunch library
        self.config_namespace = Bunch(self.config_dictionary)

        #update the command line args in the configuration config_namespace
        self.update_namespace()

        #process the configuration Configuration parameters
        self.process_config()

        return


    def update_namespace(self):
        """
        Updates the value of the json keys recieved from the command line to
        the namespace files

        :param none
        :return none
        :raises none
        """

        #update epoch size
        if 'epoch' in self.args.keys():
            self.config_namespace.num_epochs = int(self.args['epoch'])

        return


    def process_config(self):
        """
        :param none
        :return none
        :raises none
        """

        #saved model directory
        self.config_namespace.saved_model_dir = os.path.join("./home/anaconda3/envs/naska_env/OOP-Deepl-CIFAR10",
                                                       self.config_namespace.exp_name, "saved_images/")

        #Graph directory
        self.config_namespace.graph_dir = os.path.join("./home/anaconda3/envs/naska_env/OOP-Deepl-CIFAR10",
                                                       self.config_namespace.exp_name, "saved_images/")


        #Image directory
        self.config_namespace.image_dir = os.path.join("./home/anaconda3/envs/naska_env/OOP-Deepl-CIFAR10",
                                                       self.config_namespace.exp_name, "saved_images/")

        #Create the above directories
        self.create_dirs([self.config_namespace.graph_dir, self.config_namespace.image_dir,
                          self.config_namespace.saved_model_dir])

        return


    def create_dirs(self,dirs):
        """
        creates a directory structure for graphs and images generated during
        the run of the experiment.
        :param none
        :return exit_code: 0:succes -1:failed
        :raises none
        """

        try:

            for d in dirs:
                if not os.path.exists(d):
                    os.makedirs(d)

            return 0

        except Exception as err:
            print("creating directories error: (0)".format(err))
            exit(-1)
