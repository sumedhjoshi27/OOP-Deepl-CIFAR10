"""
Utility to generate classification report, confusion matrix and
graph for loss and accuracy for the cnn model
"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import os

class Report():

    def __init__(self,config, model):
        """
        Constructor to initialize the report.

        :param config: the json configuration namespace
        :param model: the cnn model
        :return none
        :raises none
        """

        self.config = config
        self.model = model

        return

    def plot(self):
        """
        plot the accuracy for the training and validation set of cnn model
        :param none
        :return none
        :raises none
        """

        loss_list = [s for s in self.model.history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in self.model.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in self.model.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in self.model.history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return

        #As loss always exists
        epochs = range(1, len(self.model.history.history[loss_list[0]]) + 1)

        #loss graph
        plt.figure(1)

        for l in loss_list:
            plt.plot(epochs,self.model.history.history[1],
                     'b', label = 'Training loss (' +str(str(format(self.model.history.history[l][-1],'.5f')))')')

        for l in val_loss_list:
            plt.plot(epochs,self.model.history.history[1],
                     'g', label = 'Validation loss (' +str(str(format(self.model.history.history[l][-1],'.5f')))')')

        plt.title('Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()

        #save the plots to disk
        loss_path = os.path.join(self.config.config_namespace.grpah_dir, "loss.png")

        if(self.config.config_namespace.save_plots = 'true'):
            plt.savefig(loss_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        #Accuracy graph
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, self.model.history.history[l],
                     'b',
                     label = 'Training accuracy (' +str(str(format(self.model.history.history[l][-1],'.5f')))')')

        for l in val_list:
            plt.plot(epochs, self.model.history.history[l],
                     'g',
                     label = 'Validation accuracy (' +str(str(format(self.model.history.history[l][-1],'.5f')))')')

        plt.title('Accuracy per epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        #save the plots to disk
        loss_path = os.path.join(self.config.config_namespace.grpah_dir, "accuracy.png")

        if(self.config.config_namespace.save_plots = 'true'):
            plt.savefig(loss_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        def model_classification_report(self):
            """
            Generate classification report of the cnn models
            :param none
            :return none
            :raise none
            """
            predicted_classes = np.argmax(self.model.predictions, axis = 1)
            print('Accuracy: ' + str(accuracy_score(self.model.dataset.test_labels, predicted_classes)))
            print('Classification report')
            print('-----------------------------------------------------------')
            target_names = ['Class {}'.foramt(i) for i in range(self.config.config_namespace.num_classes)]
            print(classification_report(self.model.dataset.test_labels,\\
                                        predicted_classes,
                                        target_names = target_names))

            return


        def plot_confusion_matrix(self):
            """
            Generate and plot the classification confusion matrix

            :param none
            :return none
            :raises none
            """

            predicted_classes = np.argmax(self.model.predictions, axis = 1)
            target_names = ['Class {}'.format(i) for i in range(self.config.config_namespace.num_classes)]
            title = 'Confusion matrix'
            cm = confusion_matrix(self.model.dataset.test_labels, predicted_classes)
            print(title)
            print('-----------------------------------------------------------')
            print(cm)

            plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

            fmt = 'd'
            thresh = cm.max()/2

            for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

                plt.text(j, i, format( cm[i, j], fmt ),
                         horizontalalignment = 'center',
					     color = 'white' if cm[i, j] > thresh else 'black')

                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')

            #save the plots to disk
            loss_path = os.path.join(self.config.config_namespace.grpah_dir, "confusion_matrix.png")

            if(self.config.config_namespace.save_plots = 'true'):
                plt.savefig(loss_path, bbox_inches='tight')
            else:
                plt.show()

            plt.close()
