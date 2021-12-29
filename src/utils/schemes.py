import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


class Scheme(object):

    @staticmethod
    def dataset_info(ds):
        train = ds.get_train_df()
        test = ds.get_test_df()
        val = ds.get_val_df()
        print('Train:')
        print(' - Normal: ' + str(train[train['diagnosis'] == 1].count()['image']))
        print(' - Pneumonia: ' + str(train[train['diagnosis'] == 0].count()['image']))

        print('Test:')
        print(' - Normal: ' + str(test[test['diagnosis'] == 1].count()['image']))
        print(' - Pneumonia: ' + str(test[test['diagnosis'] == 0].count()['image']))

        print('Val:')
        print(' - Normal: ' + str(val[val['diagnosis'] == 1].count()['image']))
        print(' - Pneumonia: ' + str(val[val['diagnosis'] == 0].count()['image']))

    @staticmethod
    def labeled_images(images, diagnosis, size=10):
        class_names = ['No Pneumonia', 'Pneumonia']

        plt.figure(figsize=(10, 10))
        for i in range(size):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap='gray')
            plt.xlabel(class_names[diagnosis[i]])
        plt.show()

    @staticmethod
    def training_graphs(history):
        plt.figure()
        fig, ax = plt.subplots(1, 4, figsize=(20, 3))
        ax = ax.ravel()

        for i, met in enumerate(['precision', 'recall', 'binary_accuracy', 'loss']):
            ax[i].plot(history.history[met])
            ax[i].plot(history.history['val_' + met])
            ax[i].set_title('Model {}'.format(met))
            ax[i].set_xlabel('epochs')
            ax[i].set_ylabel(met)
            ax[i].legend(['train', 'val'])
        plt.show()

    @staticmethod
    def explainer(image, model, image_size):
        explainer = lime_image.LimeImageExplainer(random_state=42)
        explanation = explainer.explain_instance(
            image,
            model.predict
        )

        plt.figure()
        plt.imshow(image)
        image, mask = explanation.get_image_and_mask(
            model.predict(
                image.reshape((1, image_size[0], image_size[1], 3))
            ).argmax(axis=1)[0],
            positive_only=True,
            hide_rest=False)
        plt.imshow(mark_boundaries(image, mask))
        plt.show()

    @staticmethod
    def confusion_matrix(prediction, true):
        cm = confusion_matrix(prediction, true)
        plt.figure()
        plot_confusion_matrix(cm, figsize=(10, 8), hide_ticks=True, cmap=plt.cm.Blues)
        plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=12)
        plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=12)
        plt.show()