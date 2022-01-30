from lime import lime_image
import numpy as np
from skimage.segmentation import mark_boundaries


class Image(object):

    @staticmethod
    def explain(images, model, image_size):
        output = list()

        for image in images:
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(image.astype('double'), model.predict, top_labels=5, hide_color=0,
                                                     num_samples=1000)
            image, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5,
                                                        hide_rest=False)

            output.append(mark_boundaries(image, mask))

        return output
