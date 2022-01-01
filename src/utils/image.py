from lime import lime_image
from skimage.segmentation import mark_boundaries


class Image(object):

    @staticmethod
    def explainer(images, model, image_size):
        output = list()

        for image in images:
            explainer = lime_image.LimeImageExplainer(random_state=42)
            explanation = explainer.explain_instance(
                image,
                model.predict
            )

            image, mask = explanation.get_image_and_mask(
                model.predict(
                    image.reshape((1, image_size[0], image_size[1], 3))
                ).argmax(axis=1)[0],
                positive_only=True,
                hide_rest=False)

            output.append(mark_boundaries(image, mask))

        return output