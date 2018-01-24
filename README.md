# Image-Classification

This project attempts to classify images using pre-trained models.

2 sets of approaches are used

1. Pre-trained models as Feature extractors: Use the pre-trained models as a feature extractor and create predictions by majority vote from all the models
2. Fine-Tuning the pre-trained models: Remove the last layer from pre-trained network and train the model on the labels which are of interest. Run the model on a few rounds. Save the weights . Now fine-tune the higher layer weights of the pre-trained models and get the final classification results

Note: These approaches do not use image augmentation techniques at the moment. This can be a later exercise
