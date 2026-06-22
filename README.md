[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/OcE5Fe4c)


# Task 1:

Assumptions and Results can be found in the Notebook inside "01-hyperparameters"


# Task 2:

I recorded 3 images of myself, one for each required gesture and created an annotation file containing bounding boxes of all hands. 

To get the bounding boxes i drew boxes around the hands in paint, read off the corner coordinates and normalized them with the total image resolution

"gesture dataset" contains all images and annotations of me and the tutors, i ordered them in the same way as the hagrid dataset so i could use our already existing code for loading annotations

"classifier_training.ipynb" is the notebook i used to train a model for the required gestures

"matrix.ipynb" is the notebook i used to load the tutors and my data and create a confusion matrix of the prediction results

# Task 3:

I used the classifier_training Notebook to train the gesture_recognition.keras model on 4 gestures (like, dislike, stop, peace).

media_control.py loads the pretrained model and starts the webcam to apply its gesture recognition on the users hand gestures.

I limited the camera area that is used for gesture recognition to increase the accuracy of the predictions, it is displayed as a blue frame so the user knows where to place their hand.

It is recommended to use a white background at the recognition area to further increase the prediction accuracy.

The hand gestures are interpreted as follows:

* stop -> START/STOP (2s)
* like -> VOLUME UP (0.5s)
* dislike -> VOLUME DOWN (0.5s)
* peace -> SKIP (3s)
* no gesture -> no action (0s)

Each action has a set cooldown (top right corner) after usage to prevent spam, as a prediction is calculated for every frame.

On the top left corner the user can see the current gesture recognition.

