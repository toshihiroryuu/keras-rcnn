keras-rcnn (WIP)
================

.. image:: https://travis-ci.org/broadinstitute/keras-rcnn.svg?branch=master
    :target: https://travis-ci.org/broadinstitute/keras-rcnn

.. image::https://codecov.io/gh/broadinstitute/keras-rcnn/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/broadinstitute/keras-rcnn


keras-rcnn is **the** Keras package for region-based convolutional neural
networks.

Status
------

The status of keras-rcnn as of **Wednesday, August 2, 2017**:

We’ve been meeting in the #keras-rcnn channel on the keras.io Slack server. You can join the server by inviting yourself from the following website:

https://keras-slack-autojoin.herokuapp.com/

Work on the region proposal network (RPN) continues. You can track the progress of this feature from the following pull request:

https://github.com/broadinstitute/keras-rcnn/pull/41

We expect to be finished with this feature in a week or two. Contributions and feedback are welcome!

Here’s a brief tutorial:

Load a dataset. I recommend experimenting with the malaria dataset from Hung, et al. that’s provided with the package:

import keras_rcnn.datasets
import keras_rcnn.preprocessing

training, test = keras_rcnn.datasets.malaria.load_data()

generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()

classes = {
    "rbc": 1,
    "not":2
}

generator = generator.flow(training, classes)

Create an RCNN instance:

import keras.layers
import keras_rcnn.models

image = keras.layers.input((448, 448, 3))

model = keras_rcnn.models.RCNN(image, classes=len(classes) + 1)

Specify your preferred optimizer and pass that to the compile method:

optimizer = keras.optimizers.Adam(0.001)

model.compile(optimizer)

Train the model:

model.fit_generator(generator, 256, epochs=32, callbacks=callbacks)

Finally, make a prediction from the trained model:

x = generator.next()[0]

y_anchors, y_deltas, y_proposals, y_scores = model.predict(x)
