from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pb')


class TLClassifier(object):
    def __init__(self):
        # load classifier
        self.graph = tf.Graph()
        self.threshold = 0.5

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # DONE: implement light color prediction
        with self.graph.as_default():
            img = np.expand_dims(image, axis=0)
            _, scores, classes, _ = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        if scores[0] > self.threshold:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
