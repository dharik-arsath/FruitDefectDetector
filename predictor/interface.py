from typing import Callable

import keras
import tensorflow as tf
import numpy as np
import cv2
from typing import Protocol


class BaseInfer(Protocol):
    def load_model(self):
        ...

    def infer(self, frame):
        ...


class Infer(BaseInfer):
    def __init__(self, path_to_model: str):
        self.model = None
        self.path_to_model = path_to_model
        self.load_model()

    def load_model(self):
        self.model = tf.keras.models.load_model(self.path_to_model)
        # self.model = model.model
    def infer(self, frame):
        pred_proba  = self.model.predict(frame)
        predictions = np.argmax(pred_proba)

        return predictions, pred_proba


class UIInterface:
    def __init__(self, infer_obj: BaseInfer, camera_number: int = 0):
        self.Infer = infer_obj
        self.camera_number = camera_number
        self.vid = cv2.VideoCapture(self.camera_number)


    def _read_test_frame(self):
        image = tf.io.read_file("C:\\Users\\dhari\\.keras\\datasets\\image.jpg")
        frame = tf.image.decode_jpeg(image)

        return frame
    def _read_frame(self):
        # define a video capture object
        ret, frame = self.vid.read()
        if ret is False:
            raise Exception("Camera Could not be accessed...")

        if frame is None:
            raise Exception("Frame Cannot be determined, see if camera is being accessed...")

        return frame

    def _destroyCameraInstance(self) -> None:
        self.vid.release()
        cv2.destroyAllWindows()

    def make_live_inference(self):
        try:
            while True:
                # frame = self._read_test_frame()
                frame = self._read_frame()
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                img = cv2.resize(img, (256, 256))  # Resize the image to match the input size of the model
                img = np.expand_dims(img, axis=0)  # Add a batch dimension to the image tensor

                pred, pred_proba = self.Infer.infer(img)
                max_indx = np.argmax(pred_proba)
                if  pred_proba[:, max_indx] < 0.8:
                    continue
                print(pred, pred_proba)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self._destroyCameraInstance()


infer = Infer(path_to_model="current_model_binary_classV3_regularizedV1.h5")

cvInter = UIInterface(infer)

cvInter.make_live_inference()

