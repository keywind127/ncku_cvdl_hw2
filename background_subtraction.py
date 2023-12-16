from typing import *

import numpy as np

import cv2

import os

class BackgroundSubtractor(object):

    def __init__(self, num_history_frames : int,
                       pixel_difference   : int, *, 
                       detect_shadows     : Optional[ bool     ] = True,
                       frame_blur_funct   : Optional[ Callable ] = None,
            ) -> None:
        
        assert isinstance(num_history_frames, int)

        assert num_history_frames > 0

        assert isinstance(pixel_difference, int)

        assert pixel_difference >= 0

        assert isinstance(detect_shadows, bool)

        assert (frame_blur_funct is None) or (callable(frame_blur_funct))


        self.num_history_frames : int      = num_history_frames

        self.pixel_difference   : int      = pixel_difference

        self.detect_shadows     : bool     = detect_shadows

        self.frame_blur_funct   : Callable = frame_blur_funct


        if (self.frame_blur_funct is None):
            self.frame_blur_funct = lambda x : cv2.GaussianBlur(x, (5, 5), 0)


        self.background_subtractor = cv2.createBackgroundSubtractorKNN(
            self.num_history_frames, self.pixel_difference, self.detect_shadows
        )


    def process_frame(self, frame : np.ndarray) -> Tuple[ np.ndarray, np.ndarray ]:

        assert isinstance(frame, np.ndarray)

        assert np.prod(frame.shape) > 0

        blurred_frame   : np.ndarray = self.frame_blur_funct(frame)

        foreground_mask : np.ndarray = self.background_subtractor.apply(blurred_frame)

        result_frame    : np.ndarray = cv2.bitwise_and(frame, frame, mask = foreground_mask)

        # fg_mask, fg_frame
        return (foreground_mask, result_frame)


if (__name__ == "__main__"):

    filename = os.path.join(os.path.dirname(__file__), "data/Q1/traffic.mp4")


    history_frames = 100

    min_distance   = 200

    detect_shadows = True


    background_subtractor = BackgroundSubtractor(
        history_frames, min_distance, detect_shadows = detect_shadows
    )


    cap = cv2.VideoCapture(filename)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        (fg_mask, result_frame) = background_subtractor.process_frame(frame)

        cv2.imshow('Original Frame', frame)

        cv2.imshow('Foreground Mask', fg_mask)

        cv2.imshow('Result Frame', result_frame)

        if (cv2.waitKey(30) & 0xFF == ord('q')):
            break


    cap.release()

    cv2.destroyAllWindows()