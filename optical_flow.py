from typing import *

import numpy as np

import cv2 

import os

def draw_cross(frame     : np.ndarray, 
               coord     : Tuple[ int, int ], 
               radius    : int, *,
               thickness : Optional[ int   ] = 5,
               color     : Optional[ tuple ] = (0, 0, 255)
                    
            ) -> np.ndarray:

    assert isinstance(frame, np.ndarray)

    assert isinstance(coord, tuple)

    assert isinstance(radius, int)

    assert radius > 0

    assert np.prod(frame.shape) > 0

    assert isinstance(thickness, int)

    assert isinstance(color, tuple)

    assert thickness > 0

    start_x = max(coord[0] - radius, 0)

    start_y = max(coord[1] - radius, 0)

    end_x = min(coord[0] + radius, frame.shape[1] - 1)

    end_y = min(coord[1] + radius, frame.shape[0] - 1)

    frame = cv2.line(
        img       = frame, 
        pt1       = (start_x, coord[1]), 
        pt2       = (end_x, coord[1]), 
        color     = color, 
        thickness = thickness
    )

    frame = cv2.line(
        img       = frame, 
        pt1       = (coord[0], start_y), 
        pt2       = (coord[0], end_y), 
        color     = color, 
        thickness = thickness
    )

    return frame

def find_good_corners(frame         : np.ndarray, *,
                      max_corners   : Optional[   int ] = 1, 
                      quality_level : Optional[ float ] = 0.3, 
                      min_distance  : Optional[   int ] = 7, 
                      block_size    : Optional[   int ] = 7
            ) -> Union[ np.ndarray, None ]:
    
    assert isinstance(frame, np.ndarray)

    assert isinstance(max_corners, int)

    assert isinstance(quality_level, float)

    assert isinstance(min_distance, int)

    assert isinstance(block_size, int)

    assert np.prod(frame.shape) > 0

    assert max_corners > 0

    assert quality_level >= 0

    assert min_distance >= 0

    assert block_size > 0


    corners = []

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _found_corners = cv2.goodFeaturesToTrack(
        image        = gray_image, 
        maxCorners   = max_corners, 
        qualityLevel = quality_level, 
        minDistance  = min_distance, 
        blockSize    = block_size
    )

    found_corners = np.intp(_found_corners)

    for corner in found_corners:
        corners.append(corner.ravel())

    if (len(corners) == 0):
        return None

    return np.stack(corners, dtype = np.int32), _found_corners


def draw_history_points(frame          : np.ndarray, 
                        history_points : List[ Tuple[ int, int ] ], *,
                        color          : Optional[ Tuple[ int, int, int ] ] = (255, 0, 0),
                        thickness      : Optional[ int ]                    = 5
                        
            ) -> np.ndarray:

    assert isinstance(frame, np.ndarray)

    assert isinstance(history_points, list)

    assert isinstance(color, tuple)

    assert isinstance(thickness, int)

    assert thickness > 0

    assert np.prod(frame.shape) > 0

    output_frame = frame.copy()

    for i in range(1, len(history_points)):

        output_frame = cv2.line(
            img       = output_frame, 
            pt1       = history_points[i - 1], 
            pt2       = history_points[i], 
            color     = color,
            thickness = thickness
        )

    return output_frame


def resize_to_width(frame : np.ndarray, target_width : int) -> np.ndarray:

    assert isinstance(frame, np.ndarray)

    assert isinstance(target_width, int)

    assert np.prod(frame.shape) > 0

    return cv2.resize(frame, None, None, fx = 512 / frame.shape[1], fy = 512 / frame.shape[1])

def predict_next_corner(frame_1 : np.ndarray, 
                        frame_2 : np.ndarray, 
                        corners : List[ tuple ]) -> tuple:

    assert isinstance(frame_1, np.ndarray)

    assert isinstance(frame_2, np.ndarray)

    # print(corners.shape)

    corners_next, status, err = cv2.calcOpticalFlowPyrLK(
        frame_1, frame_2, corners, None, winSize = (15, 15), maxLevel = 1,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.003)
    )

    # print(status)

    return np.stack([ np.intp(corners_next).ravel() ]), corners_next.reshape((-1, 1, 2))


if (__name__ == "__main__"):

    filename = os.path.join(os.path.dirname(__file__), "data/Q2/optical_flow.mp4")

    cap = cv2.VideoCapture(filename)

    history_points = []

    corner = None

    corners = None

    prev_frame = None

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if (corners is None):

            corner, corners = find_good_corners(frame)
            prev_frame = frame.copy()

        else:

            # assert prev_frame is not frame

            corner, corners = predict_next_corner(
                frame_1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                frame_2 = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY),
                corners = corners
            )

            prev_frame = frame.copy()

        print(corner)

        if (corner is None):
            continue

        corner = (x, y) = tuple(corner[0])

        frame = draw_cross(frame, corner, radius = 30)

        history_points.append(corner)
        
        frame = resize_to_width(draw_history_points(frame, history_points), 512)

        cv2.imshow('Corners', frame)

        if (cv2.waitKey(30) & 0xFF == ord('q')):
            break

    cap.release()

    cv2.destroyAllWindows()