from typing import *
import numpy, cv2, os
import itertools

class CameraParams(dict):
    def __init__(self, params : Dict[ str, numpy.ndarray ]) -> None:
        super(CameraParams, self).__init__()
        self.update(params)

def files_in_folder(folder_name : str) -> List[ str ]:

    assert isinstance(folder_name, str)

    assert os.path.exists(folder_name)

    return list(filter(lambda x : ((os.path.dirname(x) == folder_name) and os.path.isfile(x)), map(lambda x : os.path.join(folder_name, x), os.listdir(folder_name))))

def load_images(image_files : List[ str ], preprocess : Optional[ Callable ] = None) -> Iterator[ numpy.ndarray ]:

    # print(image_files)

    assert (isinstance(image_files, list) or isinstance(image_files, tuple))

    for image_name in image_files:
        image = cv2.imread(image_name)
        if (preprocess is not None):
            image = preprocess(image)
        yield image

def to_grayscale(image : numpy.ndarray) -> numpy.ndarray:

    assert isinstance(image, numpy.ndarray)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def find_chessboard_corners(images : List[ numpy.ndarray ], chessboard_size : Tuple[ int, int ]) -> numpy.ndarray:

    assert (isinstance(images, list) or isinstance(images, numpy.ndarray))

    assert isinstance(chessboard_size, tuple)

    corner_coordinates = []

    for image in images:

        image_shape = image.shape[0:2][::-1]

        status_success, corner_coords = cv2.findChessboardCorners(image, chessboard_size, None)

        if not (status_success):
            continue

        corner_coords = cv2.cornerSubPix(image, corner_coords, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        corner_coordinates.append(corner_coords[:,0,:])

    return numpy.array(corner_coordinates, dtype = numpy.float32)

def mark_coordinates(images : List[ numpy.ndarray ], coordinates : numpy.ndarray) -> List[ numpy.ndarray ]:

    assert (isinstance(images, list) or isinstance(images, numpy.ndarray))

    assert isinstance(coordinates, numpy.ndarray)

    marked_images = []

    for idx, image in enumerate(images):

        target_coords = coordinates[idx]

        marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for cidx, target_coord in enumerate(target_coords):

            # if (cidx >= 11):
            #     break

            marked_image = cv2.circle(marked_image, numpy.int32(target_coord), radius = 10, color = (0, 0, 255), thickness = 5)

        marked_images.append(marked_image)

    return marked_images

def generate_object_coords(grid_size : Tuple[ int, int ], num_images : Optional[ int ] = 1) -> numpy.ndarray:

    assert isinstance(grid_size, tuple)

    assert (isinstance(num_images, int) and (num_images > 0))

    (y, x) = grid_size

    assert (isinstance(y, int) and (y > 0))

    assert (isinstance(x, int) and (x > 0))

    return numpy.array([  [  (_x, _y, 0) for _x, _y in itertools.product(range(x), range(y))  ] for _ in range(num_images)  ], dtype = numpy.float32)

def obtain_camera_parameters(object_coords : numpy.ndarray,
                             images_coords : numpy.ndarray,
                             image_shape   : Tuple[ int, int ]
        ) -> Union[ Dict[ str, numpy.ndarray ], CameraParams ]:

    assert isinstance(object_coords, numpy.ndarray)

    assert isinstance(images_coords, numpy.ndarray)

    assert isinstance(image_shape, tuple)

    (status_success, i_matrix, d_matrix, r_vector, t_vector) = cv2.calibrateCamera(object_coords, images_coords, image_shape, None, None)

    if not (status_success):
        return None

    return CameraParams({
        "i_matrix" : i_matrix, # intrinsic
        "d_matrix" : d_matrix, # distortion
        "r_vector" : r_vector, # rotation
        "t_vector" : t_vector, # translation
    })

def obtain_image_vectors(images        : List[ numpy.ndarray ],
                         object_coords : numpy.ndarray,
                         images_coords : numpy.ndarray,
                         image_shape   : Tuple[ int, int ],
                         camera_params : CameraParams
        ) -> Iterator[ CameraParams ]:

    assert (isinstance(images, list) or isinstance(images, numpy.ndarray))

    assert isinstance(object_coords, numpy.ndarray)

    assert isinstance(images_coords, numpy.ndarray)

    assert isinstance(image_shape, tuple)

    assert isinstance(camera_params, CameraParams)

    for idx, image in enumerate(images):

        status_success, r_vector, t_vector = cv2.solvePnP(
            object_coords[idx],
            images_coords[idx],
            camera_params["i_matrix"],
            camera_params["d_matrix"]
        )

        if status_success:
            e_matrix = numpy.hstack((cv2.Rodrigues(r_vector)[0], t_vector))

        yield CameraParams({
            "r_vector" : r_vector, # rotation vector
            "t_vector" : t_vector, # translation vector
            "e_matrix" : e_matrix, # extrinsic matrix
        })

def undistort_images(images        : List[ numpy.ndarray ],
                     object_coords : numpy.ndarray,
                     images_coords : numpy.ndarray,
                     image_shape   : Tuple[ int, int ],
                     camera_params : CameraParams
        ) -> Iterator[ numpy.ndarray ]:

    assert (isinstance(images, list) or isinstance(images, numpy.ndarray))

    assert isinstance(object_coords, numpy.ndarray)

    assert isinstance(images_coords, numpy.ndarray)

    assert isinstance(image_shape, tuple)

    assert isinstance(camera_params, CameraParams)

    new_camera_params, _ = cv2.getOptimalNewCameraMatrix(
        camera_params["i_matrix"],
        camera_params["d_matrix"],
        image_shape, 1, image_shape
    )

    for idx, image in enumerate(images):

        # status_success, r_vector, t_vector = cv2.solvePnP(
        #     object_coords[idx],#.reshape((-1, 3)),
        #     images_coords[idx],#.reshape((-1, 2)),
        #     camera_params["i_matrix"],
        #     camera_params["d_matrix"]
        # )

        # __cam_params = _cam_params[idx]

        # r_vector = __cam_params["r_vector"]

        # t_vector =

        undistorted_img = cv2.undistort(
            image,
            camera_params["i_matrix"],
            camera_params["d_matrix"],
            None,
            new_camera_params
        )

        yield undistorted_img

def write_word_on_board(source_image : numpy.ndarray,
                        write_string : str,
                        file_storage : cv2.FileStorage,
                        letter_coord : List[ Tuple[ int, int ] ],
                        corner_coord : numpy.ndarray,
                        checker_size : Tuple[ int, int ],
                        camera_param : CameraParams,
                        image_vector : CameraParams) -> numpy.ndarray:
    
    assert isinstance(source_image, numpy.ndarray)

    assert isinstance(write_string, str)

    assert isinstance(file_storage, cv2.FileStorage)

    assert isinstance(letter_coord, list)

    assert isinstance(corner_coord, numpy.ndarray)

    assert isinstance(checker_size, tuple)

    assert isinstance(camera_param, CameraParams)

    assert isinstance(image_vector, CameraParams)

    target_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)

    def to_board_coord(coord : Tuple[ int, int, int ], letter_idx : int) -> Tuple[ int, int, int ]:

        nonlocal letter_coord, checker_size, corner_coord

        y = letter_coord[letter_idx][0] + coord[0]

        x = letter_coord[letter_idx][1] + coord[1]

        z = -coord[2]

        return (x, y, z)

    for letter_idx, write_char in enumerate(write_string):

        # (N, 2, 3)
        char_coords = file_storage.getNode(write_char.upper()).mat()

        # (N, 2, 3) => (..., 3)
        char_coords = numpy.array([
            [  to_board_coord(_char_coord, letter_idx) for _char_coord in char_coord  ]
                for char_coord in char_coords
        ]).reshape((-1, 3))

        # (..., 2)
        coordinates, _ = cv2.projectPoints(
            char_coords.astype(numpy.float32),
            image_vector["r_vector"],
            image_vector["t_vector"],
            camera_param["i_matrix"],
            camera_param["d_matrix"]
        )

        char_coords = coordinates.reshape((-1, 2, 2))

        for coord_1, coord_2 in char_coords:

            target_image = cv2.line(target_image, numpy.int32(coord_1), numpy.int32(coord_2), color = (0, 255, 0), thickness = 10)

    return target_image