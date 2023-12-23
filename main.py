from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QSpinBox, QGroupBox, QLineEdit, QLabel, QFileDialog
from PyQt5.QtGui import QValidator, QRegExpValidator
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QPixmap, QImage

from PIL import Image 

from typing import *

import sys, cv2, os

import numpy as np

from utilities import *

from background_subtraction import BackgroundSubtractor

from optical_flow import resize_to_width, find_good_corners, draw_cross, draw_history_points, predict_next_corner

from principal_component_analysis import image_pca, find_optimal_component_n

from matplotlib import pyplot as plt

from mnist_vgg19 import summary, load_mnist_model, ModifiedVGG19, predict_mnist_image, draw_mnist_distribution

import matplotlib.patches as patches

from dog_cat_resnet50 import load_dog_cat_model, ModifiedResNet50, predict_dog_cat_image

class GraffitiBoard(QWidget):

    def __init__(self, *args, **kwargs):

        super().__init__()

        self.initUI(*args, **kwargs)

    def initUI(self, *args, **kwargs):

        self.setGeometry(*args, **kwargs)

        self.image = np.zeros(shape = (args[3], args[2], 1), dtype = np.uint8)

        self.last_point = None

    def paintEvent(self, event):

        painter = QPainter(self)
        pixmap = QPixmap(400, 400)
        pixmap.fill(Qt.black)

        # Convert the NumPy array to QImage
        image = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_Grayscale8)
        
        pixmap.convertFromImage(image)

        painter.drawPixmap(self.rect(), pixmap)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):

        if event.buttons() and Qt.LeftButton and self.last_point:

            painter = QPainter(self)
            painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.drawOnImage(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):

        if event.button == Qt.LeftButton:
            self.last_point = None

    def drawOnImage(self, start, end):

        image = QImage(
            self.image.ctypes.data, 
            self.image.shape[1], 
            self.image.shape[0], 
            self.image.strides[0], 
            QImage.Format_Grayscale8
        )

        painter = QPainter(image)
        painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start, end)
        painter.end()

        def qimage_to_numpy(qimage : QImage) -> np.ndarray:

            buffer = qimage.bits()
            buffer.setsize(qimage.byteCount())

            qimage_array = np.array(buffer).reshape((qimage.height(), qimage.width(), 1)) 

            return qimage_array

        np.copyto(self.image, qimage_to_numpy(image))

    def clearBoard(self):

        self.image.fill(0)

        self.update()

class MyWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.initUI()

    def create_section(self, section_name : str, xywh_offset : Tuple[ int, int ], *widget_list) -> QGroupBox:
        
        group_box = QGroupBox(title = section_name, **({} if (self is None) else { "parent" : self }))

        group_box.setAlignment(Qt.AlignCenter)

        group_box.setGeometry(*xywh_offset)

        for widget, offset, callback in widget_list:

            widget.setParent(group_box)

            widget.setGeometry(*offset)

            callback(widget)

        return group_box

    def __initialize_background_subtractor(self) -> None:

        self.background_subtractor = BackgroundSubtractor(
            num_history_frames = 100, 
            pixel_difference = 200, 
            detect_shadows = True
        )

    def initUI(self):

        self.setGeometry(50, 50, 1400, 1000) 

        self.setWindowTitle("CVDLHW2 - P76124786")

        # section 0

        section_0_items = [
            (QPushButton("Load Image"),  (50,  50, 120, 60), lambda x : x.clicked.connect(self.load_image)), 
            (QPushButton("Load Video"),  (50, 150, 120, 60), lambda x : x.clicked.connect(self.load_video)), 
        ]

        self.section_0 = self.create_section("0. Data Loading", (20, 20, 220, 250), *section_0_items)

        # section 1

        self.__initialize_background_subtractor()

        section_1_items = [
            (QPushButton("1. Background Subtraction"),  (50,  50, 200, 60), lambda x : x.clicked.connect(self.background_subtraction))
        ]

        self.section_1 = self.create_section("1. Background Subtraction", (250, 20, 300, 150), *section_1_items)

        # section 2

        section_2_items = [
            (QPushButton("2.1 Preprocessing"),   (50,  50, 200, 60), lambda x : x.clicked.connect(self.optical_flow_preprocessing)),
            (QPushButton("2.2 Video Tracking"),  (50, 120, 200, 60), lambda x : x.clicked.connect(self.optical_flow_tracking))
        ]

        self.section_2 = self.create_section("2. Optical Flow", (250, 190, 300, 230), *section_2_items)

        # section 3

        section_3_items = [
            (QPushButton("3. Dimension Reduction"),   (50,  50, 200, 60), lambda x : x.clicked.connect(self.principal_component_analysis)),
        ]

        self.section_3 = self.create_section("3. PCA", (250, 430, 300, 170), *section_3_items)

        # section 4

        self.sect4_predict_output = QLabel("Predicted = <N/A>")

        self.sect4_graffiti_board = GraffitiBoard(0, 0, 340, 340)

        section_4_items = [
            (QPushButton("4.1 Show Model Structure"),   ( 50,  50, 240,  40), lambda x : x.clicked.connect(self.summarize_mnist_model)),
            (QPushButton("4.2 Show Accuracy and Loss"), ( 50, 120, 240,  40), lambda x : x.clicked.connect(self.show_mnist_training_history)),
            (QPushButton("4.3 Predict"),                ( 50, 190, 240,  40), lambda x : x.clicked.connect(self.predict_mnist_image)),
            (QPushButton("4.4 Reset"),                  ( 50, 260, 240,  40), lambda x : x.clicked.connect(self.clear_mnist_graffiti_board)),
            (self.sect4_predict_output,                 ( 50, 330, 240,  40), lambda x : x.setAlignment(Qt.AlignCenter)),
            (self.sect4_graffiti_board,                 (320,  30, 340, 340), lambda x : x)
        ]

        self.section_4 = self.create_section("4. MNIST Classifier Using VGG19", (580, 20, 720, 400), *section_4_items)

        # section 5

        self.sect5_predict_output = QLabel("Predicted = <N/A>")

        self.sect5_selected_image = QLabel()

        section_5_items = [
            (QPushButton("5.0 Load Image"),           ( 50,  50, 240,  40), lambda x : x.clicked.connect(self.load_dog_cat_image)),
            (QPushButton("5.1 Show Images"),          ( 50, 120, 240,  40), lambda x : x.clicked.connect(self.show_dog_cat_images)),
            (QPushButton("5.2 Show Model Structure"), ( 50, 190, 240,  40), lambda x : x.clicked.connect(self.summarize_dog_cat_model)),
            (QPushButton("5.3 Show Comparison"),      ( 50, 260, 240,  40), lambda x : x.clicked.connect(self.show_dog_cat_comparison)),
            (QPushButton("5.4 Inference"),            ( 50, 330, 240,  40), lambda x : x.clicked.connect(self.predict_dog_cat_image)),
            (self.sect5_predict_output,               ( 50, 400, 240,  40), lambda x : x.setAlignment(Qt.AlignCenter)),
            (self.sect5_selected_image,               (320,  50, 360, 360), lambda x : x.setPixmap(QPixmap()))
        ]

        self.section_5 = self.create_section("5. ResNet50", (580, 430, 720, 470), *section_5_items)

        # instantiate class variables 

        self.__initailize_optical_flow_variables()

        self.__initialize_principal_component_analysis_variables()

        self.__initialize_mnist_vgg19_variables()

        self.__initialize_dog_cat_resnet50_variables()

    def __initialize_dog_cat_resnet50_variables(self) -> None:

        self.cat_relative_folder = "Cat"

        self.dog_relative_folder = "Dog"

        self.dog_cats_images = None

        self.dog_cat_resnet50_model_name_1 = os.path.join(os.path.dirname(__file__), "models/resnet50_bn_dogs_cats_d172023_2.pt")

        self.dog_cat_resnet50_model_name_2 = os.path.join(os.path.dirname(__file__), "models/resnet50_bn_dogs_cats_d172023_3.pt")

        self.dog_cat_resnet50_model_1 = None

        self.dog_cat_resnet50_model_2 = None

        self.dog_cat_resnet50_model_input_size = (3, 224, 224)

        self.dog_cat_img_temp_path = os.path.join(os.path.dirname(__file__), "data/dog_cat_tmp.png")

        os.makedirs(os.path.dirname(self.dog_cat_img_temp_path), exist_ok = True)

        self.dog_cat_model_comparison_chart_fn = os.path.join(os.path.dirname(__file__), "models/resnet50_bn_dog_cats_comparison.png")

        self.dog_cat_model_comparison_chart = None

    def __initialize_mnist_vgg19_variables(self) -> None:

        self.mnist_vgg19_model_name = os.path.join(os.path.dirname(__file__), "models/vgg19_bn_MNIST_d162023.pt")

        self.mnist_vgg19_model = None

        self.mnist_vgg19_model_input_size = (3, 224, 224)

        self.mnist_training_history_accuracy = None
        
        self.mnist_training_history_accuracy_filename = os.path.join(os.path.dirname(__file__), "models/mnist_vgg19_training_accuracy.png")

        self.mnist_training_history_loss = None
            
        self.mnist_training_history_loss_filename = os.path.join(os.path.dirname(__file__), "models/mnist_vgg19_training_loss.png")

    def __initialize_principal_component_analysis_variables(self) -> None:

        self.image = None

    def select_folder(self) -> str:

        folder_path = QFileDialog.getExistingDirectory(self, "Select a folder")

        print(folder_path)

        return folder_path

        # images_in_folder = files_in_folder(folder_path)
        

        # print(images_in_folder)

        # self.images = list(load_images(images_in_folder, to_grayscale))

        # print(len(self.images), self.images[0].shape)

    def load_image(self) -> None:

        self.image = cv2.imread(QFileDialog.getOpenFileName(self, "Select a file")[0])

        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def load_video(self) -> None:

        self.video = cv2.VideoCapture(QFileDialog.getOpenFileName(self, "Select a file")[0])

    def background_subtraction(self) -> None:

        WINDOW_NAME_ORIGINAL = "Original"

        WINDOW_NAME_FOREGROUND = "Foreground"

        WINDOW_NAME_RESULT_FRAME = "Result"

        if (self.video is None):
            self.load_video()

        while self.video.isOpened():

            ret, frame = self.video.read()

            if not ret:
                break

            (fg_mask, result_frame) = self.background_subtractor.process_frame(frame)

            cv2.imshow(WINDOW_NAME_ORIGINAL, frame)

            cv2.imshow(WINDOW_NAME_FOREGROUND, fg_mask)

            cv2.imshow(WINDOW_NAME_RESULT_FRAME, result_frame)

            if (cv2.waitKey(30) & 0xFF == ord('q')):
                break

        self.video.release()

        cv2.destroyWindow(WINDOW_NAME_ORIGINAL)

        cv2.destroyWindow(WINDOW_NAME_FOREGROUND)

        cv2.destroyWindow(WINDOW_NAME_RESULT_FRAME)

        self.__initialize_background_subtractor()

    def __initailize_optical_flow_variables(self) -> None:
            
        self.video = None

        self.corners = None

        self.prev_frame = None

        self.history_corners = []

    def optical_flow_preprocessing(self) -> None:

        WINDOW_NAME_CORNER = "Corners"

        if (self.video is None):
            self.load_video()

        cv2.namedWindow(WINDOW_NAME_CORNER)

        while self.video.isOpened():

            ret, frame = self.video.read()

            if not ret:
                break

            corner, self.corners = find_good_corners(frame)

            if (corner is None):
                continue

            corner = tuple(corner[0])

            self.history_corners.append(corner)

            frame = draw_cross(frame, corner, radius = 30)

            break

        self.prev_frame = frame.copy()

        frame = resize_to_width(frame, 512)

        try:

            cv2.imshow(WINDOW_NAME_CORNER, frame)
            
            cv2.waitKey(0)

            cv2.destroyWindow(WINDOW_NAME_CORNER)

        except KeyboardInterrupt:
            raise

        except Exception:
            pass 

    def optical_flow_tracking(self) -> None:

        WINDOW_NAME_CORNER = "Corners"

        if ((self.video is None) or 
            (self.corners is None) or 
            (self.prev_frame is None) or 
            (self.history_corners.__len__() == 0)
        ):
            self.optical_flow_preprocessing()

        history_points = []

        corner = None

        cv2.namedWindow(WINDOW_NAME_CORNER)

        while self.video.isOpened():

            ret, frame = self.video.read()

            if not ret:
                break

            if (self.corners is None):

                corner = ()
                self.prev_frame = frame.copy()

            else:

                corner, self.corners = predict_next_corner(
                    frame_1 = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY),
                    frame_2 = cv2.cvtColor(frame,           cv2.COLOR_BGR2GRAY),
                    corners = self.corners
                )

                self.prev_frame = frame.copy()

            # print(corner)

            if (corner is None):
                continue

            corner = (x, y) = tuple(corner[0])

            frame = draw_cross(frame, corner, radius = 30)

            history_points.append(corner)
            
            frame = resize_to_width(draw_history_points(frame, history_points), 512)

            cv2.imshow(WINDOW_NAME_CORNER, frame)

            if (cv2.waitKey(30) & 0xFF == ord('q')):
                break

        self.video.release()

        cv2.destroyWindow(WINDOW_NAME_CORNER)

        self.__initailize_optical_flow_variables()

    def principal_component_analysis(self) -> None:

        WINDOW_NAME_PCA = "Principal Component Analysis"

        if (self.image is None):
            self.load_image()

        rgb_image = self.image.copy()

        min_n_components = find_optimal_component_n(rgb_image)

        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # gray_image_normalized = gray_image / 255.0

        # print("Minimum number of components with error <= 3.0:", min_n_components)

        reconstructed_image = image_pca(rgb_image, min_n_components)

        plt.figure(figsize = (8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gray_image, cmap = 'gray')
        plt.title('Original Gray Image')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap = 'gray')
        plt.title(f'Reconstructed Image (n={min_n_components})')

        figure = plt.gcf();  figure.canvas.draw()

        data = np.frombuffer(figure.canvas.tostring_rgb(), dtype = np.uint8)

        width, height = figure.canvas.get_width_height()

        image_rgb = data.reshape((height, width, 3))

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        try:

            cv2.imshow(WINDOW_NAME_PCA, image_bgr)

            cv2.waitKey(0)

            cv2.destroyWindow(WINDOW_NAME_PCA)

        except KeyboardInterrupt:
            raise

        except Exception:
            pass

    def _load_mnist_model(self) -> None:

        if (self.mnist_vgg19_model is None):
            self.mnist_vgg19_model = load_mnist_model(self.mnist_vgg19_model_name)

    def _load_dog_cat_models(self) -> None:

        if (self.dog_cat_resnet50_model_1 is None):
            self.dog_cat_resnet50_model_1 = load_dog_cat_model(self.dog_cat_resnet50_model_name_1)

        if (self.dog_cat_resnet50_model_2 is None):
            self.dog_cat_resnet50_model_2 = load_dog_cat_model(self.dog_cat_resnet50_model_name_2)

    def summarize_mnist_model(self) -> None:

        self._load_mnist_model()

        summary(self.mnist_vgg19_model, self.mnist_vgg19_model_input_size)

    def summarize_dog_cat_model(self) -> None:

        self._load_dog_cat_models()

        summary(self.dog_cat_resnet50_model_2, self.dog_cat_resnet50_model_input_size)

    def show_mnist_training_history(self) -> None:

        if (self.mnist_training_history_accuracy is None):
            self.mnist_training_history_accuracy = cv2.imread(self.mnist_training_history_accuracy_filename)

        if (self.mnist_training_history_loss is None):
            self.mnist_training_history_loss = cv2.imread(self.mnist_training_history_loss_filename)

        WINDOW_NAME_LOSS = "MNIST VGG19 Loss History"

        WINDOW_NAME_ACCU = "MNIST VGG19 Accuracy History"

        try:

            cv2.imshow(WINDOW_NAME_LOSS, self.mnist_training_history_loss)

            cv2.imshow(WINDOW_NAME_ACCU, self.mnist_training_history_accuracy)

            cv2.waitKey(0)

            cv2.destroyWindow(WINDOW_NAME_ACCU)

            cv2.destroyWindow(WINDOW_NAME_LOSS)

        except KeyboardInterrupt:
            raise

        except Exception:
            pass

    def predict_mnist_image(self) -> None:

        WINDOW_NAME_MNIST_DISTRIBUTION = "MNIST Inference Probability Distribution"

        self._load_mnist_model()

        image_to_predict = self.sect4_graffiti_board.image

        (digit_class, predict_proba) = predict_mnist_image(
            self.mnist_vgg19_model, 
            image_to_predict
        )

        # print(digit_class)

        # print(predict_proba)

        self.sect4_predict_output.setText(f"Predicted = {digit_class}")

        distribution_img = draw_mnist_distribution(predict_proba.tolist())

        try:

            cv2.imshow(WINDOW_NAME_MNIST_DISTRIBUTION, distribution_img)

            cv2.waitKey(0)

            cv2.destroyWindow(WINDOW_NAME_MNIST_DISTRIBUTION)

        except KeyboardInterrupt:
            raise

        except Exception:
            pass

    def clear_mnist_graffiti_board(self) -> None:

        self.sect4_graffiti_board.clearBoard()

    def show_dog_cat_images(self) -> None:

        WINDOW_NAME_CAT_DOG = "Cat & Dog"

        selected_folder = self.select_folder()

        cat_absolute_folder = os.path.join(selected_folder, self.cat_relative_folder)

        dog_absolute_folder = os.path.join(selected_folder, self.dog_relative_folder)

        files_in_cats = files_in_folder(cat_absolute_folder)

        files_in_dogs = files_in_folder(dog_absolute_folder)

        self.dog_cats_images = {
            "dogs" : list(load_images(files_in_dogs)),
            "cats" : list(load_images(files_in_cats))
        }

        cat_image = self.dog_cats_images["cats"][np.random.randint(0, len(self.dog_cats_images["cats"]))]

        dog_image = self.dog_cats_images["dogs"][np.random.randint(0, len(self.dog_cats_images["dogs"]))]

        fig, axs = plt.subplots(1, 2, figsize = (10, 5))

        axs[0].imshow(dog_image)
        axs[0].set_title("Dog")
        axs[0].axis("off") 

        rect_dog = patches.Rectangle((0, 0), 1, 0.1, linewidth = 1, 
            edgecolor = "none", facecolor = "white", transform = axs[0].transAxes
        )

        axs[0].add_patch(rect_dog)

        axs[1].imshow(cat_image)
        axs[1].set_title('Cat')
        axs[1].axis('off')  

        rect_cat = patches.Rectangle((0, 0), 1, 0.1, linewidth=1, 
            edgecolor='none', facecolor='white', transform=axs[1].transAxes
        )

        axs[1].add_patch(rect_cat)

        plt.tight_layout()

        figure = plt.gcf()

        figure.canvas.draw()

        image_plot = np.uint8(figure.canvas.renderer._renderer)

        plt.clf()

        try:

            cv2.imshow(WINDOW_NAME_CAT_DOG, image_plot)

            cv2.waitKey(0)

            cv2.destroyWindow(WINDOW_NAME_CAT_DOG)

        except KeyboardInterrupt:
            raise

        except Exception:
            pass

    def load_dog_cat_image(self) -> None:

        self.load_image()

        cv2.imwrite(self.dog_cat_img_temp_path, cv2.resize(self.image, (360, 360)))

        self.image = cv2.resize(self.image, (224, 224))

        self.sect5_selected_image.setPixmap(QPixmap(self.dog_cat_img_temp_path))

    def show_dog_cat_comparison(self) -> None:

        WINDOW_NAME_DOG_CAT_COMP = "Model Comparison"

        if (self.dog_cat_model_comparison_chart is None):
            self.dog_cat_model_comparison_chart = cv2.imread(self.dog_cat_model_comparison_chart_fn)

        try:

            cv2.imshow(WINDOW_NAME_DOG_CAT_COMP, self.dog_cat_model_comparison_chart)

            cv2.waitKey(0)

            cv2.destroyWindow(WINDOW_NAME_DOG_CAT_COMP)

        except KeyboardInterrupt:
            raise

        except Exception:
            pass

    def predict_dog_cat_image(self) -> None:

        self._load_dog_cat_models()

        if (self.image is None):
            self.load_dog_cat_image()

        image_to_predict = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        (digit_class, *_) = predict_dog_cat_image(
            self.dog_cat_resnet50_model_2, 
            image_to_predict
        )

        animal_label = ("Dog" if (digit_class) else ("Cat"))

        self.sect5_predict_output.setText(f"Predicted = {animal_label}")

if (__name__ == '__main__'):

    app = QApplication(sys.argv)

    window = MyWindow()

    window.show()

    sys.exit(app.exec_())