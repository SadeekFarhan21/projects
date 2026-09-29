import sys
import cv2
import numpy as np
import mss
import mediapipe as mp
import os

from PyQt5.QtWidgets import QApplication, QWidget, QBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor, QIcon, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtSvg import QSvgRenderer

# For Keras models.
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

#####################
# Constants
#####################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Colors and appearance.
TOOLBAR_BG_COLOR = "rgba(30, 30, 30, 230)"
TOOLBAR_BORDER_COLOR = "#FF9900"  # Example: orange border.
TOOLBAR_BORDER_WIDTH = 2
ICON_COLOR = "#FFFFFF"  # White icons.

# Sizing.
BUTTON_SIZE = 40
TOOLBAR_MARGIN = 10
LAYOUT_SPACING = 20
TOOLBAR_CONTENT_MARGIN = 20

# SVG icon file paths (in the assets subdirectory).
DRAG_HANDLE_SVG = os.path.join(BASE_DIR, "assets", "drag.svg")
FACE_SVG = os.path.join(BASE_DIR, "assets", "face.svg")
HAND_SVG = os.path.join(BASE_DIR, "assets", "hand.svg")
CLOSE_SVG = os.path.join(BASE_DIR, "assets", "close.svg")

# Model paths.
# Face detection model files.
FACE_MODEL_PROTO = os.path.join(BASE_DIR, "models", "face", "deploy.prototxt")
FACE_MODEL_WEIGHTS = os.path.join(BASE_DIR, "models", "face", "res10_300x300_ssd_iter_140000.caffemodel")
# Emotion model (Keras) for faces.
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion", "emotion.keras")
EMOTION_INPUT_SIZE = (48, 48)  # Expected input size for the emotion model.
# Gesture model for hand gestures (if you want to add gesture logic later).
GESTURE_MODEL_PATH = os.path.join(BASE_DIR, "models", "gestures", "gesture_model.keras")
GESTURE_INPUT_SIZE = (64, 64)

#####################
# Helper function to load and tint an SVG icon.
#####################
def load_svg_icon(path, size, color):
    # Create a transparent pixmap.
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    # Render the SVG onto the pixmap.
    renderer = QSvgRenderer(path)
    painter = QPainter(pixmap)
    renderer.render(painter)
    # Tint the pixmap.
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), QColor(color))
    painter.end()
    return QIcon(pixmap)

#####################
# Helper function for icon buttons.
#####################
def createIconButton(icon_path, fixed_size):
    btn = QPushButton()
    btn.setFixedSize(fixed_size, fixed_size)
    btn.setIcon(load_svg_icon(icon_path, fixed_size, ICON_COLOR))
    btn.setIconSize(btn.size())
    btn.setStyleSheet("border: none; background: transparent;")
    btn.setFlat(True)
    btn.setFocusPolicy(Qt.NoFocus)
    return btn

#####################
# Drag Handle Widget (non-clickable)
#####################
class DragHandle(QLabel):
    def __init__(self, fixed_size):
        super().__init__()
        self.setFixedSize(fixed_size, fixed_size)
        self.setPixmap(load_svg_icon(DRAG_HANDLE_SVG, fixed_size, ICON_COLOR).pixmap(fixed_size, fixed_size))
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: none; background: transparent;")
        # Allow mouse events to pass through so that the parent can handle dragging.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

########################################
# Face & Emotion Detection Thread
########################################
class FaceDetectorThread(QThread):
    # Emits a list of detections: [x, y, w, h, emotion]
    facesDetected = pyqtSignal(list)

    def __init__(self, monitor, parent=None):
        super().__init__(parent)
        self.monitor = monitor
        self.running = True
        # Load the face detection model.
        self.face_net = cv2.dnn.readNetFromCaffe(FACE_MODEL_PROTO, FACE_MODEL_WEIGHTS)
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # Load the emotion model.
        self.emotion_model = load_model(EMOTION_MODEL_PATH)

    def run(self):
        with mss.mss() as sct:
            while self.running:
                sct_img = sct.grab(self.monitor)
                orig_frame = np.array(sct_img)
                frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGRA2BGR)
                scale = 0.5
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                (h, w) = small_frame.shape[:2]
                blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300),
                                             (104.0, 177.0, 123.0))
                self.face_net.setInput(blob)
                detections = self.face_net.forward()
                results = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x2, y2) = box.astype("int")
                        # Scale coordinates back to original frame.
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int((x2 - x) / scale)
                        orig_h = int((y2 - y) / scale)
                        # Crop face from the original frame.
                        face_img = orig_frame[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                        # Convert to grayscale.
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        # Resize to the expected input size.
                        resized_face = cv2.resize(gray_face, EMOTION_INPUT_SIZE)
                        # Preprocess for emotion model.
                        img_array = img_to_array(resized_face) / 255.0
                        input_face = np.expand_dims(img_array, axis=0)
                        # Predict emotion.
                        preds = self.emotion_model.predict(input_face)
                        emotion_idx = np.argmax(preds, axis=1)[0]
                        emotion = self.decode_emotion(emotion_idx)
                        results.append([orig_x, orig_y, orig_w, orig_h, emotion])
                self.facesDetected.emit(results)
                self.msleep(100)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def decode_emotion(self, idx):
        mapping = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }
        return mapping.get(idx, "Neutral")

########################################
# Hand Detection Thread (using MediaPipe with landmarks)
########################################
class HandDetectorThread(QThread):
    # Emits a list for each hand:
    # [x, y, w, h, label, landmarks]
    handsDetected = pyqtSignal(list)

    def __init__(self, monitor, parent=None):
        super().__init__(parent)
        self.monitor = monitor
        self.running = True
        self.mp_hands = mp.solutions.hands
        # Lower thresholds can improve sensitivity on smaller hands.
        self.hands = self.mp_hands.Hands(
            max_num_hands=10,
            min_detection_confidence=0.3,  
            min_tracking_confidence=0.3)
        # Downscale factor to enlarge the hand appearance.
        self.detection_scale = 0.5

    def run(self):
        with mss.mss() as sct:
            while self.running:
                sct_img = sct.grab(self.monitor)
                frame = np.array(sct_img)
                original_height, original_width = frame.shape[:2]
                # Downscale the frame to make hands appear larger.
                frame_small = cv2.resize(frame, (0, 0), fx=self.detection_scale, fy=self.detection_scale)
                rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGRA2RGB)
                rgb_frame.flags.writeable = False
                results = self.hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                hand_boxes = []
                if results.multi_hand_landmarks:
                    small_height, small_width, _ = frame_small.shape
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Compute bounding box on the downscaled image.
                        xs = [landmark.x * small_width for landmark in hand_landmarks.landmark]
                        ys = [landmark.y * small_height for landmark in hand_landmarks.landmark]
                        x_min, x_max = int(min(xs)), int(max(xs))
                        y_min, y_max = int(min(ys)), int(max(ys))
                        # Scale coordinates back to original.
                        x_min_orig = int(x_min / self.detection_scale)
                        y_min_orig = int(y_min / self.detection_scale)
                        x_max_orig = int(x_max / self.detection_scale)
                        y_max_orig = int(y_max / self.detection_scale)
                        w_box_orig = x_max_orig - x_min_orig
                        h_box_orig = y_max_orig - y_min_orig

                        # Compute landmark positions (scaled back to original frame).
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            lx = int(landmark.x * small_width / self.detection_scale)
                            ly = int(landmark.y * small_height / self.detection_scale)
                            landmarks.append((lx, ly))

                        hand_boxes.append([x_min_orig, y_min_orig, w_box_orig, h_box_orig, "Hand", landmarks])
                self.handsDetected.emit(hand_boxes)
                self.msleep(100)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

########################################
# Overlay Widget (draws face emotion, hand bounding boxes, landmarks, and connecting lines)
########################################
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Allow clicks to pass through.
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.showFullScreen()
        # For faces: each element [x, y, w, h, emotion]
        self.faces = []
        # For hands: each element [x, y, w, h, label, landmarks]
        self.hands = []
        self.show_faces = True
        self.show_hands = True

    def updateFaces(self, faces):
        self.faces = faces
        self.update()

    def updateHands(self, hands):
        self.hands = hands
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw face detection boxes and emotion text.
        if self.show_faces:
            pen_face = QPen(QColor(0, 255, 0), 3)
            painter.setPen(pen_face)
            for (x, y, w, h, emotion) in self.faces:
                painter.drawRect(x, y, w, h)
                font_metrics = painter.fontMetrics()
                text_rect = font_metrics.boundingRect(emotion)
                text_padding = 4
                bg_rect = text_rect.adjusted(-text_padding, -text_padding, text_padding, text_padding)
                bg_rect.moveTo(x, y - bg_rect.height() - 2)
                painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(bg_rect, Qt.AlignCenter, emotion)
        # Draw hand detection boxes, labels, landmarks, and connection lines.
        if self.show_hands:
            pen_hand = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen_hand)
            for hand in self.hands:
                # Expect hand to be [x, y, w, h, label, landmarks]
                if len(hand) >= 6:
                    x, y, w, h, label, landmarks = hand
                else:
                    x, y, w, h, label = hand
                    landmarks = []
                painter.drawRect(x, y, w, h)
                font_metrics = painter.fontMetrics()
                text_rect = font_metrics.boundingRect(str(label))
                text_padding = 4
                bg_rect = text_rect.adjusted(-text_padding, -text_padding, text_padding, text_padding)
                bg_rect.moveTo(x, y - bg_rect.height() - 2)
                painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(bg_rect, Qt.AlignCenter, str(label))
                # Draw hand landmarks as small unfilled circles.
                painter.setPen(QPen(QColor(0, 0, 255), 2))
                painter.setBrush(Qt.NoBrush)
                for (lx, ly) in landmarks:
                    radius = 4
                    painter.drawEllipse(QPoint(lx, ly), radius, radius)
                # Draw connecting lines using MediaPipe's HAND_CONNECTIONS.
                connections = mp.solutions.hands.HAND_CONNECTIONS
                painter.setPen(QPen(QColor(0, 255, 255), 2))
                if len(landmarks) >= 21:
                    for connection in connections:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            start_point = QPoint(landmarks[start_idx][0], landmarks[start_idx][1])
                            end_point = QPoint(landmarks[end_idx][0], landmarks[end_idx][1])
                            painter.drawLine(start_point, end_point)

########################################
# Simplified Toolbar (toggle buttons for faces and hands, plus drag-handle and close)
########################################
class Toolbar(QWidget):
    toggleFaceOverlay = pyqtSignal(bool)
    toggleHandOverlay = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"background-color: {TOOLBAR_BG_COLOR};"
            f"border: {TOOLBAR_BORDER_WIDTH}px solid {TOOLBAR_BORDER_COLOR};"
            "border-radius: 5px;"
        )
        self.buttonSize = BUTTON_SIZE
        self.layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.layout.setContentsMargins(TOOLBAR_CONTENT_MARGIN, TOOLBAR_CONTENT_MARGIN,
                                       TOOLBAR_CONTENT_MARGIN, TOOLBAR_CONTENT_MARGIN)
        self.layout.setSpacing(LAYOUT_SPACING)
        self.dragHandle = DragHandle(self.buttonSize)
        self.layout.addWidget(self.dragHandle)
        self.faceBtn = createIconButton(FACE_SVG, self.buttonSize)
        self.faceBtn.setCheckable(True)
        self.faceBtn.setChecked(True)
        self.faceBtn.clicked.connect(lambda: self.toggleFaceOverlay.emit(self.faceBtn.isChecked()))
        self.layout.addWidget(self.faceBtn)
        self.handBtn = createIconButton(HAND_SVG, self.buttonSize)
        self.handBtn.setCheckable(True)
        self.handBtn.setChecked(True)
        self.handBtn.clicked.connect(lambda: self.toggleHandOverlay.emit(self.handBtn.isChecked()))
        self.layout.addWidget(self.handBtn)
        self.closeBtn = createIconButton(CLOSE_SVG, self.buttonSize)
        self.closeBtn.clicked.connect(QApplication.instance().quit)
        self.layout.addWidget(self.closeBtn)
        self.setLayout(self.layout)
        self._drag_active = False
        self._drag_position = QPoint(0, 0)
        self.pinToScreenEdge()

    def pinToScreenEdge(self):
        desktop = QApplication.desktop().screenGeometry()
        x = desktop.x() + (desktop.width() - self.width()) // 2
        y = desktop.y() + TOOLBAR_MARGIN
        self.move(x, y)
        self.layout.setDirection(QBoxLayout.LeftToRight)
        self.adjustToolbarSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_active and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = False
            self.snapToEdge()
            event.accept()

    def snapToEdge(self):
        desktop = QApplication.desktop().screenGeometry()
        cur_geo = self.frameGeometry()
        left_dist = cur_geo.left() - desktop.left()
        right_dist = desktop.right() - cur_geo.right()
        top_dist = cur_geo.top() - desktop.top()
        bottom_dist = desktop.bottom() - cur_geo.bottom()
        min_dist = min(left_dist, right_dist, top_dist, bottom_dist)
        if min_dist == left_dist:
            new_x = desktop.left()
            new_y = cur_geo.top()
            self.layout.setDirection(QBoxLayout.TopToBottom)
        elif min_dist == right_dist:
            new_x = desktop.right() - self.width() + 1
            new_y = cur_geo.top()
            self.layout.setDirection(QBoxLayout.TopToBottom)
        elif min_dist == top_dist:
            new_x = cur_geo.left()
            new_y = desktop.top()
            self.layout.setDirection(QBoxLayout.LeftToRight)
        else:
            new_x = cur_geo.left()
            new_y = desktop.bottom() - self.height() + 1
            self.layout.setDirection(QBoxLayout.LeftToRight)
        self.move(new_x, new_y)
        self.adjustToolbarSize()

    def adjustToolbarSize(self):
        count = self.layout.count()
        spacing = self.layout.spacing()
        margins = self.layout.contentsMargins()
        if self.layout.direction() in (QBoxLayout.LeftToRight, QBoxLayout.RightToLeft):
            total_width = count * self.buttonSize + (count - 1) * spacing + margins.left() + margins.right()
            total_height = self.buttonSize + margins.top() + margins.bottom()
            self.setFixedSize(total_width, total_height)
        else:
            total_width = self.buttonSize + margins.left() + margins.right()
            total_height = count * self.buttonSize + (count - 1) * spacing + margins.top() + margins.bottom()
            self.setFixedSize(total_width, total_height)

########################################
# Main Application
########################################
class MainApp(QWidget):
    def __init__(self, monitor):
        super().__init__()
        self.overlay = Overlay()
        self.toolbar = Toolbar()
        self.toolbar.show()
        self.monitor = monitor

        self.faceThread = FaceDetectorThread(self.monitor)
        self.handThread = HandDetectorThread(self.monitor)
        
        self.faceThread.facesDetected.connect(self.overlay.updateFaces)
        self.handThread.handsDetected.connect(self.overlay.updateHands)
        
        self.toolbar.toggleFaceOverlay.connect(self.setFaceOverlay)
        self.toolbar.toggleHandOverlay.connect(self.setHandOverlay)
        
        self.faceThread.start()
        self.handThread.start()

    def setFaceOverlay(self, show):
        self.overlay.show_faces = show
        self.overlay.update()

    def setHandOverlay(self, show):
        self.overlay.show_hands = show
        self.overlay.update()

    def closeEvent(self, event):
        self.faceThread.stop()
        self.handThread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    with mss.mss() as sct:
        monitor = sct.monitors[1]
    main_app = MainApp(monitor)
    exit_code = app.exec_()
    sys.exit(exit_code)
