import sys
import keyboard
import pyautogui
import time
import threading
import numpy as np
import random
from PIL import ImageGrab
import cv2
import math

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit
from PyQt5.QtCore import pyqtSignal, QObject, QThread

class LogEmitter(QObject):
    log = pyqtSignal(str)

class HumanLikeImageFollower:
    def __init__(self, logger):
        self.logger = logger
        self.running = False
        self.w_holding = False
        self.template = None
        self.template_size = None
        self.last_position = None
        self.last_target_position = None
        self.last_move_time = time.time()
        self.idle_time = 0
        self.last_speed = 0

        # Configuration
        self.template_path = 'name.png'
        self.confidence = 0.7
        self.scan_interval = 0.005
        self.search_region = None

        # Human-like movement
        self.smoothing_window = 1
        self.position_history = []
        self.max_jitter = 2
        self.vertical_offset_range = (0, 12)
        self.movement_variation = 0.1
        self.micro_move_frequency = 0.1
        self.micro_move_distance = 2
        self.sudden_move_threshold = 50
        self.q_cooldown = 0.5

        self.load_template()
        keyboard.add_hotkey('insert', self.start)
        keyboard.add_hotkey('delete', self.stop)
        self.logger.log.emit("Initialized. Press Insert to start, Delete to stop.")

    def log(self, message):
        self.logger.log.emit(message)

    def load_template(self):
        try:
            self.template = cv2.imread(self.template_path, cv2.IMREAD_COLOR)
            if self.template is None:
                raise FileNotFoundError
            self.template_size = self.template.shape[1], self.template.shape[0]
            self.log(f"Loaded template image: {self.template_path} (size: {self.template_size})")
        except Exception as e:
            self.log(f"Failed to load template image: {e}")
            self.template = None

    def start(self):
        if not self.running and self.template is not None:
            self.running = True
            self.w_holding = True
            self.last_q_press = 0
            threading.Thread(target=self.hold_w, daemon=True).start()
            threading.Thread(target=self.tracking_loop, daemon=True).start()
            self.log("Started tracking...")

    def stop(self):
        if self.running:
            self.running = False
            self.w_holding = False
            keyboard.release('w')
            self.log("Stopped tracking")

    def hold_w(self):
        keyboard.press('w')
        while self.w_holding:
            time.sleep(0.1)
        keyboard.release('w')

    def press_q(self):
        now = time.time()
        if now - self.last_q_press > self.q_cooldown:
            keyboard.press('q')
            time.sleep(0.05 + random.uniform(-0.02, 0.02))
            keyboard.release('q')
            self.last_q_press = now
            self.log("Pressed Q (sudden movement detected)")

    def get_smoothed_position(self, new_x, new_y):
        vertical_offset = random.randint(*self.vertical_offset_range)
        new_y += vertical_offset
        new_x += random.randint(-self.max_jitter, self.max_jitter)
        new_y += random.randint(-self.max_jitter, self.max_jitter)

        self.position_history.append((new_x, new_y))
        if len(self.position_history) > self.smoothing_window:
            self.position_history.pop(0)

        avg_x = sum(p[0] for p in self.position_history) / len(self.position_history)
        avg_y = sum(p[1] for p in self.position_history) / len(self.position_history)

        return int(avg_x), int(avg_y)
    
    def human_like_movement(self, target_x, target_y):
        if not self.running:
            return

        current_x, current_y = pyautogui.position()
        if self.last_target_position:
            time_elapsed = time.time() - self.last_move_time
            dx_prev = target_x - self.last_target_position[0]
            dy_prev = target_y - self.last_target_position[1]
            distance_moved = math.sqrt(dx_prev ** 2 + dy_prev ** 2)
            speed = distance_moved / time_elapsed if time_elapsed > 0 else 0
            self.last_speed = speed

            if speed > self.sudden_move_threshold:
                self.press_q()

            if distance_moved < 5:
                self.idle_time += time_elapsed
                if self.idle_time >= self.micro_move_frequency:
                    micro_dx = random.randint(-self.micro_move_distance, self.micro_move_distance)
                    micro_dy = random.randint(-self.micro_move_distance, self.micro_move_distance)
                    target_x += micro_dx
                    target_y += micro_dy
                    self.idle_time = 0
            else:
                self.idle_time = 0

        self.last_target_position = (target_x, target_y)
        self.last_move_time = time.time()

        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < 2:
            return

        steps = max(3, int(distance / (10 * random.uniform(0.8, 1.2))))
        steps = min(steps, 20)

        for i in range(1, steps + 1):
            if not self.running:
                return
            t = i / steps
            ease_t = t ** 0.5
            new_x = current_x + dx * ease_t
            new_y = current_y + dy * ease_t
            smooth_x, smooth_y = self.get_smoothed_position(new_x, new_y)
            pyautogui.moveTo(smooth_x, smooth_y)
            time.sleep(0.02)

        self.last_position = (target_x, target_y)

    def tracking_loop(self):
        while self.running and self.template is not None:
            try:
                screenshot = np.array(ImageGrab.grab(bbox=self.search_region))
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                result = cv2.matchTemplate(screenshot, self.template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val >= self.confidence:
                    target_x = max_loc[0] + self.template_size[0] // 2
                    target_y = max_loc[1] + self.template_size[1] // 2
                    if self.search_region:
                        target_x += self.search_region[0]
                        target_y += self.search_region[1]
                    self.human_like_movement(target_x, target_y)
                    self.log(f"Tracking target at ({target_x}, {target_y}) - Confidence: {max_val:.2f}")
                time.sleep(self.scan_interval)
            except Exception as e:
                self.log(f"Tracking error: {e}")
                time.sleep(1)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("aurapilot (PoE1)")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.logger = LogEmitter()
        self.logger.log.connect(self.append_log)

        self.follower = HumanLikeImageFollower(self.logger)

        self.start_btn.clicked.connect(self.follower.start)
        self.stop_btn.clicked.connect(self.follower.stop)

    def append_log(self, text):
        self.log_box.append(text)

if __name__ == "__main__":
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(400, 500)
    window.show()
    sys.exit(app.exec_())
