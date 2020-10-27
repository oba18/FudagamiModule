import numpy as np
import cv2

class Video:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Video(self[index])

class Frame:
    def __init__(self):
        pass

# videoインスタンスの生成
def new(src):
    # srcがファイル名の場合
    if isinstance(src, str):
        frames = []
        cap = cv2.VideoCapture(src)
        while (cap.isOpened()):
            end_flag, frame = cap.read()
            if frame is None:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frames += [frame]
        frames = np.array(frames)
        cap.release()
        video = Video(frames)
    # srcがlistの場合
    elif isinstance(src, list):
        video = Video(src)
    # srcがndarryの場合
    elif isinstance(src, np.ndarray):
        video = Video(src)
    return video

