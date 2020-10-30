import numpy as np
import cv2

class Video:
    def __init__(self, src):
        self.frames = self.new(src)
        self.shape = self.frames.shape + self.frames[0].shape
        self.fps = 15

    def __getitem__(self, index):
        """
        case1: videoのみ指定
        $ video
        => Videoクラスが返る
        $ video.shape
        => (3975, 480, 640, 3)
        videoクラスはFrameクラスの複数持ったリスト

        case2: videoの1つのフレームを指定
        $ video[0]
        => Frameクラスが返る
        $ video[0].shape
        => (480, 640, 3)

        case3: videoに対して時間とy軸に切り取り
        $ video[0:10, 10:20]
        => Videoクラスが返る
        $ video[0:10, 10:20].shape
        => (10, 10, 640, 3)

        case4: videoに対して時間とy軸とx軸に切り取り
        $ video[0:10, 10:20, 20:30]
        => Videoクラスが返る
        $ video[0:10, 10:20, 20:30].shape
        => (10, 10, 10, 3)
        """
        
        # indexの分離
        if isinstance(index, tuple):
            time_index = index[0] # 時間方向
            frame_index = index[1:3] # 空間方向
        else:
            time_index = index
            frame_index = None

        # 時間の切り取り
        if isinstance(time_index, int):
            frames = self.frames[time_index][np.newaxis,:,:,:]
        elif isinstance(time_index, slice):
            frames = self.frames[time_index]

        # 空間方向の切り取り
        if isinstance(frame_index, tuple):
            frames = [frame[frame_index] for frame in frames]
            frames = np.array(frames)

        if frames.shape[0] == 1:
            return frames[0]
        else:
            return Video(frames)

    # 動画インスタンスを作成する
    def new(self, src):
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
                frame = Frame(frame) # Frameインスタンスの作成
                frames += [frame]
            frames = np.array(frames)
            cap.release()
            cv2.destroyAllWindows()
        # srcがlistの場合
        elif isinstance(src, list):
            frames = np.array(src)
        # srcがndarryの場合
        elif isinstance(src, np.ndarray):
            frames = src
        else:
           """
           - shapeが違うときのエラー処理
           - 生成できなかった時のエラー処理
           """
           pass
        return frames

    def save_video(self, path):
        size = self.shape[1:3][::-1]
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(path, fmt, self.fps, size)
        for frame in self.frames:
            writer.write(frame.frame)
        writer.release()

class Frame:
    def __init__(self, frame):
        self.frame = np.array(frame)
        self.shape = self.frame.shape

    def __getitem__(self, index):
        return Frame(self.frame[index])
    
    def save_frame(self, path):
        cv2.imwrite(path, self.frame)
