import numpy as np
import cv2
import matplotlib.pyplot as plt
from kernel import Kernel

class Video:
    """
    動画を扱うクラス

    できること:
    - 動画インスタンスを作成する
        - __init__
    - 動画を自由に切り取ったりできる（時間方向，大きさ）
        - __getitem__ スライス
    - 動画を保存することができる
        - save_video
    - カーネルを適用したときの様子で保存できる
        - save_video_applied_kernel
    """
    def __init__(self, src, fps=15, start=0, end=10000):
        """
        fps: integer
            fps
        start: integer
            読み込む動画の時間の始まり(秒)
        end: integer
            読み込む動画の時間の終わり(秒)
        """
        self.start = start
        self.end = end
        self.fps = 15
        self.start_frame = start * self.fps
        self.end_frame = end * self.fps
        self.frames = self.new(src)
        self.shape = self.frames.shape + self.frames[0].shape

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

    def new(self, src):
        """
        動画インスタンスを生成する
        """
        # srcがファイル名の場合
        frame_count = 0
        if isinstance(src, str):
            frames = []
            cap = cv2.VideoCapture(src)
            while (cap.isOpened()):
                end_flag, frame = cap.read()
                if frame is None:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if frame_count >= self.start_frame and frame_count <= self.end_frame:
                    frame = Frame(frame) # Frameインスタンスの作成
                    frames += [frame]
                # 終了時間以降は読み込まない
                if frame_count == self.end_frame:
                    break
                frame_count += 1
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
           print ("shape is wrong, please retry.")
           pass
        return frames

    def save_video(self, path):
        """
        動画を保存する
        """
        size = self.shape[1:3][::-1]
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        if len(self.frames[0].frame.shape) == 2:
            # カーネル適用したものはグレー画像
            writer = cv2.VideoWriter(path, fmt, self.fps, size, False)
        else:
            # カーネル適用したもの以外はカラー画像
            writer = cv2.VideoWriter(path, fmt, self.fps, size, True)
        for frame in self.frames:
            if len(frame.frame.shape) == 2:
                frame = frame.frame.reshape(size + (1,))
            else:
                frame = frame.frame
            #  writer.write(frame.frame)
            writer.write(frame)
        writer.release()

    def save_video_applied_kernel(self, path='./test.mp4', kernel_size=5):
        """
        カーネルを適用したときにどんな感じになるかを視覚化
        """
        size = self.shape[1:3][::-1]
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(path, fmt, self.fps, size)
        for frame in self.frames:
            for x in range(0, frame.shape[1], kernel_size):
                cv2.line(frame.frame, (x, 0), (x, frame.shape[0]), (255,255,255), 1)
            for y in range(0, frame.shape[0], kernel_size):
                cv2.line(frame.frame, (0, y), (frame.shape[1], y), (255,255,255), 1)
            writer.write(frame.frame)
        writer.release()

    def apply_kernel(self, kernel_name='bgr_mean', size=5):
        """
        動画に対してカーネルを適用し動画を返す
        """
        # カーネルの読み込み
        kernel = Kernel(kernel_name, size=size)
        kernel_size_x = int(self.frames[0].frame.shape[0] / size)
        kernel_size_y = int(self.frames[0].frame.shape[1] / size)
        applied_kernel_frames = []
        for frame in self.frames:
            applied_kernel_frame = kernel(frame.frame)
            # カーネルを適用した後の画像を生成
            applied_kernel_frame = np.array(applied_kernel_frame).reshape([kernel_size_y, kernel_size_x])
            applied_kernel_frame = Frame(applied_kernel_frame)
            applied_kernel_frames += [applied_kernel_frame]
        # 動画クラスとして返す
        applied_kernel_video = Video(applied_kernel_frames)
        return applied_kernel_video

    def to_time_series(self):
        """
        動画の各要素を時系列で並べる
        """
        time_series_values = []
        for i in range(self.frames[0].frame.shape[0]):
            for j in range(self.frames[0].frame.shape[1]):
                a_time_series_value = []
                for frame in self.frames:
                    a_time_series_value += [frame.frame[i][j]]
                time_series_values += [a_time_series_value]
        time_series_list = []
        for i, t in enumerate(time_series_values):
            time_series = TimeSeries(t)
            time_series_list += [time_series]
        return time_series_list

class Frame:
    """
    フレーム（画像）を扱うクラス
    できること:
    - 画像を表示することができる
        - show
    - 画像を保存することができる
        - save
    """
    def __init__(self, frame):
        self.frame = np.array(frame).astype(np.uint8)
        self.shape = self.frame.shape
        self.h = self.frame.shape[0]
        self.w = self.frame.shape[1]

    def __getitem__(self, index):
        return Frame(self.frame[index])

    def show(self):
        """
        画像の表示
        """
        plt.imshow(self.frame)
        plt.show()

    def save_frame(self, path):
        """
        画像の保存
        """
        cv2.imwrite(path, self.frame)

    def test_kernel(self, size=5):
        """
        カーネルを適用したときにどんな感じになるかを視覚化
        """
        kernel_size_x = int(self.frame.shape[0] / size)
        kernel_size_y = int(self.frame.shape[1] / size)
        print ("kernel_size_x", kernel_size_x)
        print ("kernel_size_y", kernel_size_y)
        # カーネルを作成する
        count_x = 0
        count_y = 0
        for x in range(0, self.w, size):
            cv2.line(self.frame, (x, 0), (x, self.h), (255,255,255), 1)
            count_x += 1
        for y in range(0, self.h, size):
            cv2.line(self.frame, (0, y), (self.w, y), (255,255,255), 1)
            count_y += 1
        print (count_x * count_y)
        plt.imshow(self.frame)
        plt.show()
        return kernel_size_x, kernel_size_y

    def apply_kernel(self, kernel_name='bgr_mean', size=5):
        """
        カーネルを適用する
        """
        # カーネルの読み込み
        kernel_size_x = int(self.frame.shape[0] / size)
        kernel_size_y = int(self.frame.shape[1] / size)
        kernel = Kernel(kernel_name, size=size)
        applied_kernel_frame = kernel(self.frame)
        # カーネルを適用した後の画像を生成
        applied_kernel_frame = np.array(applied_kernel_frame).reshape([kernel_size_y, kernel_size_x])
        return Frame(applied_kernel_frame)

class TimeSeries:
    def __init__(self, values):
        # 時系列データ
        self.values = values

    def fft(self, original=True, absolute=False):
        """
        フーリエ変換をして結果を返す
        """
        fft_list = np.fft.fft(a=self.values)
        if not original:
            fft_list = fft_list[1:len(self.values)//2]
        if abs:
            fft_list = abs(fft_list)
        return fft_list

    def amplitude(self):
        """
        フーリエ変換をして振幅を返す
        """
        fft = self.fft()
        amp = np.abs(fft/len(self.values)/2)
        return amp

    def freq(self):
        """
        振動数を返す
        """
        freq = np.fft.fftfreq(len(self.values), 1/15)
        return freq

    def plot(self):
        """
        画像化
        """
        plt.plot(self.values)
        plt.show()
