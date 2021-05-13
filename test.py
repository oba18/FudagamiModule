import FudagamiModule as fm
from data import Video
from data import Util
from kernel import Kernel
import cv2
import matplotlib.pyplot as plt
import numpy as np

src = '../team_OK/mps/experiment_videos/094448.mp4'
# 動画読み込み
video = Video(src=src, fps=15, start=0, end=15)
# これで[frames, frames, ...]みたいなframeのリストが返ってくるようにしたい
frames = video[0:15, 230:330, 400:500]
applied_kernel_video = frames.apply_kernel(kernel_name="bgr_mean", size=5)
time_series_video = applied_kernel_video.to_time_series()
fft_origin = time_series_video[0].fft(original=True)
fft_half = time_series_video[0].fft(original=False)
#  fft = time_series_video[0].amplitude()
freq = time_series_video[0].freq()
print(fft_origin)
print(fft_half)
print(freq)

#  data_list = []
#  for v in frames:
#      data = v.apply_kernel(kernel_name="bgr_mean")
#      data_list += [data]
#
#  # フレーム * 時間
#  applied_kernel_frames_list = np.array(data_list)
#
#  u = Util()
#  time_series_data = u.to_time_series(applied_kernel_frames_list)
#  print (time_series_data.shape)
#  print (time_series_data[0])
#  for tsd in time_series_data:
#      fft = np.fft.fft(tsd)
#      print (fft.shape)
#
