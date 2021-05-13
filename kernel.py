class Kernel:
    
    """
    とにかく特徴量をとれるもの
    """
    def __init__(self, kernel_name='bgr_mean', size=5):
        self.kernel_dict = {
                'bgr_mean': self.bgr_mean,
                'hogehoge': self.hogehoge
                }
        self.kernel_name = kernel_name
        self.kernel = self.kernel_dict[kernel_name]
        self.size = size

    def __call__(self, frame):
        return self.kernel(frame)

    def bgr_mean(self, frame):
        """
        色の平均で取得する
        
        params
        ------
        frame: numpy.ndarray
            Frame型の画像
        """
        result = []
        for i in range(0, frame.shape[0], self.size):
            for j in range(0, frame.shape[1], self.size):
                result += [[frame[i:i+self.size, j:j+self.size].mean()]]
        return result

    def hogehoge(self, frame):
        return 'hogehoge'
