import cv2
import os


class Video(object):
    def __init__(self, video_path_str):
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.__cv2_version = int(major_ver)

        self.__video_path = video_path_str
        self.__video = cv2.VideoCapture(video_path_str)
        self.__fps = self.__find_fps__()
        self.__total_frames = self.__find_num_frames__()
        self.__duration_sec = self.__find_duration_sec__()
        self.__iter_index = 0

    def __find_fps__(self):
        if self.__cv2_version < 3:
            fps = self.__video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = self.__video.get(cv2.CAP_PROP_FPS)

        return fps

    def __find_num_frames__(self):
        if self.__cv2_version < 3:
            return int(self.__video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        else:
            return int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __find_duration_sec__(self):
        return self.__total_frames / self.__fps

    def __iter__(self):
        return self

    def __next__(self):
        if self.__iter_index == self.__total_frames:
            raise StopIteration

        # Iterate through all frames
        if self.__cv2_version < 3:
            self.__video.set(cv2.CV_CAP_PROP_POS_FRAMES, self.__iter_index)
        else:
            self.__video.set(cv2.CAP_PROP_POS_FRAMES, self.__iter_index)
        res, frame = self.__video.read()

        self.__iter_index += 1

        if res:
            return frame
        else:
            # Was not able to return image from that position
            return 0

    def get_frame(self, sec):
        if self.__cv2_version < 3:
            self.__video.set(cv2.CV_CAP_PROP_POS_MSEC, sec * 1000)
        else:
            self.__video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

        has_frames, image = self.__video.read()
        if has_frames:
            return image
        return False

    def export_all_frames(self, save_dir, file_type):
        count = 1
        self.__video.set(cv2.CV_CAP_PROP_POS_FRAMES, 0)
        while self.__video.isOpened():
            has_frames, frame = self.__video.read()
            if has_frames:
                cv2.imwrite(os.path.join(save_dir, str(count) + "." + file_type), frame)

            # keyboard break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1

    def video_path(self):
        return self.__video_path

    def fps(self):
        return self.__fps

    def total_frames(self):
        return self.__total_frames

    def duration_in_sec(self):
        return self.__duration_sec
