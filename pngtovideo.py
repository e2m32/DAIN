import cv2
import os
from os.path import isfile, join
import re
from spinner import Spinner


def read_png_to_video(read_dir, vid_path, fps):

    if not os.path.exists(read_dir):
        raise Exception(read_dir+'does not exist!')

    files = [f for f in os.listdir(read_dir) if isfile(join(read_dir, f))]
    if len(files) < 1:
        raise Exception("Directory is empty.")

    # for sorting the file names properly
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    with Spinner():
        i = 0
        filename = read_dir + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        out_video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        while i <= len(files) - 1:
            # write frame to video
            filename = read_dir + files[i]
            img = cv2.imread(filename)
            out_video.write(img)

            # read next image
            img = cv2.imread(filename)

            # keyboard break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            i += 1

    out_video.release()
