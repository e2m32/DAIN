import cv2
import sys
import os
from spinner import Spinner

IMG_TYPE = 'png'

# Set video to be converted
try:
    vid_str = sys.argv[1]
    vidcap = cv2.VideoCapture(vid_str)
except IndexError:
    vid_str = 'test.mp4'
    vidcap = cv2.VideoCapture(vid_str)
# Set output directory
try:
    save_dir = sys.argv[2]
except IndexError:
    # save_dir = vid_str+'_frames'
    save_dir = 'test_export_png'

print('Attempting to convert '+vid_str+' to '+save_dir)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def find_fps(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)

    print("Frames per second: {0}".format(fps))
    return fps


# useful to capture at a slower frame rate
def get_frame(video, sec):
    video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    has_frames, image = video.read()
    if has_frames:
        cv2.imwrite(os.path.join(save_dir, str(count)+"."+IMG_TYPE), image)
    return has_frames


def print_all_frames(video, save_dir):
    count = 1
    while (video.isOpened()):
        has_frames, frame = video.read()
        if has_frames:
            cv2.imwrite(os.path.join(save_dir, str(count) + "." + IMG_TYPE), frame)
        else:
            break
        # keyboard break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1


fps = find_fps(vidcap)
with Spinner():
    print_all_frames(vidcap, save_dir)
# sec = 0
# count = 1
# success = get_frame(vidcap, sec)
# while success:
#     count = count + 1
#     sec = sec + fps
#     sec = round(sec, 2)
#     success = get_frame(vidcap, sec)

vidcap.release()
print('Conversion completed for '+vid_str+'. All frames in '+save_dir+' with <num>.'+IMG_TYPE+' format')
