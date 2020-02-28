from dain_lib import Dain
from read_video import Video
from pngtovideo import read_png_to_video
import sys
import progressbar
import cv2
import copy
import os, shutil
from datetime import datetime
import pdb

# Set video to be interpolated from passed in argument
try:
    vid_str = sys.argv[1]
except IndexError:
    vid_str = 'test.mp4'
# Set output file from passed in argument
try:
    new_video = sys.argv[2]
except IndexError:
    new_video = vid_str.split('.')[0]+'_new.avi'
# Set frames multiplier from passed in argument
try:
    num_passes = float(sys.argv[3])
except IndexError:
    num_passes = 2
print('Attempting to interpolate '+vid_str+' to '+new_video)


def del_all_files_dir(folder):
    """
    For clearing out temp directory
    :param folder:
    :return:
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def use_dain_to_interpolate(video, new_video, num_passes, dain=Dain()):
    """
    Use DAIN algorithm to create interpolated frames
    :param video: input video
    :param new_video: where to save
    :param num_passes: number of times to double fps
    :param dain: dain processing object
    :return: null
    """
    original_fps = video.fps()
    original_num_frames = video.total_frames()
    num_passes = int(num_passes)
    new_fps = original_fps * 2 * num_passes
    # new_num_frames = round(new_fps * video.duration_in_sec(), 0) - 1
    new_num_frames = video.total_frames() * 2 * num_passes
    new_duration = new_num_frames / new_fps
    temp_image_dir = 'test_small_export_png'
    if not os.path.exists(temp_image_dir):
        os.mkdir(temp_image_dir)
    del_all_files_dir(temp_image_dir)

    print('\nVideo Stats:'
          '\nPath: {}'
          '\nfps: {}'
          '\nnum_frames: {}'
          '\nduration: {}'
          '\nConvert to: {}'
          '\nnew_fps: {}'
          '\nnew num_frames: {}'
          '\nnew duration: {}'
          '\n'
          .format(video.video_path(), original_fps, original_num_frames, video.duration_in_sec(),
                  new_video, new_fps, new_num_frames, new_duration)
          )

    pb_frames_to_process = 0
    for i in range(1, num_passes + 1):
        pb_frames_to_process += original_num_frames * 2 * i
    new_frame_counter = 0
    with progressbar.ProgressBar(max_value=pb_frames_to_process, redirect_stdout=True) as bar:
        for iteration in range(1, num_passes + 1):
            del_all_files_dir(temp_image_dir)
            current_fps_processing = video.fps() * 2
            print("**** currently increasing frames to {} fps ****".format(current_fps_processing))
            
            video_compare_frames = copy.copy(video)
            next(video_compare_frames)  # skip first frame
            for frame0 in video:
                # save first frame
                cv2.imwrite("{}/{:09d}.png".format(temp_image_dir, new_frame_counter), frame0)

                # grab next frame
                try:
                    frame1 = next(video_compare_frames)
                except StopIteration:
                    break

                # interpolate a frame in between
                begin_interpolation = datetime.now()
                frameI = dain.dain_interpolate(frame0, frame1)
                end_interpolation = datetime.now()

                # save interpolated frame
                new_frame_counter += 1
                cv2.imwrite("{}/{:09d}i.png".format(temp_image_dir,  new_frame_counter), frameI)
                print("Time to create frame {:09d}: {}"
                      .format(new_frame_counter, end_interpolation - begin_interpolation))

                bar.update(new_frame_counter)
                new_frame_counter += 1

            if os.path.exists(new_video):
                os.remove(new_video)
            read_png_to_video("{}/".format(temp_image_dir), new_video, current_fps_processing)
            del video
            video = Video(new_video)
            print("**** New video created {} at {} fps with {} frames ****"
                  .format(new_video, video.fps(), video.total_frames()))

    # video created, clear temp directory
    del_all_files_dir(temp_image_dir)

input_video = Video(vid_str)
use_dain_to_interpolate(input_video, new_video, num_passes)

