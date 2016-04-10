# -*- coding: utf-8 -*-
import os
import shutil

SPEED_COEFFICIENT = 0.05  # This is also in deep_drive.h
SPIN_THRESHOLD = 0.01  # This is also in Agent.h
INPUT_DIR  = 'D:\\data\\gtav\\keep\\'
OUTPUT_DIR = 'D:\\data\\gtav\\4hz_spin_speed_001_clean\\'
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'images')
LABEL_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'labels.txt')


def go():
    directory = INPUT_DIR
    dirs = sorted(os.listdir(directory))
    dirs.remove('img')
    dirs.remove('dat')
    if os.path.exists(LABEL_OUTPUT_FILENAME):
        os.remove(LABEL_OUTPUT_FILENAME)
    if os.path.exists(IMAGE_OUTPUT_DIR):
        shutil.rmtree(IMAGE_OUTPUT_DIR)
    os.makedirs(IMAGE_OUTPUT_DIR)
    index = 0
    for d in dirs:
        dir_path = os.path.join(directory, d)
        index = add_to_label_file(dir_path, dir_path, index)
    add_to_label_file(os.path.join(directory, 'dat'), os.path.join(directory, 'img'), index)


def copy_image(path, new_image_name):
    shutil.copyfile(path, os.path.join(IMAGE_OUTPUT_DIR, new_image_name))


def file_sort(filename):
    step = int(filename.split('_')[1].split('.')[0])
    return step


def add_to_label_file(dat_dir, img_dir, index):
    files = os.listdir(dat_dir)
    files = [f for f in files if f.startswith('dat')]
    files = files[1:]  # Skip the first, usually black image
    files = sorted(files, key=file_sort)
    last_speed = 0.0   # Start at average (10kph) - first speed_change will be wrong
    with open(LABEL_OUTPUT_FILENAME, 'a') as out_file:
        for j, f in enumerate(files):
            file_path = os.path.join(dat_dir, f)
            orig_image_path = os.path.join(img_dir, f.replace('dat_', 'img_').replace('.txt', '.bmp'))
            if not os.path.exists(orig_image_path):
                print('Could not find image for ' + file_path)
            else:
                image_name = 'img_%s.bmp' % get_file_num(index)
                copy_image(orig_image_path, image_name)
                with open(file_path, 'r') as content_file:
                    content = content_file.read()
                    _, spin, speed = content.split(', ')
                    spin = float(spin[6:])
                    if spin <= -SPIN_THRESHOLD:
                        direction = -1.0
                    elif spin >= SPIN_THRESHOLD:
                        direction = 1.0
                    else:
                        direction = 0.0
                    speed = float(speed[7:])
                    speed *= SPEED_COEFFICIENT
                    speed_change = speed - last_speed
                    last_speed = speed
                    index += 1
                    out_file.write('%s %f %f %f %f\n' % (image_name, speed, speed_change, spin, direction))

    return index


def get_file_num(index):
    return str(index).zfill(9)

if __name__ == '__main__':
    go()
