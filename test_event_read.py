#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import c3d_model

segment = 64


def readFile():
    f = open("../../dataset/test.txt", 'r')
    lines = list(f)
    videos = []
    for l in range(0, 22):
        video_id = lines[l].strip('\n')
        cap = cv2.VideoCapture("../../dataset/video/" + video_id)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        f_init_arr = np.arange(500, count - segment + 1, int(segment * 0.25))
        videos.append([video_id, f_init_arr])
    f.close()

    return videos


def readTrainData(batch, line, batch_size):

    events = []
    segments = []

    for b in range(batch * batch_size, batch * batch_size + batch_size):

        frames = []
        video = line[0]
        start = int(line[1][b])
        cap = cv2.VideoCapture("../../dataset/video/" + video)

        if segment == 32:
            skip_frame = 1
            first_frame = start
        elif segment == 64:
            first_frame = random.randint(start, start + 31)
            skip_frame = 1
        elif segment == 128:
            first_frame = random.randint(start, start + 63)
            skip_frame = 2
        elif segment == 256:
            first_frame = random.randint(start, start + 127)
            skip_frame = 4

        for n in range(32):
            block = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + skip_frame*n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (c3d_model.width, c3d_model.height), interpolation=cv2.INTER_CUBIC)

            block.append(per_image_standard(
                cv2.resize(b[0:224, 0:224, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[0:224, 112:336, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[0:224, 224:448, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[0:224, 336:560, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 0:224, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 112:336, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 224:448, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 336:560, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112 * 112))

            frames.append(block)

        events.append(frames)
        segments.append([video, str(start), str(start+segment)])
    events = np.array(events).astype(np.float32)
    print(segments[0])
    return events, segments


def per_image_standard(image, num_rgb):
    mean = np.mean(image)
    stddev = np.std(image)
    image = (image - mean) / (max(stddev, 1.0 / np.sqrt(num_rgb)))
    return image
