#!/usr/bin/python
#-*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import c3d_model


event_dic = {'background': 0,
             'shot': 1,
             'corner': 2,
             'free-kick': 3,
             'yellow-card': 4,
             'foul': 5,
             'goal': 6,
             'offside': 7,
             'overhead-kick': 8,
             'solo-drive': 9,
             'penalty-kick': 0,
             'red-card': 11,
             } 
             

def readTestFile(batch_size, num_frames):

    f = open("../../dataset/event_test.txt", 'r')
    lines = list(f)
    random.shuffle(lines)
    events = []
    labels = []

    for l in range(batch_size):

        frames = []
        event = lines[l].strip('\n').split(' ')
        event_type = event[0]
        start = int(event[1])
        video = event[3]
        labels.append(int(event_dic[str(event_type)]))
        cap = cv2.VideoCapture("../../dataset/video/" + video)

        for n in range(num_frames):

            block = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, start + n)  # 设置要获取的帧号
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
    f.close()
    events = np.array(events).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    return events, labels


def readFile():
    f = open("../../dataset/event_train.txt", 'r')
    lines = list(f)
    random.shuffle(lines)
    f.close()

    return lines


def readTrainData(batch, lines, batch_size, num_frames):

    events = []
    labels = []

    for b in range(batch*batch_size, batch*batch_size + batch_size):

        frames = []

        event = lines[b].strip('\n').split(' ')
        event_type = event[0]
        start = int(event[1])
        end = int(event[2])
        video = event[3]
        labels.append(int(event_dic[str(event_type)]))
        cap = cv2.VideoCapture("../../dataset/video/" + video)

        if (end - start) <= 32:
            first_frame = start
            skip_frame = 1
        elif (end - start) < 64:
            first_frame = random.randint(start, end - 32)
            skip_frame = 1
        elif (end - start) < 128:
            first_frame = random.randint(start, end - 64)
            skip_frame = 2
        else:
            first_frame = random.randint(start, end - 128)
            skip_frame = 4

        for n in range(num_frames):
            block = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + skip_frame*n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (c3d_model.width, c3d_model.height), interpolation=cv2.INTER_CUBIC)

            # b = per_image_standard(b, c3d_model.width * c3d_model.height)
            block.append(per_image_standard(
                cv2.resize(b[0:224, 0:224, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[0:224, 112:336, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[0:224, 224:448, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[0:224, 336:560, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 0:224, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 112:336, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 224:448, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))
            block.append(per_image_standard(
                cv2.resize(b[112:336, 336:560, :], (112, 112), interpolation=cv2.INTER_CUBIC), 112*112))

            frames.append(block)

        events.append(frames)
    events = np.array(events).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    return events, labels


def per_image_standard(image, num_rgb):
    mean = np.mean(image)
    stddev = np.std(image)
    image = (image - mean)/(max(stddev, 1.0/np.sqrt(num_rgb)))
    return image
