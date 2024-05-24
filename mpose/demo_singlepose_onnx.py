#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime
import os

def my_letter_box(img, model_size):  #
    size = (model_size, model_size)
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv.resize(img, (new_w, new_h))
    img = cv.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top

def getFileNames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L

def run_inference_nhwc(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]
    print(f"{image_width}, {image_height}")
    # Pre process:Resize, BGR->RGB, Reshape, float32 cast
    input_image,_,_,_ = my_letter_box(image, input_size)#cv.resize(image, dsize=(input_size, input_size))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    print(f"cv shape:{input_image.shape}")
    #input_image_t = input_image.reshape(-1, input_size, input_size, 3)
    #print(f"1*192*192*3 shape:{input_image_t.shape}")
    
    input_image_tt = np.expand_dims(input_image,0)
    print(f"1*192*192*3 shape:{input_image_tt.shape}")
    
    input_image = input_image_tt.astype('float32')
    np.savetxt('./input_crop_cv_nhwc_{}.txt'.format(input_size), input_image.flatten(), fmt='%0.8f')
    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    np.savetxt('./output_nhwc_{}.txt'.format(input_size), keypoints_with_scores.flatten(), fmt='%0.8f')
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Postprocess:Calc Keypoint
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores

def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]
    print(f"{image_width}, {image_height}")
    # Pre process:Resize, BGR->RGB, Reshape, float32 cast
    input_image, _, _, _ = my_letter_box(image, input_size) #cv.resize(image, dsize=(input_size, input_size))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    print(f"cv shape:{input_image.shape}")
    #img = (img_rsz - img_mean) * img_scale
    #img = np.asarray(img, dtype=np.float32)
    #img = np.expand_dims(img,0)
    #img = img.transpose(0,3,1,2)
    #input_image_tt = input_image.reshape(-1, 3, input_size, input_size)
    input_image_tt = np.expand_dims(input_image,0)
    print(f"1*192*192*3 shape:{input_image_tt.shape}")
    input_image_tt = input_image_tt.transpose(0,3,1,2)
    print(f"1*3*192*192 shape:{input_image_tt.shape}")
    #input_image_tt = input_image_tt.transpose(0,2,3,1)
    #print(f"1*192*192*3 shape:{input_image_tt.shape}")
    input_image = input_image_tt.astype('float32')
    np.savetxt('./input_crop_cv_{}.txt'.format(input_size), input_image.flatten(), fmt='%0.8f')
    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    np.savetxt('./output_{}.txt'.format(input_size), keypoints_with_scores.flatten(), fmt='%0.8f')
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # Postprocess:Calc Keypoint
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def main():
    input_path = "/home/leon/mount_point_c/githubs_new/movenet/movenet.pytorch/data/samples"
    save_dir = "./rst-1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default='/home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/movenet-only-onnx/3-int16-nhwc/model_float32.onnx',
    ) 
    #/home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/movenet-only-onnx/int16-err-include-vknn-preprocess/view_modified.onnx
    # modified_modified_modified model_float32
    # /home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/movenet-only-onnx/3-int16-nhwc
    # /home/leon/mount_point_c/acuity-toolkit-binary-6.24.7/movenet-only-onnx/int16-no-vknn-node/model_float32_include_preprocess.
    parser.add_argument(
        "--input_size",
        type=int,
        default=192,
        choices=[192, 256],
    )
    parser.add_argument("--keypoint_score", type=float, default=0.3)  #0.3

    args = parser.parse_args()
    model_path = args.model
    input_size = args.input_size
    keypoint_score_th = args.keypoint_score

    # Initialize video capture
    ##cap = cv.VideoCapture(0)

    # Load model
    onnx_session = onnxruntime.InferenceSession(model_path)
    a = 0
    
    input_paths = getFileNames(input_path)
    for p in input_paths:
        
        start_time = time.time()
        
        frame = cv.imread(p)
        ret = True
        # Capture read
        #ret, frame = cap.read()
        if not ret:
            break
        #debug_image = copy.deepcopy(frame)

        # Inference execution
        ##keypoints, scores = run_inference(onnx_session,input_size,frame,)
        keypoints, scores = run_inference_nhwc(onnx_session,input_size,frame,)
        elapsed_time = time.time() - start_time

        # Draw
        frame = draw_debug(
            frame,
            elapsed_time,
            keypoint_score_th,
            keypoints,
            scores,
        )
        
        ## key = cv.waitKey(1)
        ##if key == 27:  # ESC
            ##break
        ## cv.imshow('MoveNet(singlepose) Demo', debug_image)
        cv.imwrite(os.path.join(save_dir, p.split("/")[-1]), frame)
        print("finished.")
    ## cap.release()
    ## cv.destroyAllWindows()
    print("exit.")

def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    connect_list = [
        [0, 1, (255, 0, 0)],  # nose → left eye
        [0, 2, (0, 0, 255)],  # nose → right eye
        [1, 3, (255, 0, 0)],  # left eye → left ear
        [2, 4, (0, 0, 255)],  # right eye → right ear
        [0, 5, (255, 0, 0)],  # nose → left shoulder
        [0, 6, (0, 0, 255)],  # nose → right shoulder
        [5, 6, (0, 255, 0)],  # left shoulder → right shoulder
        [5, 7, (255, 0, 0)],  # left shoulder → left elbow
        [7, 9, (255, 0, 0)],  # left elbow → left wrist
        [6, 8, (0, 0, 255)],  # right shoulder → right elbow
        [8, 10, (0, 0, 255)],  # right elbow → right wrist
        [11, 12, (0, 255, 0)],  # left hip → right hip
        [5, 11, (255, 0, 0)],  # left shoulder → left hip
        [11, 13, (255, 0, 0)],  # left hip → left knee
        [13, 15, (255, 0, 0)],  # left knee → left ankle
        [6, 12, (0, 0, 255)],  # right shoulder → right hip
        [12, 14, (0, 0, 255)],  # right hip → right knee
        [14, 16, (0, 0, 255)],  # right knee → right ankle
    ]

    # Connect Line
    for (index01, index02, color) in connect_list:
        if scores[index01] > keypoint_score_th and scores[
                index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, color, 2)

    # Keypoint circle
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 3, (0, 255, 0), -1)

    # Inference elapsed time 
    
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()
