import os
import sys

import cv2
import numpy as np

MINIMUM_SIMILARITY_THRESHOLD = 0.4

def process_video(video_path, min_similarity, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    match_frame = None
    prev_frame = None

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        i += 1

        print(f"Frame {i}: ", end='')

        if prev_frame is None:
            prev_frame = frame
            match_frame = frame
            print("Initial frame")
            continue

        cv2.imshow('Frame', frame)

        similarity = orb_similarity(match_frame, frame)

        print(f"Similarity: {similarity}, ", end='')

        if similarity < min_similarity:
            match_frame = prev_frame

            if(not os.path.isdir(output_folder)):
                os.mkdir(output_folder)

            cv2.imwrite(os.path.join(output_folder, os.path.basename(video_path)+"-"+str(i)+".png"), match_frame)
            print("Lost continuity, new match frame")
        else:
            print("Skipped!")

        prev_frame = frame

        if cv2.waitKey(1) == ord('q'): break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


orb = cv2.ORB_create()

def orb_similarity(img1, img2):
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    similarity = len(matches) / min(len(kp1), len(kp2))
    return similarity

if len(sys.argv) != 4:
    print("Usage: python main.py <video file> <min similarity threshold> <output folder>")
else:
    process_video(sys.argv[1], float(sys.argv[2]), sys.argv[3])