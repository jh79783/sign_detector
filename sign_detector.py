import cv2
import numpy as np
import os
import params.params as params
from config import Config

FLANN_INDEX_LSH = 6
matchesMask = None
good_points = None
test_path = os.path.join(Config.video_path, "avoid.mp4")


def main():
    keypoints_train = []
    descriptors_train = []
    categorys = []
    basenames = []
    orb = cv2.ORB_create()
    for category, files in params.Data().selected_files.items():
        for file in files:
            t_img = cv2.imread(file)
            t_img = cv2.resize(t_img, (350, 350))
            kp_t, des_t = orb.detectAndCompute(t_img, None)
            keypoints_train.append(kp_t)
            descriptors_train.append(des_t)
            basenames.append(os.path.basename(file))
            categorys.append(category)
    detect(categorys, keypoints_train, descriptors_train, basenames)


def detect(categorys, keypoints_train, descriptors_train, basenames):
    global matchesMask, good_points
    orb = cv2.ORB_create()
    ture_pos = 0
    ture_nag = 0
    fals_pos = 0
    fals_nag = 0
    count = 0
    cam = cv2.VideoCapture(test_path)
    while True:
        _, q_frame = cam.read()
        if cam.isOpened():
            resize_frame = cv2.resize(q_frame, dsize=(350, 350))
            resize_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("frame", resize_frame)
            cv2.waitKey(10)
            kp_q, des_q = orb.detectAndCompute(resize_frame, None)
            for cate, kp_t, des_t, name in zip(categorys, keypoints_train, descriptors_train, basenames):
                result = match_image(kp_t, des_t, kp_q, des_q, cate)
                if result is True:
                    draw_match(matchesMask, resize_frame, kp_q, kp_t, good_points, cate, name)
                    print(f"detect {cate} sign")
                    count = 0
                    video_name = os.path.basename(test_path)
                    # print(video_name)
                else:
                    count += 1
                if count > 30:
                    print("NOT DETECT SIGN")
                    count = 0

        else:
            print("cam error")
            return


def match_image(kp_t, des_t, kp_q, des_q, category):
    global matchesMask, good_points
    min_pt, inlier_ratio = params.set_params(category)
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_q, des_t, k=2)
    good_points = []
    for mat in matches:
        if len(mat) > 1:
            m, n = mat
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
    if len(good_points) > min_pt:
        query_pts = np.array([kp_q[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.array([kp_t[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, inlier_ratio)
        matchesMask = mask.ravel().tolist()
        if np.array(matchesMask).sum() > min_pt:
            return True
        else:
            return False


def draw_match(matchesMask, resize_frame, kp_q, kp_t, good_points, cate, name):

    res = None
    image = None

    for files in params.Data().selected_files[cate]:
        basename = os.path.basename(files)
        if name == basename:
            image = cv2.imread(files)
            image = cv2.resize(image, (350, 350))

    if image is not None:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

        res = cv2.drawMatches(resize_frame, kp_q, image, kp_t, good_points, res, **draw_params)
        cv2.imshow("draw", res)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()

