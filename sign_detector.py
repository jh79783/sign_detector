import cv2
import numpy as np
import os
import params.params as params
from config import Config
import time


FLANN_INDEX_LSH = 6
matchesMask = None
good_points = None
test_name = params.Data().test_video.items()
total_accuracy = dict()
elapsed = dict()


def main():
    keypoints_train = []
    descriptors_train = []
    categorys = []
    basenames = []
    orb = cv2.ORB_create()
    for category, files in params.Data().selected_files.items():
        for file in files:
            t_img = cv2.imread(file)
            t_img = cv2.resize(t_img, (640, 480), interpolation=cv2.INTER_AREA)
            kp_t, des_t = orb.detectAndCompute(t_img, None)
            keypoints_train.append(kp_t)
            descriptors_train.append(des_t)
            basenames.append(os.path.basename(file))
            categorys.append(category)
    detect(categorys, keypoints_train, descriptors_train, basenames)


def detect(categorys, keypoints_train, descriptors_train, basenames):
    global matchesMask, good_points
    orb = cv2.ORB_create()

    confusion_matrix = dict()
    count = 0
    for PN_category, filename in test_name:
        ture_pos = 0
        ture_nag = 0
        fals_pos = 0
        fals_nag = 0
        test_path = os.path.join(Config.video_path, filename)
        cam = cv2.VideoCapture(test_path)
        # cam = cv2.VideoCapture(1)
        start = time.time()
        while True:
            _, q_frame = cam.read()
            if _:
                resize_frame = cv2.resize(q_frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                # resize_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("frame", resize_frame)
                # cv2.waitKey(1)
                kp_q, des_q = orb.detectAndCompute(resize_frame, None)
                for cate, kp_t, des_t, name in zip(categorys, keypoints_train, descriptors_train, basenames):
                    result = match_image(kp_t, des_t, kp_q, des_q, cate)
                    if result is True:
                        draw_match(matchesMask, resize_frame, kp_q, kp_t, good_points, cate, name)
                        print(f"detect {cate} sign")
                        count = 0
                        if cate == PN_category:
                            ture_pos += 1
                        else:
                            fals_pos += 1

                    elif result is False:
                        count += 1
                        if cate != PN_category:
                            ture_nag += 1
                        else:
                            fals_nag += 1

                    if count > 30:
                        print("NOT DETECT SIGN")
                        count = 0

            else:
                print("end test")
                break
        end = time.time()
        elapsed[PN_category] = end - start
        cam.release()
        confusion_matrix[PN_category] = [[ture_pos], [fals_nag], [fals_pos], [ture_nag]]
        # total_accuracy[PN_category] = ((ture_pos + ture_nag) / (ture_pos + fals_nag + fals_pos + ture_nag)) * 100
    # confusion_matrix.append([ture_pos, fals_nag])
    # confusion_matrix.append([fals_pos, ture_nag])
    # confusion_matrix = np.array(confusion_matrix)
        print(confusion_matrix)
    record_result(confusion_matrix, total_accuracy, elapsed)


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
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    if image is not None:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

        res = cv2.drawMatches(resize_frame, kp_q, image, kp_t, good_points, res, **draw_params)
        # cv2.imshow("draw", res)
        # cv2.waitKey(1)


def record_result(confusion_matrix, total_accuracy, elapsed):
    TP = []
    TN = []
    FP = []
    FN = []
    for sign in confusion_matrix.keys():
        TP.append(confusion_matrix[sign][0])
        FN.append(confusion_matrix[sign][1])
        TN.append(confusion_matrix[sign][3])
        FP.append(confusion_matrix[sign][2])
    matrix = np.squeeze(np.array([TP, FN, FP, TN]))
    # print("conf shape", matrix)
    f = open(Config.result_path, "w", encoding="UTF8")
    f.write(f"confusion_matrix\n{matrix}\n")
    f.write(f"total_accuracy:{total_accuracy}\n")
    f.write(f"time:{elapsed}\n")
    f.close()

if __name__ == "__main__":
    main()

