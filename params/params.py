import glob
import os
from config import Config

class SelectParams:
    def __init__(self, min_pt, inlier_ratio):
        self.min_pt = min_pt
        self.inlier_ratio = inlier_ratio


class Data:
    def __init__(self):
        self.train_files = dict()
        self.selected_files = dict()
        self.test_video = dict()

        path = Config.train_img_path
        self.train_files["tun"] = glob.glob(os.path.join(path, "tun*.png"))
        self.train_files["uturn"] = glob.glob(os.path.join(path, "uturn*.png"))
        self.train_files["park"] = glob.glob(os.path.join(path, "park*.png"))
        self.train_files["avoid"] = glob.glob(os.path.join(path, "avoid*.png"))
        # self.train_files["dont"] = glob.glob(os.path.join(path, "dont*.png"))

        select_path = Config.selected_img_path
        self.selected_files["avoid"] = glob.glob(os.path.join(select_path, "avoid*.png"))
        self.selected_files["park"] = glob.glob(os.path.join(select_path, "park*.png"))
        self.selected_files["uturn"] = glob.glob(os.path.join(select_path, "uturn*.png"))
        # self.selected_files["dont"] = glob.glob(os.path.join(select_path, "dont*.png"))
        self.selected_files["tun"] = glob.glob(os.path.join(select_path, "tun*.png"))

        test_video_path = Config.video_path
        self.test_video["avoid"] = os.path.join(test_video_path, "avoid.mp4")
        self.test_video["park"] = os.path.join(test_video_path, "park.mp4")
        self.test_video["uturn"] = os.path.join(test_video_path, "uturn.mp4")
        self.test_video["tun"] = os.path.join(test_video_path, "tun.mp4")


def set_params(category):
    param = {"avoid": SelectParams(8, 10.),
              "park": SelectParams(6, 15.),
              "uturn": SelectParams(4, 20.),
              # "dont": SelectParams(10, 5.0),
              "tun": SelectParams(5, 20.),
             }
    min_pt = param[category].min_pt
    inlier_ratio = param[category].inlier_ratio
    return min_pt, inlier_ratio


