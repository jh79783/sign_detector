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

        self.test_path = {"avoid": os.path.join(Config.video_path, "avoid.mp4"),
                          "park": os.path.join(Config.video_path, "park.mp4"),
                          "uturn": os.path.join(Config.video_path, "uturn.mp4"),
                          "tun": os.path.join(Config.video_path, "tun.mp4")}


def set_params(category):
    param = {"avoid": SelectParams(4, 30.0),
              "park": SelectParams(5, 10.0),
              "uturn": SelectParams(5, 50.0),
              # "dont": SelectParams(10, 5.0),
              "tun": SelectParams(4, 50.0),
             }
    min_pt = param[category].min_pt
    inlier_ratio = param[category].inlier_ratio
    return min_pt, inlier_ratio


