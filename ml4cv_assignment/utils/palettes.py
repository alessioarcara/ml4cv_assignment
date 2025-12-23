import numpy as np

STREETHAZARD_COLORS = np.array(
    [
        [0, 0, 0],  # unlabeled: 0,
        [70, 70, 70],  # building: 1,
        [190, 153, 153],  # fence: 2,
        [250, 170, 160],  # other: 3,
        [220, 20, 60],  # pedestrian: 4,
        [153, 153, 153],  # pole: 5,
        [157, 234, 50],  # road line: 6,
        [128, 64, 128],  # road: 7,
        [244, 35, 232],  # sidewalk: 8,
        [107, 142, 35],  # vegetation: 9,
        [0, 0, 142],  # car: 10,
        [102, 102, 156],  # wall: 11,
        [220, 220, 0],  # traffic sign: 12,
        [60, 250, 240],  # anomaly:cityscapes 13,
    ]
)


CITYSCAPES_COLORS = np.array(
    [
        [128, 64, 128],  # 0: road
        [244, 35, 232],  # 1: sidewalk
        [70, 70, 70],  # 2: building
        [102, 102, 156],  # 3: wall
        [190, 153, 153],  # 4: fence
        [153, 153, 153],  # 5: pole
        [250, 170, 30],  # 6: traffic light
        [220, 220, 0],  # 7: traffic sign
        [107, 142, 35],  # 8: vegetation
        [152, 251, 152],  # 9: terrain
        [70, 130, 180],  # 10: sky
        [220, 20, 60],  # 11: person
        [255, 0, 0],  # 12: rider
        [0, 0, 142],  # 13: car
        [0, 0, 70],  # 14: truck
        [0, 60, 100],  # 15: bus
        [0, 80, 100],  # 16: train
        [0, 0, 230],  # 17: motorcycle
        [119, 11, 32],  # 18: bicycle
    ]
)
