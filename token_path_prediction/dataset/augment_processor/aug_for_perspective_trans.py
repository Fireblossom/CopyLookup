import json
import random

from token_path_prediction.modules.perspective_transform import perspective_transform, coordinate_transform


class AugForPerspectiveTrans(object):
    '''
    segment进行随机切分
    注意：这里需要同时兼容4点和2点坐标
    '''

    def __init__(self, prob=0.2, max_angle=5):
        self.prob = prob
        self.max_angle = max_angle

    def process(self, sample):
        json_obj = sample['json']

        if random.random() > self.prob:
            return sample
        self.trans(json_obj)

        return sample

    def perspective_box_trans(self, ptList, M):
        '''
        这里需要兼容2点和4点
        '''
        if isinstance(ptList[0], list):
            for pt in ptList:
                x, y = coordinate_transform(pt, M)
                pt[0] = x
                pt[1] = y
        else:
            x0, y0, x1, y1 = ptList
            x, y = coordinate_transform([x0, y0], M)
            ptList[0] = x
            ptList[1] = y
            x, y = coordinate_transform([x1, y1], M)
            ptList[2] = y
            ptList[3] = y

    def trans(self, obj):

        max_angle = self.max_angle

        b_height = obj['img']['height']
        b_width = obj['img']['width']

        rnd = random.Random()
        anglex = rnd.normalvariate(0, 0.4) * 10
        angley = rnd.normalvariate(0, 0.4) * 10
        anglez = rnd.normalvariate(0, 0.4) * 5

        anglex = max(min(anglex, max_angle), -max_angle)
        angley = max(min(angley, max_angle), -max_angle)
        anglez = max(min(anglez, max_angle), -max_angle)

        M = perspective_transform([b_width, b_height],
                                  anglex=anglex,
                                  angley=angley,
                                  anglez=anglez,
                                  fov=42)

        bboxs = obj['document']

        for idx in range(len(bboxs)):
            bbox = bboxs[idx]

            bbox_list = bbox['box']

            self.perspective_box_trans(bbox_list, M)


if __name__ == '__main__':
    t = {"img": {"height": 1000, "width": 762, "image_path": "0011859695.png"}, "document": [
        {"id": 0, "box": [223, 62, 433, 79], "bndbox": [[223, 65], [433, 62], [433, 77], [223, 79]],
         "text": "AMERICAN BROADCASTING COMPANY", "words": [
            {"id": 0, "box": [223, 64, 233, 79], "bndbox": [[223, 65], [233, 64], [233, 78], [223, 79]], "text": "A"},
            {"id": 1, "box": [233, 64, 240, 78], "bndbox": [[233, 64], [240, 64], [240, 78], [233, 78]], "text": "M"},
            {"id": 2, "box": [240, 64, 249, 78], "bndbox": [[240, 64], [249, 64], [249, 78], [240, 78]], "text": "E"},
            {"id": 3, "box": [249, 64, 256, 78], "bndbox": [[249, 64], [256, 64], [256, 78], [249, 78]], "text": "R"},
            {"id": 4, "box": [256, 64, 263, 78], "bndbox": [[256, 64], [263, 64], [263, 78], [256, 78]], "text": "I"},
            {"id": 5, "box": [263, 64, 270, 78], "bndbox": [[263, 64], [270, 64], [270, 78], [263, 78]], "text": "C"},
            {"id": 6, "box": [270, 64, 278, 78], "bndbox": [[270, 64], [278, 64], [278, 78], [270, 78]], "text": "A"},
            {"id": 7, "box": [278, 64, 284, 78], "bndbox": [[278, 64], [284, 64], [284, 78], [278, 78]], "text": "N"},
            {"id": 8, "box": [284, 64, 291, 78], "bndbox": [[284, 64], [291, 64], [291, 78], [284, 78]], "text": " "},
            {"id": 9, "box": [291, 63, 298, 78], "bndbox": [[291, 64], [298, 63], [298, 78], [291, 78]], "text": "B"},
            {"id": 10, "box": [298, 63, 305, 78], "bndbox": [[298, 63], [305, 63], [305, 78], [298, 78]], "text": "R"},
            {"id": 11, "box": [305, 63, 313, 78], "bndbox": [[305, 63], [313, 63], [313, 78], [305, 78]], "text": "O"},
            {"id": 12, "box": [313, 63, 320, 78], "bndbox": [[313, 63], [320, 63], [320, 78], [313, 78]], "text": "A"},
            {"id": 13, "box": [320, 63, 329, 78], "bndbox": [[320, 63], [329, 63], [329, 77], [320, 78]], "text": "D"},
            {"id": 14, "box": [329, 63, 338, 77], "bndbox": [[329, 63], [338, 63], [338, 77], [329, 77]], "text": "C"},
            {"id": 15, "box": [338, 63, 344, 77], "bndbox": [[338, 63], [344, 63], [344, 77], [338, 77]], "text": "A"},
            {"id": 16, "box": [344, 63, 350, 77], "bndbox": [[344, 63], [350, 63], [350, 77], [344, 77]], "text": "S"},
            {"id": 17, "box": [350, 63, 358, 77], "bndbox": [[350, 63], [358, 63], [358, 77], [350, 77]], "text": "T"},
            {"id": 18, "box": [358, 63, 363, 77], "bndbox": [[358, 63], [363, 63], [363, 77], [358, 77]], "text": "I"},
            {"id": 19, "box": [363, 62, 369, 77], "bndbox": [[363, 63], [369, 62], [369, 77], [363, 77]], "text": "N"},
            {"id": 20, "box": [369, 62, 375, 77], "bndbox": [[369, 62], [375, 62], [375, 77], [369, 77]], "text": "G"},
            {"id": 21, "box": [375, 62, 381, 77], "bndbox": [[375, 62], [381, 62], [381, 77], [375, 77]], "text": " "},
            {"id": 22, "box": [381, 62, 388, 77], "bndbox": [[381, 62], [388, 62], [388, 77], [381, 77]], "text": "C"},
            {"id": 23, "box": [388, 62, 395, 77], "bndbox": [[388, 62], [395, 62], [395, 77], [388, 77]], "text": "O"},
            {"id": 24, "box": [395, 62, 403, 77], "bndbox": [[395, 62], [403, 62], [403, 77], [395, 77]], "text": "M"},
            {"id": 25, "box": [403, 62, 410, 77], "bndbox": [[403, 62], [410, 62], [410, 77], [403, 77]], "text": "P"},
            {"id": 26, "box": [410, 62, 418, 77], "bndbox": [[410, 62], [418, 62], [418, 77], [410, 77]], "text": "A"},
            {"id": 27, "box": [418, 62, 425, 77], "bndbox": [[418, 62], [425, 62], [425, 77], [418, 77]], "text": "N"},
            {"id": 28, "box": [425, 62, 433, 77], "bndbox": [[425, 62], [433, 62], [433, 77], [425, 77]],
             "text": "Y"}]},
        {"id": 1, "box": [259, 89, 400, 103], "bndbox": [[259, 89], [400, 89], [400, 103], [259, 103]],
         "text": "TELEVISION NETWORK", "words": [
            {"id": 29, "box": [259, 89, 268, 103], "bndbox": [[259, 89], [268, 89], [268, 103], [259, 103]],
             "text": "T"},
            {"id": 30, "box": [268, 89, 276, 103], "bndbox": [[268, 89], [276, 89], [276, 103], [268, 103]],
             "text": "E"},
            {"id": 31, "box": [276, 89, 286, 103], "bndbox": [[276, 89], [286, 89], [286, 103], [276, 103]],
             "text": "L"},
            {"id": 32, "box": [286, 89, 295, 103], "bndbox": [[286, 89], [295, 89], [295, 103], [286, 103]],
             "text": "E"},
            {"id": 33, "box": [295, 89, 302, 103], "bndbox": [[295, 89], [302, 89], [302, 103], [295, 103]],
             "text": "V"},
            {"id": 34, "box": [302, 89, 307, 103], "bndbox": [[302, 89], [307, 89], [307, 103], [302, 103]],
             "text": "I"},
            {"id": 35, "box": [307, 89, 313, 103], "bndbox": [[307, 89], [313, 89], [313, 103], [307, 103]],
             "text": "S"},
            {"id": 36, "box": [313, 89, 320, 103], "bndbox": [[313, 89], [320, 89], [320, 103], [313, 103]],
             "text": "I"},
            {"id": 37, "box": [320, 89, 327, 103], "bndbox": [[320, 89], [327, 89], [327, 103], [320, 103]],
             "text": "O"},
            {"id": 38, "box": [327, 89, 333, 103], "bndbox": [[327, 89], [333, 89], [333, 103], [327, 103]],
             "text": "N"},
            {"id": 39, "box": [333, 89, 340, 103], "bndbox": [[333, 89], [340, 89], [340, 103], [333, 103]],
             "text": " "},
            {"id": 40, "box": [340, 89, 347, 103], "bndbox": [[340, 89], [347, 89], [347, 103], [340, 103]],
             "text": "N"},
            {"id": 41, "box": [347, 89, 356, 103], "bndbox": [[347, 89], [356, 89], [356, 103], [347, 103]],
             "text": "E"},
            {"id": 42, "box": [356, 89, 364, 103], "bndbox": [[356, 89], [364, 89], [364, 103], [356, 103]],
             "text": "T"},
            {"id": 43, "box": [364, 89, 372, 103], "bndbox": [[364, 89], [372, 89], [372, 103], [364, 103]],
             "text": "W"},
            {"id": 44, "box": [372, 89, 381, 103], "bndbox": [[372, 89], [381, 89], [381, 103], [372, 103]],
             "text": "O"},
            {"id": 45, "box": [381, 89, 389, 103], "bndbox": [[381, 89], [389, 89], [389, 103], [381, 103]],
             "text": "R"},
            {"id": 46, "box": [389, 89, 400, 103], "bndbox": [[389, 89], [400, 89], [400, 103], [389, 103]],
             "text": "K"}]}], "uid": "0011859695"}
    obj = json.loads(t)
    # show_cdip(obj, '0.jpg')
    aug = AugForPerspectiveTrans()
    aug.trans(obj)

    print(json.dumps(obj, ensure_ascii=False))
