from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw, ImageFont


class BirbDetector:
    fps = 30
    resolution = (850, 480)

    def __init__(self):
        self.purge()
        self.video_splitter()
        self.hand_drawing()
        self.yolo()
        self.editing_video()
        self.video_merger()
        self.iou()

    @staticmethod
    def purge():
        try:
            for file in os.listdir('frames'):
                os.remove('frames/' + file)
            os.remove('edited_video.mp4')
            os.remove('runs')
        except FileNotFoundError:
            pass

    def video_splitter(self):
        vc = cv2.VideoCapture('video.mp4')
        vc.set(cv2.CAP_PROP_FPS, self.fps)
        c = 0
        read = vc.isOpened()
        while read:
            read, frame = vc.read()
            cv2.imwrite('frames/' + str(c) + '.jpg', frame)
            c += 1
        vc.release()

    def video_merger(self):
        video = cv2.VideoWriter('edited_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.resolution)
        for i in range(0, len(os.listdir('runs/detect/predict'))):
            img = cv2.imread('runs/detect/predict/' + str(i) + '.jpg')
            video.write(img)
        video.release()

    @staticmethod
    def yolo():
        model = YOLO("yolov8n.pt")
        model.train(data='coco128.yaml', epochs=3, imgsz=640, patience=2)

        print(model('frames/0.jpg'))

    @staticmethod
    def hand_drawing():
        img = Image.open('frames/0.jpg')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', 25)
        draw.rectangle(((155, 180), (260, 290)), None, 'purple', 2)
        draw.rectangle(((400, 255), (500, 370)), None, 'green', 2)
        draw.rectangle(((570, 120), (650, 200)), None, 'orange', 2)
        draw.text((0, 0), 'Three birbs', font=font)
        img.save('frames_hand/0.jpg')

    def editing_video(self):
        vc = cv2.VideoCapture('video.mp4')
        vc.set(cv2.CAP_PROP_FPS, self.fps)
        model = YOLO("yolov8n.pt")
        model.predict('frames', save_dir='frames_edited', save=True, save_txt=True, conf=0.5)

    def iou(self):
        with open('runs/detect/predict3/labels/0.txt', 'r') as f:
            lines = f.readlines()
        a = self.resolution[0]
        b = self.resolution[1]
        for line in lines:
            line = line.split(' ')
            union_rect = [min(float(line[1]) * a, float(line[3]) * a), min(float(line[2]) * b, float(line[4]) * b),
                          max(float(line[1]) * a, float(line[3]) * a), max(float(line[2]) * b, float(line[4]) * b)]
        int_rect = [max(155, 400, 570), max(180, 255, 120), min(260, 500, 650), min(290, 370, 200)]
        int_area = (int_rect[2] - int_rect[0]) * (int_rect[3] - int_rect[1])
        union_area = (union_rect[2] - union_rect[0]) * (union_rect[3] - union_rect[1])
        iou = int_area / union_area
        print(iou)


if __name__ == '__main__':
    BirbDetector()
