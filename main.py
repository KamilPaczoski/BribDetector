from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw, ImageFont, ImageColor


class BirbDetector:
    fps = 30
    resolution = (654, 480)

    def __init__(self):
        self.purge()
        self.video_splitter()
        self.hand_drawing()
        self.yolo()
        self.editing_video()
        self.video_merger()


    @staticmethod
    def purge():
        try:
            for file in os.listdir('frames'):
                os.remove('frames/' + file)
            os.remove('edited_video.mp4')
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
        for i in range(0, len(os.listdir('frames'))):
            img = cv2.imread('frames/' + str(i) + '.jpg')
            video.write(img)

        video.release()

    @staticmethod
    def yolo():
        model = YOLO("yolov8n.pt")
        model.train(data='coco128.yaml', epochs=4, imgsz=640, patience=2, batch=16, workers=8, device='0')
        model.predict()
        model.save('yolov8n.pt')
        output = model('frames/0.jpg')
        print(output.xyxy[0])
        print("__________________________")
        print(model.val())

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
        model.predict('frames', save_dir='frames_edited', save=True, conf=0.5)

    def compare(self):
        auto = Image.open('frames_edited/0.jpg')
        hand = Image.open('frames_hand/0.jpg')


if __name__ == '__main__':
    BirbDetector()
