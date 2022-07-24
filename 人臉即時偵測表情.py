# 環境建置
# 參考建置鏈結
# CUDA, cudNN : https://www.youtube.com/watch?v=hHWkvEcDBO0&t=426s
# dlib : https://www.youtube.com/watch?v=-pZEDxDRyGQ
from fer import FER
import cv2
import dlib

detector1 = dlib.get_frontal_face_detector()
detector = FER()
# 讀取視訊鏡頭 內建為 0 , 外接為 1
# cap = cv2.VideoCapture(0)
# 讀取影片
# cap = cv2.VideoCapture('video.mp4')
# 1: True 0:false
while 1:
# ret 變數 : 取得下一幀圖片是否取得成功 布林值 , frame : 取得下一幀圖片
    ret, frame = cap.read()
    # frame = cv2.resize(frame,(0,0), fx = 0.3, fy = 0.3)
    if ret:
        face_rects = detector1(frame, 0)
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()

            emotion, score = detector.top_emotion(frame)
            cv2.putText(frame, emotion,  (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 250, 250), 2)
            cv2.imshow('video', frame)
        # print(emotion, score)
    else:
        break
    # keybroad input q will exit , waitkey 為等待鍵盤某個按鍵被按下
    if cv2.waitKey(1) == ord('q'):
        break
 