import cv2
import os
import numpy as np


recoginizer = cv2.face.LBPHFaceRecognizer_create()
recoginizer.read('trainer/trainer.yml')

cascadePath = "haarcascade_frontalface_default.xml"
faecCascade = cv2.CascadeClassifier(cascadePath)

# Kiểu chữ.
font = cv2.FONT_HERSHEY_SIMPLEX

# 
id = 0

# Tên đặt cho ids.
names = ['None', 'Phuc',]

# Bật camera.
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Chiều rộng camera.
cam.set(4, 480) # Chiều cao canera.

# Xác định kích thước cửa sổ tối thiểu được nhận dạng như một khuôn mặt.
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faecCascade.detectMultiScale(
        gray,
        scaleFactor= 1.2,
        minNeighbors= 5,
        minSize= (int(minW), int(minH)),
    )

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recoginizer.predict(gray[y:y+h, x:x+w])

        # kiểm tra xem độ tin cậy có thấp hơn 100 ==> “0" là phù hợp hoàn hảo không
        if (confidence < 100):
            id = names[id]
            confidence =" {0}%".format(round(100 - confidence))

        else:
            id = "unknown"
            confidence =" {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)
    
    cv2.imshow('Cửa sổ camera', img)

    k = cv2.waitKey(10) & 0xff #  Nhấn ESC để thoát cửa sổ amera
    if k == 27:
        break
# 
print("\n [INFO] Đang thoát chương trình và dọn dẹp công cụ")
cam.release()
cv2.destroyWindow()