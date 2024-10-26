# Nhận diện khuôn mặt trong camera.
import cv2
import os

# Bật camera.
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Chiều rộng camera.
cam.set(4, 480) # Chiều cao canera.


# Khai báo dữ liệu 'haarcascade_frontalface_default.xml'.
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nhập id khuôn mặt của mỗi người.
face_id = input("\n Nhập ID khuôn mặt <return> ==> ")

print("\n [INFO] Đang khởi tạo camera ...")
count = 0

while True:

    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)  # Lật hình ảnh camera theo chiều dọc.1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Cửa sổ cam', frame)

    k = cv2.waitKey(100) & 0xff #  Nhấn ESC để thoát cửa sổ amera
    if k == 27:
        break
    elif count >= 50: # Chụp 50 2bức ảnh và thoát camera
        break

print("\n [INFO] Thoát")
cam.release()
cv2.destroyWindow()

