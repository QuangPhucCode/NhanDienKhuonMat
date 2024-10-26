import cv2
import os
from PIL import Image
import numpy as np




# 
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Chức năng để lấy ảnh và nhãn dữ liệu.
def getImagesAndLabels(path):

    imagePath =  [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples =[]
    ids = []

    for imagePath in imagePath:

        PIL_img = Image.open(imagePath).convert('L') # Chuyển sang chế độ màu xám.
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    
    return faceSamples, ids

print("\n [INFO] Đang quét khuôn mặt. Quá trình sẽ mất vài giây. Vui lòng chờ ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# lưu mô hinh vào trainer/trainer.yml.
recognizer.write('trainer/trainer.yml') 

# In ra số mặt được quét và kết thúc chương trình.
print("\n [INFO] {0} khuôn mặt đã được quét. Đang kết thúc chương trình ...".format(len(np.unique(ids))))

