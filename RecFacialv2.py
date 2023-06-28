from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import os

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

path = 'Images'
name_list = []
embedding_list = []

def collate_fn(x):
    return x[0]

def dataset():
    global name_list
    global embedding_list
    dataset = datasets.ImageFolder('Images')
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)

    for img, idx in loader:
        face, prob = mtcnn0(img, return_prob=True)
        if face is not None and prob>0.92:
            emb = resnet(face.unsqueeze(0))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt')

    load_data = torch.load('data.pt')
    embedding_list = load_data[0]
    name_list = load_data[1]

cam = cv2.VideoCapture(0)
dataset()
while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha na captura do quadro, tente novamente")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)
                min_dist_idx = dist_list.index(min_dist)
                name = name_list[min_dist_idx]

                box = boxes[i]

                original_frame = frame.copy()

                if min_dist < 0.90:
                    frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    frame = cv2.putText(frame, "Nao cadastrado", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    cv2.imshow("IMG", frame)

    k = cv2.waitKey(1)
    if k == ord("q"):
        print('Esc pressionado, fechando...')
        break

    elif k == ord("a"):
        print('Digite seu nome:')
        name = input()

        if not os.path.exists(path + '/' + name):
            os.mkdir(path + '/' + name)

        img_name = path + "/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print("OK")
        dataset()

cam.release()
cv2.destroyAllWindows()