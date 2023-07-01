from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
from deepface.commons.functions import extract_faces
import os
import glob
from custom_files.utils import saveload, align_face_dlib_RGB_crop
import tensorflow as tf

db_path = "../FaceData/db_set/"
imgs_dir = "../FaceData/test_imgs_set/"
N = 1024
model_name = '../custom_files/model_weights/toplayer_n{}_h{}_b{}'.format(9325, N, 256)


class TopModel(tf.keras.Model):
    def __init__(self, num_class=152):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(N, activation='relu')
        self.fc2 = tf.keras.layers.Dense(N, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        # z = self.classifier(x)
        return x


finetune_model = TopModel()
# model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics='accuracy')
finetune_model.load_weights(model_name)


preds = []
trues = []
acc = []
skipped_img = []
files = os.listdir(imgs_dir)
detmodel = "retinaface"
embmodel1 = "Facenet512"  #512
embmodel2 = "SFace"  # 128
embmodel3 = "ArcFace"  # 512
#embmodel4 = "Dlib"  # 512
embmodel5 = "VGG-Face"  #2622
align = False
deepalign = not align

#model_embeddings = '{}_{}_{}_{}_{}_{}_embeddings'.format(detmodel,embmodel1,embmodel2, embmodel3,embmodel4, embmodel5)
model_embeddings = 'finetune_{}_align{}_{}_{}_{}_{}_embeddings'.format(detmodel,align, embmodel1,embmodel2, embmodel3, embmodel5)

# build embedding list if not avaiable
# if os.path.isfile(model_embeddings+'.pickle'):
#     overwrite = input('Overwrite existing {} ? (y/n)'.format(model_embeddings))
#
#     if overwrite.startswith('y'):
#         pass
#     else:
#         exit()

embeddings = []
noface = []

print('Total train data: {}'.format(len(os.listdir(db_path))))

for i, name in enumerate(os.listdir(db_path)):
    print(i, name)
    indemb = []

    for j, indname in enumerate(glob.glob(db_path+name+"/*.jpg")):

        # face detection + alignment
        try:
            extracted_faces = extract_faces(indname,
                          target_size=(224, 224), detector_backend=detmodel,
                          grayscale=False,  enforce_detection=True, align=deepalign)
        except ValueError:
            print(name)
            extracted_faces = []
            noface.append(name)
            # for model in ["dlib", "ssd","mtcnn","opencv", 'mediapipe']:
            #     print(model)
            #     extracted_faces = extract_faces(indname,
            #                                     target_size=(224, 224), detector_backend=model,
            #                                     grayscale=False, enforce_detection=True, align=True)

        if len(extracted_faces)>0:
            # faces detected, align face

            if len(extracted_faces)>1:
                # for n, face in enumerate(extracted_faces):
                #     plt.figure()
                #     plt.imshow(face[0][0])
                #     plt.title('Face {}'.format(n))
                # plt.pause(5)
                # facesave = int(input('Which face to save?'))
                # plt.close('all')

                if i == 107:
                    facesave = 1
                else:
                    facesave = 0
                print(len(extracted_faces), "faces in image", facesave, "chosen")
            else:
                facesave = 0

            #for k in range(len(extracted_faces)):
            facemat, facecoord, facescore = extracted_faces[facesave]

            if align:
                facemat, angle = align_face_dlib_RGB_crop(facemat)



            embedding_region1 = DeepFace.represent(facemat,
                               model_name=embmodel1, enforce_detection=False,
                               detector_backend="skip", align=False, normalization="Facenet")

            embedding_region2 = DeepFace.represent(facemat,
                               model_name=embmodel2, enforce_detection=False,
                               detector_backend="skip", align=False, normalization="base")

            embedding_region3 = DeepFace.represent(facemat,
                               model_name=embmodel3, enforce_detection=False,
                               detector_backend="skip", align=False, normalization="ArcFace")

            # embedding_region4 = DeepFace.represent(facemat,
            #                    model_name=embmodel4, enforce_detection=False,
            #                    detector_backend="skip", align=False, normalization="base")

            embedding_region5 = DeepFace.represent(facemat,
                               model_name=embmodel5, enforce_detection=False,
                               detector_backend="skip", align=False, normalization="base")

            embedding1 = embedding_region1[0]["embedding"]

            embedding2 = embedding_region2[0]["embedding"]

            embedding3 = embedding_region3[0]["embedding"]

            #embedding4 = embedding_region4[0]["embedding"]

            embedding5 = embedding_region5[0]["embedding"]

            embedding = np.array(np.concatenate([embedding1, embedding2,embedding3,embedding5]))

            fineemb = finetune_model(embedding[None,:])

            assert len(embedding) == 512+128+512+2622 # 2622 vgg-face
            assert np.sum(facemat)>1

            indemb.append(name)
            indemb.append(fineemb)

    # store face embeddings
    if len(indemb)==2:
        embeddings.append(indemb)

print('Total train data: {}'.format(len(os.listdir(db_path))))
print('Total saved faces: {}'.format(len(embeddings)))
print('Check names: {}'.format(noface))
saveload('save',model_embeddings,[embeddings])

print(len(embedding1),len(embedding2),len(embedding3),len(embedding5))
