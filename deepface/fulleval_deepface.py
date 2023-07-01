from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
from deepface.commons.functions import extract_faces
import os
import glob
from custom_files.utils import saveload, align_face_dlib_RGB_crop
import pandas as pd
import time


db_path = "../FaceData/db_set/"
imgs_dir = "../FaceData/test_imgs_set/"
cos_score = []
acc = []
thres_acc = []
skipped_img = []
latency = []

files = os.listdir(imgs_dir)
#files = np.random.choice(files,1500)
detmodel = "retinaface"
embmodel1 = "Facenet512"  #512
embmodel2 = "SFace"  # 128
embmodel3 = "ArcFace"  # 512
#embmodel4 = "Dlib"  # 512
embmodel5 = "VGG-Face"  #2622
align = False
deepalign = not align

model_embeddings = '{}_align{}_{}_{}_{}_{}_embeddings'.format(detmodel,align, embmodel1,embmodel2, embmodel3, embmodel5)
print(model_embeddings)

# load database face embeddings
[saved_embeddings] = saveload('load',model_embeddings,1)
df = pd.DataFrame(saved_embeddings, columns=["identity", "representation"])
df["representation"] = df["representation"].apply(lambda x: np.array(x, dtype=np.float32))

emd_mat = np.vstack(df["representation"].to_numpy())
norm_emb_mat = np.linalg.norm(emd_mat, axis=1)

emd_mat_1 = emd_mat[:,:512]
emd_mat_2 = emd_mat[:,512:512+128]
emd_mat_3 = emd_mat[:,512+128:512+128+512]
#emd_mat_4 = emd_mat[:,512+128+512:512+128+512+128]
emd_mat_5 = emd_mat[:,512+128+512:]

norm_emb_mat_1 = np.linalg.norm(emd_mat[:,:512], axis=1)
norm_emb_mat_2 = np.linalg.norm(emd_mat[:,512:512+128], axis=1)
norm_emb_mat_3 = np.linalg.norm(emd_mat[:,512+128:512+128+512], axis=1)
#norm_emb_mat_4 = np.linalg.norm(emd_mat[:,512+128+512:512+128+512+128], axis=1)
norm_emb_mat_5 = np.linalg.norm(emd_mat[:,512+128+512:], axis=1)

emb_names = df["identity"].to_numpy()
match_threshold = 0.7

img_embeddings = []
wrong_faces = []
for i, f in enumerate(files):

    print('Image {}/{} -------------------'.format(i+1, len(files)))

    img_path = imgs_dir+f
    true_name = (img_path.split('/')[-1]).split('.')[0]
    print(true_name)

    t0 = time.time()
    # face detection + alignment
    try:
        extracted_faces = extract_faces(img_path,
                      target_size=(224, 224), detector_backend=detmodel,
                      grayscale=False,  enforce_detection=True, align=deepalign)
    except ValueError:
        extracted_faces = []

    # if detface, get embedding
    if len(extracted_faces) > 0:
        top_match = 0
        top_name = None
        top_face = None
        for k in range(len(extracted_faces)):
            facemat, facecoord, facescore = extracted_faces[k]

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
            #                                        model_name=embmodel4, enforce_detection=False,
            #                                        detector_backend="skip", align=False, normalization="base")

            embedding_region5 = DeepFace.represent(facemat,
                                                   model_name=embmodel5, enforce_detection=False,
                                                   detector_backend="skip", align=False, normalization="base")

            embedding1 = embedding_region1[0]["embedding"]

            embedding2 = embedding_region2[0]["embedding"]

            embedding3 = embedding_region3[0]["embedding"]

            #embedding4 = embedding_region4[0]["embedding"]

            embedding5 = embedding_region5[0]["embedding"]

            embedding = np.array(np.concatenate([embedding1, embedding2, embedding3, embedding5]))

            assert len(embedding) == 512 + 128 + 512 + 2622  # 2622 vgg-face

            # get similarity score
            norm_face_1 = np.linalg.norm(embedding1)
            fcor1 = np.matmul(emd_mat_1, embedding1) / (norm_emb_mat_1*norm_face_1)

            norm_face_2 = np.linalg.norm(embedding2)
            fcor2 = np.matmul(emd_mat_2, embedding2) / (norm_emb_mat_2*norm_face_2)

            norm_face_3 = np.linalg.norm(embedding3)
            fcor3 = np.matmul(emd_mat_3, embedding3) / (norm_emb_mat_3*norm_face_3)

            # norm_face_4 = np.linalg.norm(embedding4)
            # fcor4 = np.matmul(emd_mat_4, embedding4) / (norm_emb_mat_4*norm_face_4)

            norm_face_5 = np.linalg.norm(embedding5)
            fcor5 = np.matmul(emd_mat_5, embedding5) / (norm_emb_mat_5*norm_face_5)

            fcor = (fcor1+fcor2+fcor3+fcor5)/4

            # norm_face = np.linalg.norm(embedding)
            # fcor = np.matmul(emd_mat, embedding) / (norm_emb_mat*norm_face)

            top_cos = np.argmax(fcor)
            #cos_thres = fcor > match_threshold

            cossim = fcor[top_cos]
            name = emb_names[top_cos]
            print(name, cossim)
            img_embeddings.append([true_name, embedding, facemat])

            if cossim > top_match:
                top_match = cossim
                top_name = name
                top_face = facemat

        t1 = time.time()
        total = t1-t0
        print('Latency: ', np.round(total,3))
        latency.append(total)

        checkpred = top_name == true_name
        cos_score.append(top_match)
        if checkpred:
            acc.append(1)
            print('$$$ Correct prediction')
        else:

            try:
                true_file_path = glob.glob(db_path+true_name+"/"+true_name+"*")[0]
                true_face = extract_faces(true_file_path,
                                             target_size=(224, 224), detector_backend=detmodel,
                                                grayscale=False, enforce_detection=True, align=True)
                true_face_mat, _, _ = extracted_faces[0]

                # predicted face
                pred_file_path = glob.glob(db_path + top_name + "/" + top_name + "*")[0]
                pred_face = extract_faces(pred_file_path,
                                          target_size=(224, 224), detector_backend=detmodel,
                                          grayscale=False, enforce_detection=True, align=True)
                pred_face_mat, _, _ = extracted_faces[0]

                wrong_faces.append([i, true_name,top_name, top_face, pred_face_mat,true_face_mat])
                print("True name in DB but wrongly matched")
                acc.append(0)
            except IndexError:
                wrong_faces.append([i, true_name,top_name, top_face])
                print("DB has no true name and face")


        if checkpred and cossim > match_threshold:
            thres_acc.append(1)
        elif cossim > match_threshold and not checkpred:
            thres_acc.append(0)

    else:
        print('No faces detected', img_path)
        skipped_img.append(img_path)

unique_wrong_face = []
for i in range(len(wrong_faces)):
    if len(wrong_faces) >4:
        diagnosis = "In DB but wrong match"
    else:
        diagnosis = "Not in DB"
    print(wrong_faces[i][0], wrong_faces[i][1], wrong_faces[i][2], diagnosis)
    unique_wrong_face.append(wrong_faces[i][1])
unique_wrong_face = list(set(unique_wrong_face))
print("unique wrong faces ", len(unique_wrong_face))

print('Face Recog acc', 100*np.sum(acc)/len(acc), '%', np.sum(acc), len(acc))
print('Thres Recog acc', 100*np.sum(thres_acc)/len(thres_acc), '%', np.sum(thres_acc), len(thres_acc))
print('Skipped ',len(skipped_img))
print('Avg latency ', np.mean(latency))

saveload('save',model_embeddings+'_pred_{}n'.format(len(files)),[img_embeddings])

# plt.figure()
# plt.hist(np.array(cos_score)[np.array(acc)==0])
# plt.hist(np.array(cos_score)[np.array(acc)==1],alpha=0.5)
# plt.show()

# retinaface, FaceNet512, -
# retinaface, [FaceNet512, VGG-Face] -
# retinaface, FaceNet512 + VGG-Face - 36.9 / 39.9 (0.6 thres) / skipped 17 / lat 0.47s
# retinaface, FaceNet512 + SFace + ArcFace - 53.0 (717/1352) / 80.2 @ 0.7 (61/76) / skipped 17 / lat 0.41s
# mtcnn, FaceNet512 + SFace + ArcFace - 50.3 (688/1366) / 72.2 @ 0.7 (70/97) / skipped 3 / lat 0.75s
# mtcnn, FaceNet512 + SFace + ArcFace + Dlib + VGGFace - 51.2 (699/1366) / 70.0 @ 0.7 (245/350) / skipped 3 / lat 1.65s
# retinaface, FaceNet512 + SFace + ArcFace + Dlib + VGGFace - 53.0 (717/1352) / 73.4 @ 0.7 (237/323) / skipped 17 / lat 0.94s
# retinaface, FaceNet512 (norm) + SFace + ArcFace (norm) + Dlib + VGGFace - 54.4 (735/1352) / 72.1 @ 0.7 (215/298) / skipped 17 / lat 0.84s

# retinaface, FaceNet512 (norm) + SFace + ArcFace (norm) + Dlib + VGGFace - 54.4 (735/1352) / 72.1 @ 0.7 (215/298) / skipped 17 / lat 0.84s

# 5492: retinaface, FaceNet512 (norm) + SFace + ArcFace (norm) + Dlib + VGGFace - 41.73 (2262/5420) / 58.61 @ 0.7 (531/906) / skipped 72 / lat 1.095s


