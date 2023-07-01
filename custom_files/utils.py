import cv2
import numpy as np
from PIL import Image
import dlib
#import imutils
import matplotlib.pyplot as plt

def align_face_dlib_RGB_crop(face_img, xywh=(0,0,224,224), crop=False):
    face = cv2.cvtColor(face_img[0], cv2.COLOR_BGR2RGB)
    face = (face * 255).astype(np.uint8)
    predictor_model = "../custom_files/shape_predictor_68_face_landmarks.dat"
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    landmarks = face_pose_predictor(face, dlib.rectangle(xywh[0], xywh[1], xywh[2], xywh[3]))

    plot_landmark = False
    nose = get_nose(landmarks, face,plot_landmark=plot_landmark)
    l_eye = get_l_eye(landmarks, face,plot_landmark=plot_landmark)
    r_eye = get_r_eye(landmarks, face,plot_landmark=plot_landmark)

    center_of_forehead = np.array([(l_eye[0] + r_eye[0]) // 2, (l_eye[1] + r_eye[1]) // 2])
    #cv2.circle(face, center=(center_of_forehead[0], center_of_forehead[1]), radius=5, color=(0, 255, 0), thickness=1)

    center_pred = np.array([int((xywh[0] + xywh[2]) / 2), int((xywh[1] + xywh[1]) / 2)])  # 1 or 3?
    #cv2.circle(face, center=(center_pred[0], center_pred[1]), radius=5, color=(0, 255, 0), thickness=1)

    #plt.imshow(face)

    length_line1 = np.linalg.norm(center_of_forehead - nose)  # np.linalg.norm(ord=2)?
    length_line2 = np.linalg.norm(center_pred - nose)
    length_line3 = np.linalg.norm(center_pred - center_of_forehead)

    # length_line1 = distance(center_of_forehead, nose)  # np.linalg.norm(ord=2)?
    # length_line2 = distance(center_pred, nose)
    # length_line3 = distance(center_pred, center_of_forehead)

    cos_a = cosine_formula(length_line1, length_line2, length_line3)
    angle = np.arccos(cos_a)

    rotated_point = rotate_point(nose, center_of_forehead, angle)
    rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
    if is_between(nose, center_of_forehead, center_pred, rotated_point):
        angle = np.degrees(-angle)
    else:
        angle = np.degrees(angle)

    img = Image.fromarray(face)
    img = np.array(img.rotate(angle))

    if crop:
        align_landmarks = face_pose_predictor(img, dlib.rectangle(xywh[0], xywh[1], xywh[2], xywh[3]))
        coords = np.zeros([68,2],dtype=int)
        for i in range(68):
            coords[i] = np.array([align_landmarks.part(i).x, align_landmarks.part(i).y])

        plt.figure()
        plt.imshow(img)
        for i in range(68):
            plt.scatter(coords[i,0],coords[i,1],s=20)

        minx = np.min(coords[:,0])
        maxx = np.max(coords[:, 0])
        miny = np.min(coords[:,1])
        maxy = np.max(coords[:, 1])

        crop_img = img[minx:maxx, miny:maxy]

        plt.figure()
        plt.imshow(crop_img)
        for i in range(68):
            plt.scatter(coords[i,0]-minx,coords[i,1]-miny,s=20, c='r')

    return (img/255.0)[None,:], angle  # need to transform face to BGR? depends on model


def get_nose(landmarks, gray,plot_landmark=False):
    ## NOSE
    nose = landmarks.part(30)
    nose_x, nose_y = nose.x, nose.y
    nose = np.array([nose_x, nose_y])
    if plot_landmark:
        cv2.circle(gray, center=(nose_x, nose_y), radius=5, color=(0, 255, 0), thickness=1)
    return nose


def get_l_eye(landmarks, gray,plot_landmark=False):
    ## LEYE
    left_eye_x = sum([landmarks.part(n).x for n in range(36, 42)]) // 6
    left_eye_y = sum([landmarks.part(n).y for n in range(36, 42)]) // 6
    left_eye = np.array([left_eye_x, left_eye_y])
    if plot_landmark:
        cv2.circle(gray, center=(left_eye_x, left_eye_y), radius=5, color=(255, 255, 0), thickness=1)
    return left_eye


def get_r_eye(landmarks, gray,plot_landmark=False):
    # R Eye
    # Extract the right eye coordinates from the landmarks
    right_eye_x = sum([landmarks.part(n).x for n in range(42, 48)]) // 6
    right_eye_y = sum([landmarks.part(n).y for n in range(42, 48)]) // 6
    right_eye = np.array([right_eye_x, right_eye_y])
    if plot_landmark:
        cv2.circle(gray, center=(right_eye_x, right_eye_y), radius=5, color=(0, 0, 255), thickness=1)
    return right_eye



def saveload(opt, name, variblelist):
    import pickle
    name = name + '.pickle'
    if opt == 'save':
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var


def face_detector(img, align, model):
    # https://learnopencv.com/what-is-face-detection-the-ultimate-guide/
    resp = []
    if model == "mtcnn":
        from mtcnn import MTCNN
        face_detector = MTCNN()

        detected_face = None
        img_region = [0, 0, img.shape[1], img.shape[0]]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
        detections = face_detector.detect_faces(img_rgb)

        if len(detections) > 0:

            for detection in detections:
                x, y, w, h = detection["box"]
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                img_region = [x, y, w, h]
                confidence = detection["confidence"]

                if align:
                    keypoints = detection["keypoints"]
                    left_eye = keypoints["left_eye"]
                    right_eye = keypoints["right_eye"]
                    detected_face = alignment_procedure(detected_face, keypoints)

                resp.append((detected_face, img_region, confidence))

    elif model == 'retinaface':
        from retinaface import RetinaFace  # this is not a must dependency
        from retinaface.commons import postprocess
        obj = RetinaFace.detect_faces(img, model=model, threshold=0.9)  # Resnet50 or Mobilenetv1 (faster)?

        if isinstance(obj, dict):
            for face_idx in obj.keys():
                identity = obj[face_idx]
                facial_area = identity["facial_area"]

                y = facial_area[1]
                h = facial_area[3] - y
                x = facial_area[0]
                w = facial_area[2] - x
                img_region = [x, y, w, h]
                confidence = identity["score"]

                # detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
                detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

                if align:
                    landmarks = identity["landmarks"]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    nose = landmarks["nose"]
                    # mouth_right = landmarks["mouth_right"]
                    # mouth_left = landmarks["mouth_left"]

                    detected_face = postprocess.alignment_procedure(
                        detected_face, right_eye, left_eye, nose
                    )

                resp.append((detected_face, img_region, confidence))

    elif model == 'mediapipe':
        pass
    elif model == 'yunet':
        detector = cv2.FaceDetectorYN.create(model="face_detection_yunet_2022mar.onnx", config="", input_size=(320, 320))
        detections = detector.detect(img)


    return resp




def alignment_procedure(img, keypoints):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    nose = keypoints["nose"]
    mouth_left = keypoints["mouth_left"]
    mouth_right = keypoints["mouth_right"]

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway




def align_face(imgname, faces):
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(imgname)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    predictor = dlib.shape_predictor('../custom_files/shape_predictor_5_face_landmarks.dat')

    #detect eyes & nose
    rotimg = []
    if len(faces) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            shape = predictor(gray,rect)

            shape = shape_to_normal(shape)
            nose, left_eye, right_eye = get_eyes_nose_dlib(shape)

            # get center of forehead
            center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            # center of face bbox
            center_pred = (int((x + w) / 2), int((y + y) / 2))

            # center of all
            length_line1 = distance(center_of_forehead, nose)
            length_line2 = distance(center_pred, nose)
            length_line3 = distance(center_pred, center_of_forehead)

            # cosine angle
            cos_a = cosine_formula(length_line1, length_line2, length_line3)
            angle = np.arccos(cos_a)

            rotated_point = rotate_point(nose, center_of_forehead, angle)
            rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
            if is_between(nose, center_of_forehead, center_pred, rotated_point):
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)

            # rotate img
            img = Image.fromarray(image)
            img = np.array(img.rotate(angle))
            rotimg.append(img)
    return rotimg


def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal

def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False