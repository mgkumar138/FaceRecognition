import os
import glob
import cv2

db_path = "./db_set/"
test_dir = "./test_imgs_set/"

db_imgs = os.listdir(db_path)
test_imgs = os.listdir(test_dir)

print(len(db_imgs)," in DB")
print(len(test_imgs)," in Test")

# corrupted_db = []
# # check train files
# for i, db_img in enumerate(db_imgs):
#
#     try:
#         db_file_dir = db_path+db_img+'/'+ os.listdir(db_path+db_img)[0]
#         img = cv2.imread(db_file_dir)
#         print(img.shape)
#     except ValueError:
#         corrupted_db.append(db_img)
#         print("Corrupted",db_img)


# check test files
missing_db = []
for i, test_img in enumerate(test_imgs):

    test_name = test_img.split('.')[0]
    if test_name in db_imgs:
        #print(test_img, "valid")
        pass
    else:
        missing_db.append(test_img)
        print(test_img, "missing")

        file_dir = test_dir+test_img
        os.rename(test_dir+test_img,  test_dir+ "NonDB/" +test_img)  # move image

print(len(os.listdir(test_dir))," in modified Test")