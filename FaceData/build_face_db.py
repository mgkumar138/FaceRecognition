import requests
import csv
import os

down_type = 'test'

if down_type == 'train':
    with open('./csv/training_set_compass_vale.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            print(row)
            if row["userid"] != '' and row["trainURL"] != '':
                name = row["userid"]+'_'
                url = row["trainURL"]

                directory = './db_set/'+name
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    fnum = 1
                else:
                    files = os.listdir(directory)
                    fnum = len(files)+1
                    print('Multiple {}'.format(fnum))
                    #pass

                title_filename = name+str(fnum)
                img_data = requests.get(url).content
                with open(directory+'/'+title_filename+'.jpg', 'wb') as handler:
                    handler.write(img_data)

else:
    with open('./csv/test_set_compass_vale.csv') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            print(row)
            if row["userid"] != '' and row["testurl"] != '':#: and row["caputeddate"] != '':
                name = row["userid"]+'_'
                url = row["testurl"]
                #date = row["caputeddate"]
                idx = url.split('/')[-1]

                datedirectory = './test_imgs_set/'#+date
                if not os.path.exists(datedirectory):
                    os.makedirs(datedirectory)

                title_filename = name+'.'+idx
                img_data = requests.get(url).content
                with open(datedirectory+'/'+title_filename, 'wb') as handler:
                    handler.write(img_data)