import pandas as pd
import os
import shutil

data_dir = os.getcwd() + "/data/all_images/"

dest_dir = os.getcwd() + "/data/reorganized/"

skin_df = pd.read_csv("data/HAM10000_metadata.csv")
print(skin_df['dx'].value_counts())

labels = skin_df["dx"].unique().tolist()
label_images = []

count = 0

for label in labels:
    print(dest_dir + str(label) + "/")
    os.makedirs(dest_dir + str(label) + "/")
    sample = skin_df[skin_df["dx"] == label]['image_id']
    label_images.extend(sample)
    # if count > 10:
    #     break
    for id in label_images:
        count += 1
        # if count > 10:
        #     break
        src_f = (data_dir + id + ".jpg")
        dest_f = (dest_dir + label + '/' + id + ".jpg")
        print(count, src_f, dest_f)
        shutil.copyfile(src_f, dest_f)
    label_images = []