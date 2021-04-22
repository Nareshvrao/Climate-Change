# @Author: Naresh Venkataramanan<Nareshvrao>
# @Date:   2019-12-22, 12:44:08
# @Last modified by:   Naresh
# @Last modified time: 2019-12-22, 1:13:26



import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp

SIZE = 384

df_train = pd.read_csv('understanding_cloud_organization/train.csv')
df_test = pd.read_csv('understanding_cloud_organization/sample_submission.csv')
DATA_ROOT = 'understanding_cloud_organization/train_images/images/'
OUTPUT_DIR = 'understanding_cloud_organization/train_images/images_%d/'%SIZE

image_names = df_train['Image_Label'].apply(lambda x: x.split('_')[0]).unique().tolist()
image_names += df_test['Image_Label'].apply(lambda x: x.split('_')[0]).unique().tolist()

def preprocess_image(image_names, run_root=DATA_ROOT, out_root=OUTPUT_DIR, size=SIZE):
    for i in tqdm(range(len(image_names))):
        image_name = image_names[i]
        path = run_root+image_name
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, C = img.shape
        new_H = int(SIZE)
        new_W = int(W/H*SIZE)
        img = cv2.resize(img, (new_W, new_H))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(OUTPUT_DIR + image_name, img)

pool = mp.Pool(12)
n_cnt = len(image_names) // 12

dfs = [image_names[n_cnt*i:n_cnt*(i+1)] for i in range(12)]
dfs[-1] = image_names[n_cnt*11:]
res = pool.map(preprocess_image, [x_df for x_df in dfs])
pool.close()
