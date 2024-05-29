
import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv('./dataset.csv')

# drop label column
data = df.drop('label', axis=1).values

print(data.shape)

# 看前五筆資料圖片
for i in range(5):
    img = data[i].reshape(28, 28)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.show()
    