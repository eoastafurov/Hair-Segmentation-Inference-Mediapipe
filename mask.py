import mediapipe as mp
import cv2
import numpy as np



mp_hs = mp.solutions.hair_segmentation

image = cv2.imread('test.png')

results = 1
with mp_hs.HairSegmentation() as HS:
    results = HS.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
hair_mask = results.hair_mask
hair_mask = cv2.cvtColor(hair_mask, cv2.COLOR_BGR2RGB)
# print(hair_mask)

hair_mask = cv2.resize(hair_mask, (image.shape[1], image.shape[0]))
cv2.imwrite('out.jpg', hair_mask)

for i in range(hair_mask.shape[0]):
    for j in range(hair_mask.shape[1]):
        eps = 0
        eps += hair_mask[i,  j, 0]
        eps += hair_mask[i,  j, 1]
        eps += hair_mask[i,  j, 2]
        if eps > 100:
            image[i, j] = [0, 255, 255]

values = []
for i in range(hair_mask.shape[0]):
    for j in range(hair_mask.shape[1]):
        if hair_mask[i,  j, 0] > 0 or hair_mask[i,  j, 1] > 0 or hair_mask[i,  j, 2] > 0:
            values.append([hair_mask[i,  j, 0], hair_mask[i,  j, 1], hair_mask[i,  j, 2]])
            # print('r: {}, g: {}, b: {}'.format(hair_mask[i,  j, 0], hair_mask[i,  j, 1], hair_mask[i,  j, 2]))
values.sort()
print(values)
cv2.imwrite('applied_mask.jpg', image)


