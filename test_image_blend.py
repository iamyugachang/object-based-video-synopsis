import operator
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np

# # suppose img1 and img2 are your two images
# img1 = Image.new('RGB', size=(100, 100), color=(255, 0, 0))
# img2 = Image.new('RGB', size=(120, 130), color=(0, 255, 0))

# # suppose img2 is to be shifted by `shift` amount 
# shift = (50, 60)

# # compute the size of the panorama
# nw, nh = map(max, map(operator.add, img2.size, shift), img1.size)

# # paste img1 on top of img2
# newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
# newimg1.paste(img2, shift)
# newimg1.paste(img1, (0, 0))
# newimg1.save("./outputs/blend1.png")
# # cv2.imwrite("./outputs/blend1.png",newimg1)

# # paste img2 on top of img1
# newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
# newimg2.paste(img1, (0, 0))
# newimg2.paste(img2, shift)
# newimg2.save("./outputs/blend2.png")
# # cv2.imwrite("./outputs/blend2.png",newimg2)

# # blend with alpha=0.5
# result = Image.blend(newimg1, newimg2, alpha=0.5)
# result.save("./outputs/blend3.png")
# # cv2.imwrite("./outputs/blend3.png",result)

img1 = cv2.imread("./outputs/tmp_2/18.png")
_, img1_mask = cv2.threshold(img1, 0.1, 255, cv2.THRESH_BINARY)
img2 = cv2.imread("./outputs/tmp_2/315.png")
_, img2_mask = cv2.threshold(img2, 0.1, 255, cv2.THRESH_BINARY)



_, overlap_mask = cv2.threshold(cv2.bitwise_and(img1_mask, img2_mask), 10, 255, cv2.THRESH_BINARY)
overlap_mask_inv = cv2.bitwise_not(overlap_mask)
blended_part = cv2.addWeighted(img1&overlap_mask, 0.5, img2&overlap_mask, 0.5, 0)
print(np.count_nonzero((overlap_mask == [255,255,255]).all(axis = 2)))
print(np.count_nonzero((img1_mask == [255,255,255]).all(axis = 2)))
result = ((img1 + img2) & overlap_mask_inv) + blended_part
cv2.imwrite("./outputs/tmp_2/result.png", result)
