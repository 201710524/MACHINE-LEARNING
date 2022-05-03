import cv2
import numpy as np
import dlib
import math

frame = cv2.imread("./image/img.jpg", cv2.IMREAD_COLOR)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/landmark/shape_predictor_68_face_landmarks.dat")

C = 5
C2 = 6
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

face = detector(gray)

for i in face:
    landmarks = predictor(gray, i)
    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                 (landmarks.part(43).x, landmarks.part(43).y),
                                 (landmarks.part(44).x, landmarks.part(44).y),
                                 (landmarks.part(45).x, landmarks.part(45).y),
                                 (landmarks.part(46).x, landmarks.part(46).y),
                                 (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
    cut = frame[landmarks.part(37).y+2: landmarks.part(40).y-2, landmarks.part(37).x+2: landmarks.part(40).x-2]
    cut_2 = frame[landmarks.part(43).y: landmarks.part(46).y, landmarks.part(43).x+7: landmarks.part(46).x-7]


"""
image를 r, g, b로 각각 나눔.
"""
img_b, img_g, img_r = cv2.split(cut)
img_b_one2, img_g_one2, img_r_one2 = cv2.split(cut_2)
#
img_r_one = img_r
img_g_one = img_g
img_b_one = img_b
#

#
img_r_one = img_r_one.astype(np.float_)
img_g_one = img_g_one.astype(np.float_)
img_b_one = img_b_one.astype(np.float_)
#
img_r_one2 = img_r_one2.astype(np.float_)
img_g_one2 = img_g_one2.astype(np.float_)
img_b_one2 = img_b_one2.astype(np.float_)
# """
# threshold
# """
h, w, _ = cut.shape
#
# """
# 반사광 제거
# """
rg1 = cv2.GaussianBlur(img_r_one, (5,5), 0.4)
rg2 = cv2.GaussianBlur(img_r_one, (5,5), 0.6)
bg1 = cv2.GaussianBlur(img_b_one, (5,5), 0.4)
bg2 = cv2.GaussianBlur(img_b_one, (5,5), 0.3)
bg3 = cv2.GaussianBlur(img_b_one, (5,5), 0.3)
gg1 = cv2.GaussianBlur(img_g_one, (5,5), 0.35)
gg2 = cv2.GaussianBlur(img_g_one, (5,5), 0.3)
gg3 = cv2.GaussianBlur(img_g_one, (5,5), 0.25)
gg4 = cv2.GaussianBlur(img_g_one, (5,5), 0.1)
#
rg12 = cv2.GaussianBlur(img_r_one2, (5,5), 0.4)
rg22 = cv2.GaussianBlur(img_r_one2, (5,5), 0.6)
bg12 = cv2.GaussianBlur(img_b_one2, (5,5), 0.4)
bg22 = cv2.GaussianBlur(img_b_one2, (5,5), 0.3)
bg32 = cv2.GaussianBlur(img_b_one2, (5,5), 0.3)
gg12 = cv2.GaussianBlur(img_g_one2, (5,5), 0.35)
gg22 = cv2.GaussianBlur(img_g_one2, (5,5), 0.3)
gg32 = cv2.GaussianBlur(img_g_one2, (5,5), 0.25)
gg42 = cv2.GaussianBlur(img_g_one2, (5,5), 0.1)

red_msr = (np.log10(rg1) - np.log10(rg1 * img_r_one) + np.log10(rg2) - np.log10(rg2 * img_r_one))
blue_msr = (np.log10(bg1) - np.log10(bg1 * img_b_one) + np.log10(bg2) - np.log10(bg2 * img_b_one) + np.log10(bg3) - np.log10(bg3 * img_b_one))
green_msr = (np.log10(gg1) - np.log10(gg1 * img_g_one) + np.log10(gg2) - np.log10(gg2 * img_g_one)
             + np.log10(gg3) - np.log10(gg3 * img_g_one) + np.log10(gg4) - np.log10(gg4 * img_g_one))


red_msr2 = (np.log10(rg12) - np.log10(rg12 * img_r_one2) + np.log10(rg22) - np.log10(rg22 * img_r_one2))
# print(red_msr2)
blue_msr2 = (np.log10(bg12) - np.log10(bg12 * img_b_one2) + np.log10(bg22) - np.log10(bg22 * img_b_one2) +
             np.log10(bg32) - np.log10(bg32 * img_b_one2))

green_msr2 = (np.log10(gg12) - np.log10(gg12 * img_g_one2) + np.log10(gg22) - np.log10(gg22 * img_g_one2)
             + np.log10(gg32) - np.log10(gg32 * img_g_one2) + np.log10(gg42) - np.log10(gg42 * img_g_one2))
#
"""
define SQR(x) = (x)*(x)
s = sum(red_msr)
SSQR = sum(SQR)
mean(r) = s / image_size
std(r) = sqrt((SSQR- s*s/image_size)/image_size);
max_val = mean +1.2 * std
min_val = mean - 1.2 * std
range = max_val - min_val
if(!range):range = 1.0;
range = 255./range
for i in image_size:
red_msr = (red_msr - min_val) * range
"""

"""
1
"""
mean_red = np.mean(red_msr, dtype=np.float64)
std_red = np.std(red_msr, dtype=np.float64)

max_red = mean_red + (1.2 * std_red)
min_red = mean_red - (1.2 * std_red)
range_red =  max_red - min_red
if(not range_red):
    range_red = 1
range_red = 255/range_red
red_msr = (red_msr - min_red) * range_red
"""
2
"""
mean_red2 = np.mean(red_msr2, dtype=np.float64)
std_red2 = np.std(red_msr2, dtype=np.float64)

# print(mean_red2)
max_red2 = mean_red2 + (1.2 * std_red2)
min_red2 = mean_red2 - (1.2 * std_red2)
range_red2 =  max_red2 - min_red2
if(not range_red2):
    range_red2 = 1
range_red2 = 255/range_red2

red_msr2 = (red_msr2 - min_red2) * range_red2

#
# #blue
mean_blue = np.mean(blue_msr)
std_blue = np.std(blue_msr)

max_blue = mean_blue + (1.2 * std_blue)
min_blue = mean_blue - (1.2 * std_blue)
range_blue =  max_blue - min_blue
if(not range_blue):
    range_blue = 1
range_blue = 255/range_blue
blue_msr = (blue_msr - min_blue) * range_blue
#
#
"""
blue2
"""
mean_blue2 = np.mean(blue_msr2)
std_blue2 = np.std(blue_msr2)

max_blue2 = mean_blue2 + (1.2 * std_blue2)
min_blue2 = mean_blue2 - (1.2 * std_blue2)
range_blue2 =  max_blue2 - min_blue2
if(not range_blue2):
    range_blue2 = 1
range_blue2 = 255/range_blue2
blue_msr2 = (blue_msr2 - min_blue2) * range_blue2

"""
green1
"""
mean_green = np.mean(green_msr)
std_green = np.std(green_msr)

max_green = mean_green + (1.2 * std_green)
min_green = mean_green - (1.2 * std_green)
range_green =  max_green - min_green
if(not range_green):
    range_green = 1
range_green = 255/range_green
green_msr = (green_msr - min_green) * range_green
"""
green2
"""
mean_green2 = np.mean(green_msr2)
std_green2 = np.std(green_msr2)

max_green2 = mean_green2 + (1.2 * std_green2)
min_green2 = mean_green2 - (1.2 * std_green2)
range_green2 =  max_green2 - min_green2
if(not range_green2):
    range_green2 = 1
range_green2 = 255/range_green2
green_msr2 = (green_msr2 - min_green2) * range_green2




red_img = np.log10(C * img_r_one/(img_r_one + img_g_one + img_b_one)) * red_msr
blue_img = np.log10(C * img_b_one/(img_r_one + img_g_one + img_b_one)) * blue_msr
green_img = np.log10(C * img_g_one/(img_r_one + img_g_one + img_b_one)) * green_msr

red_img2 = np.log10(C2 * img_r_one2/(img_r_one2 + img_g_one2 + img_b_one2)) * red_msr2

blue_img2 = np.log10(C2 * img_b_one2/(img_r_one2 + img_g_one2 + img_b_one2)) * blue_msr2
green_img2 = np.log10(C2 * img_g_one2/(img_r_one2 + img_g_one2 + img_b_one2)) * green_msr2



red_img = red_img.astype(np.uint8)
blue_img = blue_img.astype(np.uint8)
green_img = green_img.astype(np.uint8)
#
red_img2 = red_img2.astype(np.uint8)
blue_img2 = blue_img2.astype(np.uint8)
green_img2 = green_img2.astype(np.uint8)

result = cv2.merge((blue_img, green_img, red_img))
result2 = cv2.merge((blue_img2, green_img2, red_img2))
#
#
frame[landmarks.part(37).y + 2: landmarks.part(40).y - 2, landmarks.part(37).x + 2: landmarks.part(40).x - 2] = result
frame[landmarks.part(43).y: landmarks.part(46).y, landmarks.part(43).x+7: landmarks.part(46).x-7] = result2

cv2.imwrite('./image/test_result52.jpg', red_img2)
cv2.imwrite('./image/test_result53.jpg', frame)
cv2.imshow("frame", frame)
cv2.imshow("cut_o", result)
cv2.imshow("cut1", result2)
cv2.imshow("cut2", blue_img2)
cv2.imshow("cut3", green_img2)
cv2.imshow("cut4", red_img2)
cv2.imshow("cut5", cut_2)



key = cv2.waitKey(0)
cv2.destroyAllWindows()