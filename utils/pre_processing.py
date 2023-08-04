import cv2
import numpy as np

# https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
# https://nanonets.com/blog/ocr-with-tesseract/
# https://github.com/TCAT-capstone/ocr-preprocessor/blob/main/main.py

def load(path):
    return cv2.imread(path)

def load_with_filter(path):
    image = load(path)
    image = grayscale(image)
    image = thresholding(image, mode="GAUSSIAN")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def isEven(num):
    return num % 2 == 0

# == Color =====================================================================
def grayscale(image, blur=False):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
def thresholding(image, mode="GENERAL",block_size=9, C=5):
    if isEven(block_size):
        print("block_size to use odd")
        return;

    if mode == "MEAN":
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    elif mode == "GAUSSIAN":
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    else:
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# == Noise =====================================================================
def remove_noise(image, kernel_size=5):
    return cv2.medianBlur(image, ksize=kernel_size)

def blur(image):
    return cv2.GaussianBlur(gray, (3,3), 0)

# == Morphology ================================================================
def dilation(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erosion(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def opening(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image):
    kernel = np.ones((5,5),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

# == Others ====================================================================
# def deskew(image):
#     coords = np.column_stack(np.where(image > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#      if angle < -45:
#          angle = -(90 + angle)
#     else:
#         angle = -angle
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# def match_template(image, template):
#     return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# # https://yunwoong.tistory.com/73
# def rotated(image):
#     image_list_title = ['gray', 'blurred', 'edged']
#     image_list = [gray, blurred, edged

#     # contours를 찾아 크기순으로 정렬
#     cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

#     findCnt = None

#     # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
#     for c in cnts:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)

#         # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
#         if len(approx) == 4:
#             findCnt = approx
#             break

#     # 만약 추출한 윤곽이 없을 경우 오류
#     if findCnt is None:
#         raise Exception(("Could not find outline."))

#     output = image.copy()
#     cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

#     image_list_title.append("Outline")
#     image_list.append(output)

#     # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
#     transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

#     plt_imshow(image_list_title, image_list)
#     plt_imshow("Transform", transform_image)

#     return transform_image
