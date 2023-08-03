import cv2
from abc import ABC, abstractmethod
from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text
from easyocr import Reader
import warnings

warnings.filterwarnings('ignore')

class BaseOcr(ABC):
    def __init__(self):
        self.img_path = None
        self.ocr_result = {}

    def get_ocr_result(self):
        return self.ocr_result

    def get_img_path(self):
        return self.img_path

    def show_img(self):
        plt_imshow(img=self.img_path)

    def show_img_with_ocr(self, bounding, description, vertices, point):
        img = cv2.imread(self.img_path)
        roi_img = img.copy()
        color = (0, 255, 0)

        x, y = point
        ocr_result =  self.ocr_result if bounding is None \
            else self.ocr_result[bounding]
        for text_result in ocr_result:
            text = text_result[description]
            rect = text_result[vertices]

            topLeft, topRight, bottomRight, bottomLeft = [
                (round(point[x]), round(point[y])) for point in rect
            ]

            cv2.line(roi_img, topLeft, topRight, color, 2)
            cv2.line(roi_img, topRight, bottomRight, color, 2)
            cv2.line(roi_img, bottomRight, bottomLeft, color, 2)
            cv2.line(roi_img, bottomLeft, topLeft, color, 2)
            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 20, font_size=15)

        plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))

    @abstractmethod
    def run_ocr(self, img_path: str, debug: bool = False):
        pass

class PororoOcr(BaseOcr):
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        super().__init__()
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)

        if self.ocr_result['description']:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = "No text detected."

        if debug:
            self.show_img_with_ocr("bounding_poly", "description", "vertices", ["x", "y"])

        return ocr_text

    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

# https://www.jaided.ai/easyocr/documentation/
class EasyOcr(BaseOcr):
    def __init__(self, lang: list[str] = ["ko", "en"], gpu=False, **kwargs):
        super().__init__()
        self._ocr = Reader(lang_list=lang, gpu=gpu, **kwargs).readtext

    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=1)

        if len(self.ocr_result) != 0:
            ocr_text = list(map(lambda result: result[1], self.ocr_result))
        else:
            ocr_text = "No text detected."

        if debug:
            self.show_img_with_ocr(None, 1, 0, [0, 1])

        return ocr_text

if __name__ == "__main__":
    p_ocr = PororoOcr()
    e_ocr = EasyOcr()
    image_path = input("Enter image path: ")
    text = p_ocr.run_ocr(image_path, debug=True)
    print('Result :', text)
    text = e_ocr.run_ocr(image_path, debug=True)
    print('Result :', text)
