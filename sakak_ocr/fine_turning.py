import ast
import os
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from pororo import Pororo
from pororo.models.brainOCR.model import Model
from pororo.models.brainOCR.recognition import get_recognizer
from pororo.pororo import SUPPORTED_TASKS
from pororo.tasks import PororoFactoryBase, PororoOcrFactory, download_or_load


def build_vocab(character: str) -> List[str]:
    """Returns vocabulary (=list of characters)"""
    vocab = ["[blank]"] + list(
        character)  # dummy '[blank]' token for CTCLoss (index 0)
    return vocab

def parse_options(opt_fp: str) -> dict:
    opt2val = dict()
    for line in open(opt_fp, "r", encoding="utf8"):
        line = line.strip()
        if ": " in line:
            opt, val = line.split(": ", 1)
            try:
                opt2val[opt] = ast.literal_eval(val)
            except:
                opt2val[opt] = val
    opt2val["vocab"] = build_vocab(opt2val["character"])
    opt2val["vocab_size"] = len(opt2val["vocab"])
    print(f"opt2val: {opt2val}")
    return opt2val

# 커스텀 OCR 데이터셋 클래스
class OCRDataset(Dataset):
    def __init__(self, real_dir, label_dir, transform=None):
        self.real_dir = real_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(real_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        real_path = os.path.join(self.real_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])

        real_image = Image.open(real_path).convert("RGB")
        label_image = Image.open(label_path).convert("L")  # Label 이미지는 흑백
        #
        # if self.transform:
        #     real_image = self.transform(real_image)
        #     label_image = self.transform(label_image)

        return real_image, label_image


def fine_turning_test():
    # MPS 장치 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # BrainOCR 모델 로드 (map_location을 사용하여 현재 장치로 매핑)
    opt_fp = download_or_load(
        f"misc/ocr-opt.txt",
        "ko",
    )
    det_model_ckpt_fp = download_or_load(
        f"misc/craft.pt",
        "ko",
    )
    rec_model_ckpt_fp = download_or_load(
        f"misc/brainocr.pt",
        "ko",
    )
    opt2val = parse_options(opt_fp)
    opt2val["det_model_ckpt_fp"] = det_model_ckpt_fp
    opt2val["rec_model_ckpt_fp"] = rec_model_ckpt_fp

    brainocr_model = Model(opt2val)
    brainocr_state_dict = torch.load("/System/Volumes/Data/Users/kukjunlee/.pororo/misc/brainocr.pt", map_location=device)
    brainocr_model.load_state_dict(brainocr_state_dict, strict=False)
    brainocr_model.to(device)  # 모델을 MPS 또는 CPU로 이동
    brainocr_model.train()  # 학습 모드로 변경

    # 옵티마이저 및 손실 함수 정의
    optimizer = torch.optim.Adam(brainocr_model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # 데이터셋 로드
    train_dataset = OCRDataset("assets/trainset/training/real/001", "assets/trainset/training/label/001")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    validation_dataset = OCRDataset("assets/trainset/validation/real/001", "assets/trainset/validation/label/001")
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

    epochs = 10
    for epoch in range(epochs):
        brainocr_model.train()
        total_loss = 0

        for images, labels in train_loader:
            # 이미지와 라벨을 MPS 장치로 이동
            images = images.to(device)
            labels = labels.to(device)

            # 모델에 입력
            outputs = brainocr_model(images)
            loss = criterion(outputs, labels)

            # 역전파 및 옵티마이저 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}')

        # 검증 단계
        brainocr_model.eval()
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = brainocr_model(images)
                loss = criterion(outputs, labels)

                print(f'Validation Loss: {loss.item():.4f}')

    # 파인튜닝 종료 후 모델 저장
    torch.save(brainocr_model.state_dict(),
               "/System/Volumes/Data/Users/kukjunlee/.pororo/misc/brainocr_finetuned.pt")
    print("파인튜닝된 모델이 저장되었습니다.")
