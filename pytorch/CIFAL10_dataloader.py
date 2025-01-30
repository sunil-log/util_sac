import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

"""
CIFAR_dataset 클래스는 DataLoader에 넣을 때마다 transform을 반복 적용하는 대신,
미리 이미지를 Tensor로 변환해 놓고 dataloader에 전달합니다.
이 방식은 매 스텝마다 PIL 이미지를 변환하는 과정을 줄여, 속도를 높이는 장점이 있습니다.
"""

class CIFAR_dataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_dataloader(batch_size=64, train=True, flatten=True, shuffle=True):
    """
    미리 변환된 CIFAR 이미지를 CIFAR_dataset 클래스에 담아 DataLoader로 반환합니다.
    """
    images, targets = prepare_data(train, flatten, cnn)
    dataset = CIFAR_dataset(images, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def prepare_data(train=True, flatten=True):
    """
    CIFAR-10 데이터를 다운로드 및 로드하고, 이미지 픽셀 값을 0-255에서 0-1 범위로 정규화합니다.
    선택적으로 이미지를 1차원으로 평탄화하거나(CNN 비사용 시) CNN용 차원을 유지합니다.

    이 함수는 다음 작업을 수행합니다:
    1. CIFAR-10 데이터셋을 다운로드합니다(필요한 경우).
    2. 각 이미지를 Tensor로 변환한 뒤, 0-1 범위로 정규화합니다.
    3. 선택적으로 이미지를 1차원으로 평탄화하거나 CNN용 차원을 유지합니다.
    4. 전처리가 완료된 images와 targets를 반환합니다.

    매개변수:
    train (bool): 학습용(train=True) 데이터셋을 로드할지 여부 (기본값: True)
    flatten (bool): 이미지를 1차원으로 평탄화할지 여부 (기본값: True)
    cnn (bool): 이미지에 (채널, 높이, 너비) 형태를 유지할지 여부 (기본값: True)

    반환값:
    images (Tensor): 전처리가 완료된 이미지 텐서
    targets (Tensor): 이미지 라벨 정보
    """
    # torchvision.datasets.CIFAR10의 기본 이미지는 PIL Image 형태이며, label만 integer로 제공됩니다.
    raw_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=train,
        download=True
    )

    # PIL -> Tensor 변환
    images_list = []
    targets_list = []
    for img, label in raw_data:
        # transforms.ToTensor()는 이미지를 [0,1] 범위로 자동 정규화함
        img_tensor = torchvision.transforms.ToTensor()(img)  # shape: (3, 32, 32)
        images_list.append(img_tensor)
        targets_list.append(label)

    # 하나의 Tensor로 스택
    images = torch.stack(images_list)  # shape: (N, 3, 32, 32)
    targets = torch.tensor(targets_list)

    # flatten=True인 경우, 이미지 1차원화
    if flatten:
        # (N, 채널*높이*너비)
        images = images.view(images.size(0), -1)

    return images, targets
