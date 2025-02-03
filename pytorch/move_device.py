
import torch
import numpy as np

def move_dict_tensors_to_device(d, device, float_dtype=torch.float32, int_dtype=torch.int64):
    """
    이 함수는 dictionary 형태로 주어진 데이터를 torch Tensor 로 변환하고,
    지정된 device 로 이동시켜줍니다. 데이터는 torch Tensor, numpy array,
    list 형태를 지원하며, float 과 int 값은 보통 PyTorch 에서 많이 사용하는
    float32 와 int64 로 변환됩니다.

    Args:
        d (dict): 여러 torch Tensor, numpy array, list 를 값으로 가지는 dictionary.
        device (torch.device): Tensor 를 옮길 대상 device.
        float_dtype (torch.dtype): float 타입 변환에 사용할 dtype (기본값: torch.float32).
        int_dtype (torch.dtype): int 타입 변환에 사용할 dtype (기본값: torch.int64).

    Returns:
        dict: 각 값이 torch Tensor 로 변환되어, 지정된 dtype 및 device 로 이동된 dictionary.
    """


    new_d = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # 이미 torch Tensor 인 경우
            if torch.is_floating_point(v):
                new_d[k] = v.to(device, dtype=float_dtype)
            else:
                new_d[k] = v.to(device, dtype=int_dtype)
        elif isinstance(v, np.ndarray):
            # numpy array 인 경우
            if np.issubdtype(v.dtype, np.floating):
                new_d[k] = torch.from_numpy(v).to(device, dtype=float_dtype)
            else:
                new_d[k] = torch.from_numpy(v).to(device, dtype=int_dtype)
        elif isinstance(v, list):
            # list 인 경우, 원소가 부동소수점인지 정수인지 판단
            if all(isinstance(x, float) for x in v):
                new_d[k] = torch.tensor(v, device=device, dtype=float_dtype)
            else:
                new_d[k] = torch.tensor(v, device=device, dtype=int_dtype)
        else:
            raise TypeError(f"Unsupported data type for key '{k}': {type(v)}")

    return new_d
