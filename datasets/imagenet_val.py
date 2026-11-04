"""
檔案功能
--------
載入 ImageNet 驗證集的一個 parquet 分片（Hugging Face ILSVRC/imagenet-1k，由使用者自行登入下載），
取 n 張影像：Resize(256) -> CenterCrop(224) -> [0,1] float32，形狀 (n,3,224,224)。
對應 Robbing the Fed（Fowl et al., ICLR 2022）的實驗資料：ImageNet 驗證集、ResNet-18、批次 64。
正規化（減平均除標準差）由攻擊端處理，因為 imprint 區塊放在模型最前面、吃的是正規化後的輸入。

使用方式
--------
    from util.registry import build
    ds = build({"module": "datasets.imagenet_val", "type": "ImageNetVal",
                "kwargs": {"root": "../imprint_attack/data/imagenet/data", "n": 64, "seed": 0}})
    ds.x   # (64,3,224,224) float32 in [0,1]
    ds.y   # (64,) int64 類別
    ds.get(np.arange(8))   # 前 8 張
"""
import glob
import io
from typing import Tuple

import numpy as np

from datasets.base import BaseDataset


class ImageNetVal(BaseDataset):
    """
    功能
        提供 x / y 與 get()。與 MNIST 轉接器相同的最小介面（x, y, __len__, get）。
    Pseudo code
        1. 讀取第一個 validation parquet 分片
        2. 以 seed 打亂；若 unique_labels，依序挑選類別不重複的影像直到 n 張（論文主實驗每張來自不同類別，避免標籤碰撞）
        3. 每張：Resize(256) -> CenterCrop(224) -> ToTensor
    用法
        ImageNetVal(root=".../data", n=64, seed=0, unique_labels=True)
    """

    def __init__(self, root: str, n: int, seed: int = 0, unique_labels: bool = True):
        import pyarrow.parquet as pq
        import torchvision
        from PIL import Image
        # [對應 步驟 1] 讀取第一個分片
        shard = sorted(glob.glob(f"{root}/validation-*.parquet"))[0]
        t = pq.read_table(shard)
        labels = np.array(t.column("label").to_pylist())
        # [對應 步驟 2] 打亂並挑選
        order = np.random.default_rng(seed).permutation(len(labels))
        pick, seen = [], set()
        for i in order:
            if unique_labels and labels[i] in seen:
                continue
            seen.add(labels[i]); pick.append(int(i))
            if len(pick) == n:
                break
        if len(pick) < n:                                   # 類別不夠時補足
            pick += [int(i) for i in order if int(i) not in pick][: n - len(pick)]
        # [對應 步驟 3] 影像前處理
        tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                                             torchvision.transforms.ToTensor()])
        imgs = [tf(Image.open(io.BytesIO(t.column("image")[i].as_py()["bytes"])).convert("RGB")) for i in pick]
        super().__init__(np.stack([im.numpy() for im in imgs]).astype(np.float32), labels[pick].astype(np.int64))
        self._shard, self._labels, self._pick, self._tf = shard, labels, pick, tf

    def class_mates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        功能
            取出同一分片中、與選中影像同類別、但不是選中影像本身的其他影像，供 IIP 度量使用
            （論文 Appendix A.5：重建影像是否比「驗證集中同類別的任何其他影像」更接近其真圖）。
            注意：只有一個分片（約 1/14 的驗證集），每類平均只有約 3 張同類影像，論文用完整驗證集（每類 50 張），
            所以這裡的 IIP 比論文容易通過。
        Pseudo code
            for 每個選中影像的類別 c:  mates_c <- 分片中類別為 c 且不在 pick 內的所有影像
        用法
            mx, my = ds.class_mates()     # mx: (m,3,224,224) float32，my: (m,) 類別
        """
        import io as _io
        import pyarrow.parquet as pq
        from PIL import Image
        t = pq.read_table(self._shard)
        want, skip = set(self.y.tolist()), set(self._pick)
        idx = [i for i in range(len(self._labels)) if int(self._labels[i]) in want and i not in skip]
        imgs = [self._tf(Image.open(_io.BytesIO(t.column("image")[i].as_py()["bytes"])).convert("RGB")).numpy() for i in idx]
        return np.stack(imgs).astype(np.float32), self._labels[idx].astype(np.int64)
