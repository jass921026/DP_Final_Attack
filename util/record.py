"""
檔案功能
--------
RunRecord：一次「訓練產物」的可儲存、可復用單位。一個資料夾，內容是
    spec.json   訓練設定（完整複本，之後不需要原始設定檔）
    arrays.npz  數值產物（例如影子模型參數、目標、成員矩陣）
    record_id   = spec 的內容雜湊；讀取時會核對，避免設定與產物被錯配
攻擊程式只讀 RunRecord，不碰訓練程式；因此同一份訓練產物可以套用不同攻擊 / 不同重建器設定。

使用方式
--------
    from util.record import RunRecord
    rec = RunRecord.save("runs/shadow_mnist_a1b2", spec={...}, arrays={"params": P, "targets": X})
    rec = RunRecord.load("runs/shadow_mnist_a1b2")
    rec.spec, rec.arrays["params"], rec.record_id
    RunRecord.exists_for_spec(root_dir, spec)   # 依 spec 找得到就復用，不用重訓
"""
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def _canonical(spec: Dict[str, Any]) -> str:
    """spec 的正規化 JSON（key 排序、無空白），用於雜湊。"""
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec: Dict[str, Any]) -> str:
    """
    功能
        spec 的內容雜湊（前 16 碼），相同設定 -> 相同 id。
    Pseudo code
        1. 把 spec 序列化成正規化 JSON
        2. sha256 取前 16 個十六進位字元
    用法
        spec_hash({"seed": 0, "epochs": 100})
    """
    # [對應 步驟 1]
    s = _canonical(spec)
    # [對應 步驟 2]
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class RunRecord:
    """已載入的訓練產物。欄位：path, spec, arrays, record_id。"""
    path: str
    spec: Dict[str, Any]
    arrays: Dict[str, np.ndarray]
    record_id: str

    @staticmethod
    def dir_for(root: str, spec: Dict[str, Any], tag: str = "run") -> str:
        """
        功能
            由 spec 決定資料夾名稱：<root>/<tag>_<spec_hash>。
        用法
            RunRecord.dir_for("runs", spec, tag="balle_shadow")
        """
        return os.path.join(root, f"{tag}_{spec_hash(spec)}")

    @classmethod
    def exists_for_spec(cls, root: str, spec: Dict[str, Any], tag: str = "run") -> bool:
        """
        功能
            判斷該 spec 的產物是否已存在（用來決定復用或重訓）。
        Pseudo code
            1. d <- dir_for(root, spec, tag)
            2. return d/spec.json 與 d/arrays.npz 都存在
        用法
            if RunRecord.exists_for_spec("runs", spec): rec = RunRecord.load(...)
        """
        # [對應 步驟 1]
        d = cls.dir_for(root, spec, tag)
        # [對應 步驟 2]
        return os.path.isfile(os.path.join(d, "spec.json")) and os.path.isfile(os.path.join(d, "arrays.npz"))

    @classmethod
    def save(cls, path: str, spec: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> "RunRecord":
        """
        功能
            寫出 spec.json、arrays.npz、record_id。
        Pseudo code
            1. mkdir path
            2. 寫 spec.json（正規化 JSON）
            3. 寫 arrays.npz（不壓縮，讀取快）
            4. 寫 record_id = spec_hash(spec)
        用法
            RunRecord.save(RunRecord.dir_for("runs", spec), spec, {"params": P})
        """
        # [對應 步驟 1]
        os.makedirs(path, exist_ok=True)
        # [對應 步驟 2]
        with open(os.path.join(path, "spec.json"), "w") as f:
            f.write(_canonical(spec))
        # [對應 步驟 3]
        np.savez(os.path.join(path, "arrays.npz"), **arrays)
        # [對應 步驟 4]
        rid = spec_hash(spec)
        with open(os.path.join(path, "record_id"), "w") as f:
            f.write(rid)
        return cls(path=path, spec=spec, arrays=dict(arrays), record_id=rid)

    @classmethod
    def load(cls, path: str) -> "RunRecord":
        """
        功能
            讀回 RunRecord，並核對 record_id 與 spec 是否一致（防止設定與產物錯配）。
        Pseudo code
            1. 讀 spec.json、arrays.npz、record_id
            2. 若 spec_hash(spec) != record_id -> 報錯
        用法
            rec = RunRecord.load("runs/xxx")
        """
        # [對應 步驟 1]
        with open(os.path.join(path, "spec.json")) as f:
            spec = json.load(f)
        with np.load(os.path.join(path, "arrays.npz")) as z:
            arrays = {k: z[k] for k in z.files}
        with open(os.path.join(path, "record_id")) as f:
            rid = f.read().strip()
        # [對應 步驟 2]
        if spec_hash(spec) != rid:
            raise ValueError(f"record at {path} is inconsistent: spec hash {spec_hash(spec)} != record_id {rid}")
        return cls(path=path, spec=spec, arrays=arrays, record_id=rid)
