"""
檔案功能
--------
把設定檔裡的 {"module": ..., "type": ..., "kwargs": {...}} 解析成真正的 Python 物件。
這是 SUMMIT (code_submit_v2/main.py) 使用的同一種格式，所以 SUMMIT 的設定片段可以直接放進本框架。
沒有註冊表：新增一個模型 / 資料集 = 在 models/ 或 datasets/ 放一個檔案，設定檔指向它即可。

使用方式
--------
    from util.registry import build
    model = build({"module": "models.mlp", "type": "MLP",
                   "kwargs": {"in_dim": 784, "hidden": [10], "out_dim": 10, "activation": "elu"}})
    # 需要覆蓋 kwargs 時：build(spec, hidden=[20])

注意：本檔不 import 任何 SUMMIT 或官方攻擊程式碼。
"""
import importlib
from typing import Any, Dict


def build(spec: Dict[str, Any], **overrides: Any) -> Any:
    """
    功能
        依 spec 動態 import 並實例化物件。
    Pseudo code (本框架自訂，非論文)
        1. mod  <- import_module(spec["module"])
        2. cls  <- getattr(mod, spec["type"])
        3. kw   <- spec.get("kwargs", {}) 以 overrides 覆蓋
        4. return cls(**kw)
    用法
        obj = build({"module": "datasets.mnist", "type": "MNIST", "kwargs": {}})
    """
    # [對應 Pseudo code 步驟 1] 動態 import 模組
    module = importlib.import_module(spec["module"])
    # [對應 步驟 2] 取出類別 / 工廠函式
    cls = getattr(module, spec["type"])
    # [對應 步驟 3] 合併 kwargs（overrides 優先）
    kwargs = dict(spec.get("kwargs", {}))
    kwargs.update(overrides)
    # [對應 步驟 4] 實例化
    return cls(**kwargs)
