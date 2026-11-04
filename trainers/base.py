"""
檔案功能
--------
訓練器的母類 BaseTrainer：所有 trainers/ 下的訓練器都繼承它。與 SUMMIT 的 trainer 一致，
訓練器只負責「組合與流程」：拿到 model / loss_fn / dataloader（或張量）/ 優化器設定，執行訓練程序；
損失定義在 loss/，資料在 datasets/（torch Dataset + DataLoader），訓練器不寫死任何一個。
介面：fit() 執行訓練並回傳結果（各子類別定義回傳什麼，例如訓練史、或所有模型的參數）。

使用方式（新增訓練器）
----------------------
    class MyTrainer(BaseTrainer):
        def __init__(self, model, loss_fn, train_loader, ...): ...
        def fit(self): ...
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """
    功能
        訓練器共同母類（抽象）。
    Pseudo code
        fit(): 組合 model、loss_fn、資料、優化器 -> 執行訓練程序 -> 回傳結果
    用法
        result = SomeTrainer(...).fit()
    """

    @abstractmethod
    def fit(self) -> Any:
        """功能：執行訓練並回傳結果。子類別必須實作。"""
