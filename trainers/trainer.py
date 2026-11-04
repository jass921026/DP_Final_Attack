"""
檔案功能
--------
一般的 mini-batch 監督式訓練器 Trainer（BaseTrainer 的子類別）：只負責程序——
逐 epoch 走訪 DataLoader、搬到裝置、（可選）資料增強、forward、呼叫 loss_fn、反向傳播、優化器更新、（可選）定期評估。
模型、損失、資料、優化器都由外部傳入。RMIA 的參考／目標模型、Balle 的重建網路 RecoNN 都用它訓練。

使用方式
--------
    torch.manual_seed(seed)                      # 先固定種子，再建模型（初始化由種子決定）
    model = build(model_spec)
    loader = make_loader(dataset, indices, batch_size, shuffle=True, seed=seed)
    trainer = Trainer(model, build(loss_spec), loader, {"name": "sgd", "lr": 0.05, "momentum": 0.9}, epochs=20, device="mps",
                      augment=random_hflip)
    history = trainer.fit()                      # model 留在 device、eval 模式
"""
from typing import Any, Callable, Dict, List, Optional

import torch

from trainers.base import BaseTrainer
from trainers.optimizers import build_optimizer


def random_hflip(xb: torch.Tensor) -> torch.Tensor:
    """功能：每張影像以 0.5 機率水平翻轉（官方 CIFAR 訓練的簡單增強）。用法：Trainer(..., augment=random_hflip)。"""
    m = torch.rand(len(xb), device=xb.device) < 0.5
    return torch.where(m[:, None, None, None], xb.flip(3), xb)


class Trainer(BaseTrainer):
    """
    功能
        mini-batch 訓練程序。
    Pseudo code
        opt <- build_optimizer(optimizer_spec)
        for epoch in 1..epochs:
            for (x_b, y_b) in loader:                       # DataLoader；每個 epoch 的順序由 loader 的 sampler 決定
                x_b <- augment(x_b) 若有
                loss <- loss_fn(model(x_b), y_b)
                opt.zero_grad(); loss.backward(); opt.step()
            每 eval_every 個 epoch 呼叫 eval_fn(model) 並記錄
        return history
    用法
        Trainer(model, loss_fn, loader, optimizer_spec, epochs, device).fit()
    """

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, train_loader, optimizer: Dict[str, Any], epochs: int,
                 device: str = "cpu", augment: Optional[Callable] = None, eval_fn: Optional[Callable] = None, eval_every: int = 10):
        self.model, self.loss_fn, self.loader = model.to(device), loss_fn.to(device), train_loader
        self.optimizer_spec, self.epochs, self.device = optimizer, epochs, device
        self.augment, self.eval_fn, self.eval_every = augment, eval_fn, eval_every

    def fit(self) -> List[Dict[str, Any]]:
        """功能：執行 Pseudo code 的訓練迴圈；回傳評估紀錄（有 eval_fn 時）。"""
        # [對應 Pseudo code: opt]
        opt = build_optimizer(self.model.parameters(), self.optimizer_spec)
        history: List[Dict[str, Any]] = []
        self.model.train()
        for epoch in range(self.epochs):
            # [對應 Pseudo code: for (x_b, y_b) in loader]
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if self.augment is not None:
                    xb = self.augment(xb)
                # [對應 Pseudo code: loss 與更新]
                loss = self.loss_fn(self.model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # [對應 Pseudo code: 每 eval_every 個 epoch 評估]
            if self.eval_fn is not None and (epoch % self.eval_every == 0 or epoch == self.epochs - 1):
                self.model.eval()
                with torch.no_grad():
                    history.append({"epoch": epoch, "train_loss_last_batch": float(loss.detach()), **self.eval_fn(self.model)})
                self.model.train()
        self.model.eval()
        return history
