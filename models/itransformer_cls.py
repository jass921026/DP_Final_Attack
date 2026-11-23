"""
檔案功能
--------
iTransformer 分類器的薄包裝（用 models/iTransformer.py 的 Model）：接受建構子參數而非 Config 物件，輸入可以是
(B, seq_len, enc_in) 或攤平的 (B, seq_len * enc_in)（Balle 管線的目標是攤平向量），輸出 (B, num_class) 的 logits。
Balle 重建攻擊的演算法 A 需要確定性：dropout 預設 0（vmap 批次訓練也不支援隨機層）。

使用方式
--------
    {"module": "models.itransformer_cls", "type": "ITransformerClassifier",
     "kwargs": {"seq_len": 13, "enc_in": 41, "d_model": 32, "d_ff": 128, "n_heads": 2, "e_layers": 3, "num_class": 2}}
"""
import torch

from models.base import BaseModel
from models.iTransformer import Model


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ITransformerClassifier(BaseModel):
    """
    功能
        前向：x -> reshape 成 (B, seq_len, enc_in) -> iTransformer 分類頭 -> (B, num_class)。
    用法
        ITransformerClassifier(seq_len=13, enc_in=41, num_class=2)(torch.zeros(4, 13 * 41))
    """

    def __init__(self, seq_len: int = 13, enc_in: int = 41, d_model: int = 32, d_ff: int = 128, n_heads: int = 2, e_layers: int = 3,
                 num_class: int = 2, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.seq_len, self.enc_in = seq_len, enc_in
        self.net = Model(_Cfg(task_name="classification", seq_len=seq_len, pred_len=0, enc_in=enc_in, num_class=num_class, d_model=d_model,
                              d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation, e_layers=e_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], self.seq_len, self.enc_in), None, None, None)
