"""
檔案功能
--------
GRU 分類器的薄包裝，跟 models/itransformer_cls.py 的 ITransformerClassifier 走同一套輸入規格：接受
(B, seq_len, enc_in) 或攤平的 (B, seq_len * enc_in)，輸出 (B, num_class) 的 logits。用最後一層、最後一個時間
步的 hidden state 接線性分類頭。
Balle 重建攻擊的演算法 A 需要確定性：dropout 預設 0（num_layers=1 時 nn.GRU 的 dropout 參數本來就不會生效，
這裡仍明寫出來，num_layers>1 時才有意義）。

使用方式
--------
    {"module": "models.gru_cls", "type": "GRUClassifier",
     "kwargs": {"seq_len": 13, "enc_in": 41, "hidden": 64, "num_layers": 1, "num_class": 2}}
"""
import torch
from torch import nn

from models.base import BaseModel


class GRUClassifier(BaseModel):
    """
    功能
        前向：x -> reshape 成 (B, seq_len, enc_in) -> GRU -> 最後一層最後一步的 hidden -> 線性分類頭 -> (B, num_class)。
    用法
        GRUClassifier(seq_len=13, enc_in=41, num_class=2)(torch.zeros(4, 13 * 41))
    """

    def __init__(self, seq_len: int = 13, enc_in: int = 41, hidden: int = 64, num_layers: int = 1, num_class: int = 2, dropout: float = 0.0):
        super().__init__()
        self.seq_len, self.enc_in = seq_len, enc_in
        self.gru = nn.GRU(input_size=enc_in, hidden_size=hidden, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], self.seq_len, self.enc_in)
        _, h_n = self.gru(x)
        return self.head(h_n[-1])
