"""
檔案功能
--------
SUMMIT（TESNE，models/SUMMIT.py，https://arxiv.org/abs/2405.16557）分類器的薄包裝，跟 models/itransformer_cls.py
的 ITransformerClassifier 走同一套輸入規格，讓兩個模型可以共用同一個資料集：接受
(B, seq_len, 2*enc_in)（datasets.p12.P12 的 mode="values_mask"：前 enc_in 欄是標準化後的值、後 enc_in 欄是
觀測遮罩）或攤平的 (B, seq_len * 2*enc_in)，輸出 (B, num_class) 的 logits。

原版 TESNE 把資料切成 num/cat 兩路分別走獨立的 SNE、且需要外部資料集額外準備 x_num_idx/x_cat_idx 這兩個
「第幾個變數」的常數位置編碼（跟資料值無關，只跟欄位順序有關，見 code_submit_v2/datasets/physio_2012_dataset.py
的 np.arange(1, LEN+1).tile(...)）。這裡兩個改動都是為了跟 iTransformer 共用同一份資料集：
    1. 不切 num/cat，41 個變數統一走一路 SNE（value_LEN=enc_in）；
    2. idx 張量不用資料集準備，直接在 forward 裡用 torch.arange 建、注入 batch 維度即可（本來就跟資料值無關）。
Balle 重建攻擊的演算法 A 需要確定性：建構子預設把三個 dropout 都設 0（SUMMIT 原版預設非 0）。

使用方式
--------
    {"module": "models.summit_cls", "type": "SUMMITClassifier",
     "kwargs": {"seq_len": 13, "enc_in": 41, "d_model": 144, "num_heads": 1, "num_layers": 1, "num_class": 2}}
"""
import torch

from models.base import BaseModel
from models.SUMMIT import Encoder, EncoderLayer, SNE, SinusoidalEmbedding3d


class SUMMITClassifier(BaseModel):
    """
    功能
        前向：x -> reshape 成 (B, seq_len, 2*enc_in) -> 拆成 value/mask -> SNE -> + 位置編碼 -> transformer
        encoder（key_padding_mask 來自觀測遮罩）-> 對 token 維取平均 -> 線性分類頭 -> (B, num_class)。
    用法
        SUMMITClassifier(seq_len=13, enc_in=41, num_class=2)(torch.zeros(4, 13 * 82))
    """

    def __init__(self, seq_len: int = 13, enc_in: int = 41, d_model: int = 144, num_heads: int = 1, ff_dim: int = 144,
                 num_layers: int = 1, num_class: int = 2, decoder_down_factor: int = 2, ff_dropout: float = 0.0,
                 attn_dropout: float = 0.0, decoder_dropout: float = 0.0, norm_type: str = "LayerNorm",
                 only_mask_first: bool = False, max_norm: float = None):
        super().__init__()
        self.seq_len, self.enc_in = seq_len, enc_in
        self.embedding = SNE(d_model=d_model, value_LEN=enc_in, independent_padding=True, max_norm=max_norm)
        self.pos_emb = SinusoidalEmbedding3d(d_model, dropout=0.0, max_seq_len=5000)
        encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, ff_dropout=ff_dropout,
                                      attn_dropout=attn_dropout, norm_type=norm_type, seq_len=seq_len, num_value=enc_in)
        self.encoder = Encoder(encoder_layer=encoder_layer, num_layers=num_layers, only_mask_first=only_mask_first)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(d_model, d_model // decoder_down_factor), torch.nn.GELU(),
                                            torch.nn.Dropout(decoder_dropout), torch.nn.Linear(d_model // decoder_down_factor, num_class))
        self.register_buffer("_col_idx", torch.arange(1, enc_in + 1).long(), persistent=False)  # 跟資料值無關的常數欄位序號

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], self.seq_len, 2 * self.enc_in)
        x_val, x_mask = x[..., :self.enc_in], x[..., self.enc_in:]
        x_idx = self._col_idx.view(1, 1, -1).expand(x.shape[0], self.seq_len, -1)
        x_mask_bin = (x_mask > 0.5).long()  # 門檻二值化而非直接 cast：資料集的遮罩本來就是 0/1，這裡只是讓它在數值上更穩健
        x_emb = self.embedding(x_idx, x_val, x_mask_bin)
        x_emb = self.pos_emb(x_emb)
        x_emb = x_emb.reshape(x_emb.shape[0], -1, x_emb.shape[-1])
        x_encoded, _ = self.encoder(x_emb, x_mask_bin)
        return self.decoder(x_encoded.mean(dim=1))
