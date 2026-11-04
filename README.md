# DP_Final_Attack — 隱私攻擊實驗框架

以同一套結構實作並執行三種攻擊：

| 攻擊 | 論文 | 入口 |
|---|---|---|
| 重建攻擊（RecoNN，知情對手） | Balle, Cherubin, Hayes, S&P 2022 | `run_balle.py` |
| imprint（Robbing the Fed） | Fowl et al., ICLR 2022 | `run_imprint.py` |
| RMIA（離線） | Zarifzadeh et al., ICML 2024 | `run_rmia.py` |

實驗的輸出（`runs/`、`results/`、`logs/`）與結果整理（`RESULTS.md`）都不納入版本控制（見 `.gitignore`）。

## 1. 設計

資料、模型、損失、訓練器、攻擊五件事彼此獨立，靠設定檔（JSON）連接：

```
datasets/   models/   loss/   trainers/        attacks/
   \          |        |       /                 |
    '--- 以 {"module","type","kwargs"} 在設定檔中指定（util/registry.build）---'
訓練結果（影子模型參數、softmax 訊號…）存成 RunRecord（spec 雜湊 -> 相同設定不重訓）
```

- **每個資料夾有一個母類，資料夾內所有類別都繼承它**（與 SUMMIT 的 `datasets/`、`models/`、`loss/`、`trainer/` 同樣的分法）：

  | 資料夾 | 母類 | 子類別 |
  |---|---|---|
  | `datasets/` | `BaseDataset`（torch `Dataset`，內存 numpy 的 x、y） | `MNIST`、`CIFAR10`、`ImageNetVal`、`SyntheticImages` |
  | `models/` | `BaseModel`（`nn.Module` + `num_params()`、`flat_params()`） | `MLP`、`SmallCNN`、`ResNet18`（`attacks/imprint.py` 的 `ImprintBlock` 也繼承它） |
  | `loss/` | `BaseLoss`（`forward` + 逐樣本的 `per_sample`） | `CrossEntropy`、`MAE`、`MSE`、`WeightedSum` |
  | `trainers/` | `BaseTrainer`（抽象 `fit()`） | `Trainer`（mini-batch）、`BatchedGDTrainer`（數千個小模型平行）、`FedSGDClient`、`FedAvgClient` |
  | `attacks/` | `BaseAttack`（抽象 `run()`） | `BalleReconstructionAttack`、`ImprintAttack`、`RMIAAttack` |

- **trainer 只負責組合與流程**：模型、損失、資料、優化器都由外部傳入。損失只在 `loss/` 定義（trainer 只呼叫 `loss_fn(output, target)`，需要自行組合時用 `loss_fn.per_sample`）；資料以 `torch Dataset` + `DataLoader` 提供（`util/data.py` 的 `make_loader`）。例外：Balle 的影子模型是「幾千個模型共用同一個固定集、全批次」，不適合 DataLoader，`BatchedGDTrainer` 直接吃從 Dataset 取出的陣列，但損失仍來自 `loss/`。
- **Hugging Face 只當資料來源與快取**：`datasets/` 的轉接器把 HF parquet 讀成 numpy，存本地 npz；對外介面一律是 torch Dataset。
- **訓練結果可復用**：`util/record.py` 的 `RunRecord` 把影響結果的一切（資料、切分、模型、損失、超參數）放進 spec，雜湊成 `record_id`；相同 spec 直接讀取，不重訓。不同攻擊可以共用同一個訓練結果（例如 RMIA 的同一組訊號掃不同的 R、γ、a）。
- **每個檔案的開頭說明功能與用法；每個函式有 docstring（功能、pseudo code、用法）；程式段落以 `# [對應 ...]` 標註對應的 pseudo code 步驟。** 來自論文的 pseudo code 抄自論文。
- **以官方實作驗證**：`tests/` 直接以檔案路徑載入官方程式碼（不需安裝），與我們的實作逐元素比對（見第 5 節）。

## 2. 目錄

```
util/        框架共用的基礎設施，與任何特定模型或攻擊無關：
               registry（設定檔 -> 物件）、data（make_loader：DataLoader 與可重現的打亂）、
               splits（不重疊的資料切分）、record（RunRecord 快取）、metrics（MSE、NN oracle）
datasets/    base, mnist, cifar10, imagenet_val, synthetic_images（測試用）
models/      base, mlp（可選 Haiku 初始化）, small_cnn, resnet（torchvision ResNet-18）
loss/        base, cross_entropy, regression（MAE、MSE）, combine（WeightedSum）
trainers/    base, trainer（mini-batch）, batched_gd（vmap/批次 GEMM 平行訓練）, fl_client（FedSGD/FedAvg）, optimizers
attacks/
  base.py               BaseAttack
  balle_recon.py        影子模型 + RecoNN 重建
  imprint.py            ImprintBlock / SparseImprintBlock、反演、雜訊、IIP、PSNR
  rmia.py               RMIA 分數、offline_a 調參、ROC 摘要（純 numpy，與模型無關）
  rmia_experiment.py    RMIA 實驗流程（成對模型 -> 訊號 -> 攻擊 -> 彙整）
configs/     所有實驗的設定（見第 4 節）
tests/       單元與一致性測試
run_balle.py / run_imprint.py / run_rmia.py   命令列入口
```

注意：頂層套件名 `datasets` 與 Hugging Face 的 `datasets` 套件同名。本框架不依賴後者；若你同時安裝了它，請確保從本 repo 根目錄執行（`pytest.ini` 已設 `pythonpath = .`），否則會 import 到錯的套件。

## 3. 安裝與執行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 版本為開發時實際使用的版本（Python 3.10, macOS, PyTorch 2.14.1）
pytest -q                                # 約 5 秒；沒有官方原始碼時，一致性測試會自動跳過
```

裝置：設定檔中的 `device`（`cpu` / `mps` / `cuda`）。Balle 的論文規模設定需要加速器。

### Balle（MNIST）
```bash
python run_balle.py --config configs/balle_mnist_paper_faithful.json --smoke   # 縮小規模，數秒
python run_balle.py --config configs/balle_mnist_paper_faithful.json           # 論文 Table VI 設定
```
首次執行會從 Hugging Face 下載 MNIST（公開）。輸出寫在 `runs/results/<name>_<record>/result.json`（不納入版本控制）。

### imprint（ImageNet 驗證集）
ImageNet 驗證集需要你自己登入 Hugging Face 並同意條款後下載（`ILSVRC/imagenet-1k` 的 validation parquet）；本框架不處理憑證。把分片放到某個目錄，並把設定檔 `data.kwargs.root` 指向它（預設 `../imprint_attack/data/imagenet/data`）。
```bash
python run_imprint.py --config configs/imprint_fedsgd_bins.json --smoke
python run_imprint.py --config configs/imprint_fedsgd_bins.json                # 64 張、bins 掃描，CPU 約 30 秒
python run_imprint.py --config configs/imprint_noise_B_thr2.json               # 雜訊 + IIP
python run_imprint.py --config configs/imprint_sparse_fedavg_8x8_lr1e-4_f64.json
```
`pretrained: true` 的設定會從 torchvision 下載 ResNet-18 權重。

### RMIA（CIFAR-10）
```bash
python run_rmia.py --config configs/rmia_cifar10_small.json --smoke
python run_rmia.py --config configs/rmia_cifar10_small.json      # 18 個小模型，MPS 約 80 分鐘
```
首次執行會下載 CIFAR-10（公開）。訓練出的訊號依設定雜湊快取，之後只改攻擊參數（R、γ、a）幾乎立即完成。
此設定使用小型 CNN；其他模型可在設定檔中替換。

## 4. 設定檔

| 前綴 | 內容 |
|---|---|
| `balle_mnist_notebook*` | 官方 notebook 的超參數（固定集 1K、影子 10K）；`_haiku_optax` 加上官方的初始化與優化器 |
| `balle_mnist_paper*` | 論文 Table VI（固定集 10K、影子 58K）；`_faithful` = 論文值 + 官方補齊論文沒寫的部分 |
| `balle_abl_*` | Table II 消融（已知初始化 / ReLU / 未知初始化），縮小規模，與官方同規模對照 |
| `imprint_fedsgd_bins` | 主實驗：bins 掃描 |
| `imprint_noise_*` | Laplace 梯度雜訊；A = notebook 區塊，B = 官方 breaching 區塊；`_thrN` = 雜訊感知門檻倍數 |
| `imprint_sparse_*` | sparse 區塊：FedSGD / FedAvg，float32 vs float64 |
| `imprint_table1_*` | 論文 Table 1：線性函數與分布 |
| `imprint_pre_*` | 預訓練 ResNet-18 版本 |
| `rmia_cifar10_small` | RMIA 本機端到端 |

設定檔結構以 `configs/balle_mnist_paper_faithful.json` 為例：`data`、`splits`、`released_model`、`algorithm_A`、`reconstructor` 各是一段 `{module, type, kwargs}` 或超參數。`_comment` 欄位說明該設定與論文／官方的關係。

## 5. 測試

```bash
pytest -q
```

`tests/` 包含單元測試，以及與官方實作的一致性測試：測試以檔案路徑直接載入官方程式碼（不需安裝），與本框架的實作逐元素比對。預期的目錄配置（與本 repo 同層）：

```
../imprint_attack/breaching/        https://github.com/JonasGeiping/breaching
../rmia/ml_privacy_meter/           https://github.com/privacytrustlab/ml_privacy_meter  （master）
```
找不到官方原始碼時，對應測試會 `skip`，其餘測試照常執行。

## 6. 擴充

每個資料夾都有母類；新增一個類別 = 繼承該資料夾的母類、放一個檔案，再在設定檔用 `{"module": "<資料夾>.<檔名>", "type": "<類別>", "kwargs": {...}}` 指定。

- **資料集**：繼承 `datasets.base.BaseDataset`，載入後呼叫 `super().__init__(x, y)`（x float32、y int64）。參考 `mnist.py`、`cifar10.py`（`split` 參數取 train/test）。
- **模型**：繼承 `models.base.BaseModel`，實作 `__init__` 與 `forward`。
- **損失**：繼承 `loss.base.BaseLoss`，實作 `per_sample`（回傳 (n,)）與 `forward`；多項損失以 `loss.combine.WeightedSum` 組合。
- **訓練器**：繼承 `trainers.base.BaseTrainer`，實作 `fit()`；參考 `trainers/trainer.py`。
- **攻擊**：繼承 `attacks.base.BaseAttack`，實作 `run()`；純函式（如 `rmia.py` 的分數計算）與流程分檔，並在 `tests/` 加與官方的一致性測試。

## 7. 設計上的限制

- Balle 的影子模型是「幾千個模型共用同一個固定集、全批次」的訓練，不經過 DataLoader；`BatchedGDTrainer` 直接吃從 Dataset 取出的陣列（損失仍來自 `loss/`）。
- 一次性實驗腳本、輸出目錄與結果整理不在版本控制內。
