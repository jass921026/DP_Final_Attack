"""
檔案功能
--------
PhysioNet 2012（P12）：set-a 當訓練、set-b 當測試（與課程 HW1 的切分相同）。前處理邏輯整個搬進這個檔案（原本
是 import 課程 HW1 的 PhysioNet12 套件，見本檔 git 歷史；行為完全沒變，只是把 common.py / P12_preprocess.py /
d_window.py / e_impute.py / g_labels.py 裡實際會用到的那幾個函式複製進來，讓 `data/` 資料夾可以只放資料、不用
再放一份課程的程式碼）：2 小時一格、取 12 到 36 小時的視窗（13 格）、41 個變數、每位病人用視窗內自己的平均
補值、視窗內完全沒有資料的病人排除（各集合 4000 人中剩 3992 人）。標籤 = 院內死亡。
為 iTransformer 做的最小改造：
    1. 觀測遮罩：補值前記錄每格每個變數是否真的被測到；
    2. mode="values_mask"：把「標準化後的值」與「遮罩」沿變數軸串接成 82 個變數（token），mode="values"：只有 41 個值；
    3. 標準化：只用訓練集（set-a）真實觀測到的值算 mean / std；補值與補不到的（缺整個變數）之處為 0（= 平均）；補零的填充格為 0。
注意（DP 實驗要處理）：標準化統計來自訓練資料，是資料相關的前處理。

使用方式
--------
    from util.registry import build
    tr = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "train", "mode": "values_mask"}})
    te = build({"module": "datasets.p12", "type": "P12", "kwargs": {"split": "test", "mode": "values_mask"}})
    tr.x.shape   # (3992, 13, 82) float32；tr.y (3992,) int64；tr.lengths 真實格數
    # split="all"：set-a 加 set-b（7984 人）；flat=True：x 攤平成 (n, 13*C)
"""
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from datasets.base import BaseDataset

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "p12", "PhysioNet12")   # 複製進 repo 內、git 忽略的本地資料資料夾（只放資料，不用再放課程的前處理程式碼）

BIN_HOURS, WINDOW_START, WINDOW_END = 2, 12, 36
MAX_LEN = (WINDOW_END - WINDOW_START) // BIN_HOURS + 1  # 13
FEATURE_COLS = [
    "Weight", "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin",
    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3",
    "HCT", "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP",
    "NIMAP", "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate",
    "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
    "pH", "Age", "Height", "Gender", "ICUType",
]
STATIC_COLS = ["Age", "Gender", "Height", "ICUType", "Weight"]
COL_ORDER = ["pid", "timestamp"] + FEATURE_COLS + ["target", "seg_entry_cnt"]


def _load_patient_raw(path: Path) -> pd.DataFrame:
    """功能：讀一個病人的原始長表格（Time,Parameter,Value），拿掉 RecordID 這個識別碼列。"""
    df = pd.read_csv(path)
    return df[df.Parameter != "RecordID"]


def _load_outcomes(path: Path) -> pd.Series:
    """功能：讀結局檔，索引設成 RecordID。"""
    return pd.read_csv(path).set_index("RecordID")["In-hospital_death"]


def _convert_to_relative_hours(df: pd.DataFrame) -> pd.DataFrame:
    """功能：把 "HH:MM" 字串轉成浮點小時數（新增 Hour 欄）。"""
    df = df.copy()
    df["Hour"] = df["Time"].apply(lambda t: int(t.split(":")[0]) + int(t.split(":")[1]) / 60)
    return df


def _assign_bins(df: pd.DataFrame, bin_hours: int = BIN_HOURS) -> pd.DataFrame:
    """功能：右閉 2 小時一格（(0,2],(2,4],...,(46,48]）；Hour=0 的記錄（例如入院時的靜態欄位）落進第一格。"""
    df = df.copy()
    df["Bin"] = (np.ceil(df["Hour"] / bin_hours) * bin_hours).astype(int)
    df.loc[df["Bin"] == 0, "Bin"] = bin_hours
    return df


def _clean_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """功能：Height 用 -1 當「未測量」的哨兵值，視為缺值。"""
    df = df.copy()
    df.loc[(df.Parameter == "Height") & (df.Value == -1), "Value"] = np.nan
    return df


def _pivot_patient(df: pd.DataFrame) -> pd.DataFrame:
    """功能：長表轉寬表（每格一列、每個變數一欄），同格內多筆取平均；靜態欄位（STATIC_COLS）只量測一次，廣播到每一列。"""
    seg_entry_cnt = df.groupby("Bin")["Hour"].nunique()
    wide = df.pivot_table(index="Bin", columns="Parameter", values="Value", aggfunc="mean")
    wide = wide.reindex(columns=FEATURE_COLS)
    wide["seg_entry_cnt"] = seg_entry_cnt
    for col in STATIC_COLS:
        known = wide[col].dropna()
        if len(known):
            wide[col] = known.iloc[0]
    return wide


def _process_patient(path: Path, outcomes: pd.Series) -> pd.DataFrame:
    """功能：單一病人的完整前處理（讀取 -> 清哨兵值 -> 轉小時 -> 分格 -> 轉寬表），回傳 0-48hr 的完整寬表。"""
    pid = int(path.stem)
    df = _clean_sentinels(_load_patient_raw(path))
    df = _assign_bins(_convert_to_relative_hours(df))
    wide = _pivot_patient(df)
    wide["pid"] = pid
    wide["timestamp"] = wide.index.astype(int)
    wide["target"] = outcomes.loc[pid]
    return wide.reset_index(drop=True)[COL_ORDER]


def _process_set(set_dir: Path, outcomes_path: Path) -> pd.DataFrame:
    """功能：整個 set（資料夾下每個病人的 .txt）都跑一次 `_process_patient`，疊起來。"""
    outcomes = _load_outcomes(outcomes_path)
    return pd.concat([_process_patient(f, outcomes) for f in sorted(set_dir.glob("*.txt"))], ignore_index=True)


def _window_patient(full: pd.DataFrame) -> pd.DataFrame:
    """功能：只留下 WINDOW_START <= timestamp <= WINDOW_END 的列（兩端都算在內）。"""
    return full[(full.timestamp >= WINDOW_START) & (full.timestamp <= WINDOW_END)].reset_index(drop=True)


def _impute_patient(windowed: pd.DataFrame) -> pd.DataFrame:
    """功能：每個變數缺值補這個病人自己視窗內的平均（病人視窗內該變數完全沒量到時仍是 NaN）。"""
    imputed = windowed.copy()
    for col in FEATURE_COLS:
        imputed[col] = imputed[col].fillna(imputed[col].mean())
    return imputed


def _build_y(pids: List[int], outcomes_path: Path) -> np.ndarray:
    """功能：依 pids 的順序取出對應的標籤（院內死亡）。"""
    return _load_outcomes(outcomes_path).loc[pids].values.astype(np.int64)


def _build_raw(root: str) -> Dict[str, np.ndarray]:
    """功能：對 set-a/set-b 各跑一次前處理，建 (X 補值後, M 觀測遮罩, lengths, y)；X 仍含補不到的 NaN。"""
    root = Path(os.path.abspath(root))
    out = {}
    for tag, sdir, oc in (("train", root / "set-a", root / "Outcomes-a.txt"), ("test", root / "set-b", root / "Outcomes-b.txt")):
        full = _process_set(sdir, oc)
        X, M, L, pids = [], [], [], []
        for pid, g in full.groupby("pid"):
            w = _window_patient(g.reset_index(drop=True))
            if len(w) == 0:
                continue                                                  # 視窗內完全沒資料：排除
            obs = w[FEATURE_COLS].notna().values                         # 補值前的觀測遮罩
            imp = _impute_patient(w)[FEATURE_COLS].values.astype(np.float32)
            x = np.zeros((MAX_LEN, len(FEATURE_COLS)), np.float32); m = np.zeros((MAX_LEN, len(FEATURE_COLS)), np.float32)
            x[:len(imp)], m[:len(imp)] = imp, obs
            X.append(x); M.append(m); L.append(len(imp)); pids.append(pid)
        out.update({f"X_{tag}": np.stack(X), f"M_{tag}": np.stack(M), f"len_{tag}": np.array(L), f"y_{tag}": _build_y(pids, oc)})
    return out


class P12(BaseDataset):
    """
    功能
        split="train"（set-a）或 "test"（set-b）；x 為 (n, 13, 41 或 82) float32，y 為院內死亡。
    Pseudo code
        raw <- 快取或 _build_raw()
        mu, sd <- set-a 中「真實觀測到的值」的逐變數平均與標準差
        x_val <- (X - mu) / sd；NaN 與填充格 -> 0
        x <- concat(x_val, M) 若 mode == "values_mask"，否則 x_val
    用法
        P12(split="train", mode="values_mask")
    """

    def __init__(self, split: str = "train", mode: str = "values_mask", cache_dir: str = "data_cache", root: str = DEFAULT_ROOT, flat: bool = False):
        assert split in ("train", "test", "all") and mode in ("values", "values_mask")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "p12_set_ab.npz")
        if os.path.isfile(path):
            raw = dict(np.load(path))
        else:
            raw = _build_raw(root)
            np.savez(path, **raw)
        Xt, Mt = raw["X_train"], raw["M_train"].astype(bool)
        # [對應 Pseudo code: mu, sd] 只用訓練集真實觀測到的值
        mu = np.array([Xt[..., f][Mt[..., f]].mean() if Mt[..., f].any() else 0.0 for f in range(Xt.shape[-1])], np.float32)
        sd = np.array([Xt[..., f][Mt[..., f]].std() if Mt[..., f].any() else 1.0 for f in range(Xt.shape[-1])], np.float32)
        sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
        parts = ("train", "test") if split == "all" else (split,)             # "all" = set-a 在前、set-b 在後（Balle 攻擊的資料池用）
        X, M, L = (np.concatenate([raw[f"{k}_{p}"] for p in parts]) for k in ("X", "M", "len"))
        y = np.concatenate([raw[f"y_{p}"] for p in parts])
        valid = (np.arange(X.shape[1])[None, :] < L[:, None])[..., None]    # 非填充格
        xv = np.where(valid & ~np.isnan(X), (X - mu) / sd, 0.0).astype(np.float32)
        x = np.concatenate([xv, M.astype(np.float32)], axis=-1) if mode == "values_mask" else xv
        self.lengths, self.mean, self.std, self.mode = L, mu, sd, mode
        super().__init__(x.reshape(len(x), -1) if flat else x, y)
