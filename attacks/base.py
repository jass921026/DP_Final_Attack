"""
檔案功能
--------
攻擊的母類 BaseAttack：所有 attacks/ 下的「完整實驗」類別都繼承它。一個攻擊 = 一份設定檔 + 輸出根目錄，
run() 執行整個流程（訓練／取得受害者資訊 -> 攻擊 -> 評估）並回傳結果字典。
命令列入口（run_balle.py、run_imprint.py、run_rmia.py）只做「讀設定 -> 建攻擊 -> run()」，所以新增攻擊時
繼承 BaseAttack、實作 run()，再加一個入口即可。底層的純函式（如 attacks/rmia.py 的 rmia_scores）保持獨立，方便測試。

使用方式（新增攻擊）
--------------------
    class MyAttack(BaseAttack):
        def run(self) -> dict:
            ...                                # 使用 self.cfg、self.root
    result = MyAttack(cfg, root="runs").run()
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAttack(ABC):
    """
    功能
        攻擊實驗的共同母類（抽象）。
    Pseudo code
        run(): 建構資料／模型／損失／訓練器（皆由 cfg 以 util.registry.build 建構）-> 執行攻擊 -> 回傳指標
    用法
        BalleReconstructionAttack(cfg).run()
    """

    def __init__(self, cfg: Dict[str, Any], root: str = "runs"):
        self.cfg, self.root = cfg, root

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """功能：執行完整實驗並回傳結果字典。子類別必須實作。"""
