"""MWM 批量跑分（外接，可随时断开）。

调用 MWM 核心 evaluate_from_yuanwen，输出 40 心所 scores_final。
不隶属于 mental_world_model 核心包；未安装或未调用时不影响 MWM 主流程。
"""

from mwm_test_adapter.runner import TestAdapterRunner, TestSampleResult

__all__ = [
    "TestAdapterRunner",
    "TestSampleResult",
]
