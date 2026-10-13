# 本轮GPT核查入口

先读简报.md、review/slow_verification.json、review/slow_frozen_version.json、review/baseline_plan.json。

真实命令在sweep.py唯一新增common_position_correction入口，原工艺q_nominal与Y参考未改变；position_logging.py分别记录original_base_ctrl/common_delta_q/抗抖delta_u/final_ctrl。新模块run_slow.py最多6条积分，A0只按4平台中心/约束构造，通过后冻结才跑线性/强SFC。没有拟合事件或重设g、μ。

输入补丁R026见input_acceptance.py/run_short.py/test_input_acceptance.py与review/R026_failure_checks.json：新生成预算耗尽未通过不formalize，坏缓存不调用仿真/力读取；合法g1无需放大分支保留。旧15条不复跑。

old__为旧S3三方案的1ms原始参照，new__为本轮公共纠偏后的实际新结果；公开公共输入仅一份，所有新方案共用。inputs/frozen_common_full_1ms.csv为原输入；inputs/public_slow_full_1ms.csv为公共基线/映射/导数。每条CSV全10,000行，17g无损往返；代码及diff随包。模型网格/完整中间NPZ不打包，完整工程在本机。

check_slow_review.py可仅用Python标准库复算文本包的共享输入哈希、目标/误差叠加、控制公式、最终命令、分段统计，不启动MuJoCo或机器人。图代码slow_report.py随包；图留本机outputs/figures。

问题库接续记录在review/持续问题库_本轮快照.md。R027为暂缓未解决，不能把本轮离线假设称为在线漂移处理完成。
