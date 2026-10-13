# GPT核查入口

先读简报.md、README.md、review/reproducibility.json、review/negative_path_checks.json及review/R017/资料核查与最小接口建议.md，再查代码差异和逐数组证据。

本包仅文本：代码、MD、CSV、JSON、XML及patch，无图片、网格、NPZ、3.7GB原采集或运行环境。完整模型/两批全部原始日志及图仍在本机完整工程；此包用于审阅，不能单独运行MuJoCo全工程。

两批各三次真正新积分，完整1ms有符号99列CSV共30000行在第一批csv_1ms。第二批同列CSV逐文件SHA完全一致，仅为缩小交付包去重；两批生成身份、验证、完成状态、指标及CSV_schema独立保留，关联见review/data_deduplication.json。49字段NPZ的294项逐数组比较见review/两批及旧结果逐数组复现.csv，NPZ留本机；包内99列不包含每轴全部布尔标志和公共二阶导数，这些在本机NPZ，原输入导数也在inputs完整文本表。本包不把包装去重当作结果自动复用。

标准库自检：python check_engineering_review.py 核查包.zip；核对全包SHA、两批CSV身份、完整行数/有符号通道定义、共同输入及两批全部分窗指标、控制代价和事件统计，不积分。Python -O隔离桩检查证明软件分支，不能当物理模型实机验收。

每批generation.json记录积分时源码/模型/环境/Git身份；review/积分时源码补存保存积分后调整前的历史查看保护源码。积分后增加图表损坏检查和桩验证，不修改动力学/控制/输入，也未额外积分。辅助资料核查、说明和打包代码在积分后完成，不宣称属于当时生成清单。

R017证据来自原C微动/补测；关节command_q为内部getTargetQ。不能辨识当前1ms补偿执行关系；原文件未全量重新哈希的范围见evidence_manifest.json。不得据此新增无依据噪声或宣称CB3已验证。本轮未发布稳定版。
