"""Offline delivery only: no simulation, controller changes, or hardware."""
import csv,difflib,hashlib,json,shutil,sys,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
W=R.parent
OLD=W/'F组_预定慢漂移补偿_20261012'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(x,msg):
    if not x:raise RuntimeError(msg)
ids=read(R/'review/accepted_run_ids.json')['run_ids']
py=Path(sys.executable)
for title,command in [('新仿真','new'),('查看第一批','show --run-id '+ids[0]),('查看第二批','show --run-id '+ids[1]),('仅重画第一批','redraw --run-id '+ids[0])]:
    (R/(title+'.cmd')).write_bytes(('@echo off\r\ncd /d "%~dp0"\r\n"'+str(py)+'" run_slow.py '+command+'\r\nif errorlevel 1 echo FAILED: no successful result or fallback.\r\npause\r\n').encode('ascii'))
protected={}
for p in [R/'review/protected_before.json',OLD/'review/protected_before.json']:protected.update(read(p))
bad=[p for p,h in protected.items() if not Path(p).is_file() or sha(p)!=h]
require(not bad,'Protected originals changed: '+str(bad))
save(R/'review/历史保护核查.json',dict(passed=True,checked_files=len(protected),changed=[],scope='Union of original F and its protected antecedents; pycache excluded'))
shutil.copy2(W/'F组持续问题库.md',R/'review/持续问题库_本轮快照.md')
post={}
for n in ['run_slow.py','test_engineering.py']:
    old=R/'review/积分时源码补存'/n
    if old.is_file():post[n]={'integration_source_sha256':sha(old),'current_sha256':sha(R/n),'change':'Post-integration historical artifact integrity guard and isolated verification only; physics/control/input unchanged'}
save(R/'review/积分后历史显示保护补充.json',dict(changes=post,real_integrations_added=0,negative_path_checks=read(R/'review/negative_path_checks.json'),historical_batches_not_claimed_current_source_reintegrated=True))
changes=[];patch=[]
for p in sorted(R.glob('*.py')):
    prev=OLD/p.name;a=prev.read_text(encoding='utf-8').splitlines(True) if prev.is_file() else []
    b=p.read_text(encoding='utf-8').splitlines(True)
    if a!=b:
        changes.append({'file':p.name,'type':'modified' if a else 'added','sha256':sha(p)})
        patch.extend(difflib.unified_diff(a,b,fromfile='accepted-v1/'+p.name if a else '/dev/null',tofile='candidate-v2/'+p.name))
(R/'review/代码差异.patch').write_text(''.join(patch),encoding='utf-8')
save(R/'review/改动文件.json',changes)
brief='''# F组工程修复候选：一页简报

本轮只修新运行、失败判定和数据追溯，并定向核查补偿执行证据；未修改控制参数、冻结背景、事件、公共纠偏、模型或初态。完整工程保存在本目录，原B/F及已审核结果保持。

新运行路径：点击“新仿真.cmd”→创建唯一run_id→A0、线性、强SFC分别实际积分→逐组核对真实返回状态、10000点、字段/形状/有限值、时间和输入身份→三组通过后生成本批图表并原子写成功标志。失败非零退出、保留有效前缀，停止依赖比较，不自动调参或回退旧结果。查看/重画必须指定run_id；检查日志、CSV、图及指标表，损坏不静默打开。根目录旧run_short/slow_report等保留供历史核查，当前新入口不经过它们。

|实际新批次|三组积分|结果|
|---|---:|---|
|`FIRST`|3 × 10000步|全部通过，未饱和|
|`SECOND`|3 × 10000步|全部通过，未饱和|

两批及旧已审核数据共294项逐数组比较，最大差异0；本轮恰好6次新积分，未追平旧数值而修改物理或参数。隔离检查9组、9种失败数据，含旧结果存在/损坏、XML变化、目录冲突、运行异常、FAILED/INVALID/短日志/非有限值/身份或计数错误、源码运行中变化及历史损坏；在Python -O下通过，桩测试0次真实积分。模型XML、全部依赖资产、输入/参数、源码、运行环境及Git状态均留档。积分后补充历史图表完整性保护，保存积分时对应源码，另做桩验证；未增加积分或改写历史生成身份。

|全程effective Y，含冷初始化|std μm|峰峰 μm|请求力RMS N|请求力矩RMS范数 N·m|
|---|---:|---:|---:|---:|
|无附加抗抖|28.826|197.586|0|0|
|线性|19.897|137.042|0.1810|0.1255|
|强SFC|15.977|109.439|0.3951|0.2742|

强SFC请求力RMS为线性的2.18倍，尚不支持等代价机制优势结论。各批保留完整1ms日志和有符号六关节向量；完整往返、去程及局部原始图分开显示physical/effective，未减均值、平滑或改幅值，指标/图绑定本批数据SHA。总跟踪误差不当作补偿独有误差，同状态力矩增量不当作补偿位置精度。

R-017：核查72次微动、5条补测；实际使用异步moveJ/moveL，RTDE约8ms。command_q来自机器人内部getTargetQ，非外部逐1ms发送流。公共纠偏最大约0.001845°，低于C最小0.01°，当前强SFC部分J1幅值虽有重叠，更新/变化速度、多轴及姿态覆盖仍不足。静态背景来自微动，动态背景主要来自路径普通窗；已有动力学也表示跟随差异，不能再把总跟踪误差拆成多个噪声。现资料不能辨识当前小补偿的独立执行关系。

交付R017证据表及最小接口建议：总目标（名义＋原背景＋公共纠偏＋抗抖）共同经过执行关系，再进入机械动力学；控制适配器只用冻结名义参数，新增关系关闭应还原基线。本轮没有实施新执行模型。R-028/R-029仅在本候选当前入口关闭；R-017模型、8ms/延迟、等代价及在线漂移仍待处理。未创建远端、推送或发布“稳定版”。
'''.replace('FIRST',ids[0]).replace('SECOND',ids[1])
(R/'简报.md').write_text(brief,encoding='utf-8')
(R/'review/发布检查清单.md').write_text('''# 候选发布检查清单

- [x] 独立完整工程及原件保护；当前入口无跨运行结果返回。
- [x] 失败状态、有效前缀、显式历史ID及图表完整性检查；-O隔离验证。
- [x] 两批6次实际积分，294项数组复现；完整1ms日志、原始图和控制代价。
- [x] R-017资料核查完成，最小接口及缺失观测已列。
- [ ] 依据确定后实现执行模型，并验证关闭/零偏差还原基线；本轮未授权实施。
- [ ] 向实机推进前核查8ms、真实反馈与延迟；当前1ms理想条件不等于CB3已验证。
- [ ] 主张SFC机制优势前完成等代价比较；在线漂移另议。
- [ ] 用户确定发布路线并验收新模型/新图后才发布。当前未建远端或推送。

现有上级Git仓库状态已记录于每批generation.json；F工程主要为未跟踪工作副本，HEAD不能单独识别本候选。逐文件SHA和差异补充身份；未提交不表示已发布。完整工程为本地备份，文本核查zip不含仿真网格，不能独立作为全模型工程运行。
''',encoding='utf-8')
# Prepare exact section11-only update for guarded external save.
doc=Path('C:/Users/kinli/Desktop/科研代码工作总览/PROJECTS/流程参考/B组_MuJoCo_SFC_V1.3与V1.2_流程快照.md')
raw=doc.read_bytes();marker='## 11. '.encode();at=raw.index(marker);prefix=raw[:20446];separator=raw[20446:at]
require(hashlib.sha256(prefix).hexdigest()=='72272015cdfa185f39e81858b5309cac5cf78a450c4ab960fb2943da94b5bfd4','Historical sections1–10 identity mismatch')
s=raw[at:].decode('utf-8')
s=s.replace('本次补充仅编辑文档，不运行仿真或修改业务代码。','此前补充仅编辑文档；随后本日F工程修复任务另有6次实际积分，证据在本节接续记录，未改变旧B历史。')
anchor='本节提醒不授权修改历史代码、重建扰动、重跑全矩阵、开发新控制器或实机实验。'
addition='''本日接续的“F组理想执行基线_工程修复候选”只继承已接受的冻结输入；新运行每次新run_id，三方案实际积分，旧结果存在或哈希相符均不允许跨运行返回。模型XML、全部依赖资产、源码、配置、输入及环境身份用于追溯，不是结果复用许可。历史查看/重画必须指定批次，只读其自带输入和完整数据，损坏/缺失/未完成不回退另一批。

R-028/R-029仅在该候选当前入口验证关闭：三组逐一核对返回状态、有效步数、数组形状/字段有限值、时间轴及输入身份，失败保留前缀并非零退出，全部通过后才生成正式报告/原子成功标志。9组隔离检查（包括9种失败数据、历史图表损坏）在-O模式通过；两批各3次真实1ms积分，共6次，294项与两批/旧审核数据比较最大差异0。未调整科研参数或物理实现。原B及旧F历史入口仍保留，不能概括为全工程都已修复。

'''
require(anchor in s,'Section11 insertion anchor missing');s=s.replace(anchor,addition+anchor)
s=s.replace('| R-014/R-015/R-017：CB3接口、UR10e差异、补偿自身误差 | 实机未验证；模型位置适配不能替代真实辨识 |','| R-014/R-015/R-017：CB3接口、UR10e差异、补偿自身误差 | R-017定向资料核查完成，执行模型仍未辨识/验证；72次微动和5条补测的内部目标约8ms，不能证明1ms小补偿；接口实机仍未验证 |\n| R-028/R-029：结果跨运行复用、失败判定 | 仅新候选当前入口已修并验证；旧B/F未回写，哈希一致不授予结果自动复用 |')
s+='''

本日工程修复接续证据（与上述首次文档补充区分）：[候选简报](<C:/Users/kinli/Documents/ChatGPT/ur10实验/F组_理想执行基线_工程修复候选_20261012/简报.md>)、[复现核查](<C:/Users/kinli/Documents/ChatGPT/ur10实验/F组_理想执行基线_工程修复候选_20261012/review/reproducibility.json>)、[R017资料与最小接口](<C:/Users/kinli/Documents/ChatGPT/ur10实验/F组_理想执行基线_工程修复候选_20261012/review/R017/资料核查与最小接口建议.md>)。总目标共同经过执行关系再进入动力学；新增不确定性时适配器只能使用冻结名义参数，不能读取实际不确定参数精确抵消。保留冻结背景，避免将已入背景的总误差重复添加。当前不实施无依据的死区/随机噪声，也不宣称CB3微补偿已验证。
'''
staged=R/'review/B流程参考_待原位保存.md';staged.write_bytes(prefix+separator+s.encode('utf-8'))
save(R/'review/B文档保存保护.json',dict(target=str(doc),before_sha256=hashlib.sha256(raw).hexdigest(),staged_sha256=sha(staged),sections1to10_sha256=hashlib.sha256(prefix).hexdigest()))
print('Prepared documentation, commands, diff and protection; originals',len(protected),'unchanged')
