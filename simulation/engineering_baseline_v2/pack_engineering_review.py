"""Text-only review package, with exact duplicate batch CSVs represented once."""
import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
ids=read(R/'review/accepted_run_ids.json')['run_ids'];files={}
def add(p,n=None):
    p=Path(p);files[n or p.relative_to(R).as_posix()]=p
for p in R.iterdir():
    if p.is_file() and p.suffix in ['.py','.md','.json','.cmd']:add(p)
for p in (R/'review').rglob('*'):
    if p.is_file() and p.suffix in ['.md','.json','.csv','.patch','.txt','.py'] and p.name not in ['B流程参考_待原位保存.md','negative_path_console.txt']:add(p)
for p in (R/'inputs').iterdir():
    if p.suffix in ['.json','.csv'] and p.name!='Fd_zero.csv':add(p)
for p in (R/'assets_ur10e').rglob('*.xml'):add(p)
for p in (R/'historical/public_slow_review').glob('*.json'):add(p)
for p in (R/'baseline_review').glob('*.json'):add(p)
dedup=dict(run_ids=ids,local_full_batches_preserved=True,rule='Only packaging exact SHA-identical CSV files once. Both batches separately actually integrated. No cross-run simulation result reuse.',cases={})
for rid in ids:
    d=R/'runs'/rid
    for p in d.glob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.md']:add(p)
    for p in (d/'full_1ms').glob('*.json'):add(p)
    for p in (d/'figures').glob('*.json'):add(p)
    for p in (d/'frozen_inputs').glob('*.json'):add(p)
for ctrl in ['A0','Alinear','ASFC_strong']:
    a=R/'runs'/ids[0]/'csv_1ms'/(ctrl+'.csv');b=R/'runs'/ids[1]/'csv_1ms'/(ctrl+'.csv')
    if sha(a)!=sha(b):raise RuntimeError('CSV differs; cannot deduplicate '+ctrl)
    add(a);dedup['cases'][ctrl]=dict(package_csv=a.relative_to(R).as_posix(),csv_sha256=sha(a),same_data_in_second_run=b.relative_to(R).as_posix(),samples=10000,signed_columns=99)
save(R/'review/data_deduplication.json',dedup);add(R/'review/data_deduplication.json')
guide='''# GPT核查入口

先读简报.md、README.md、review/reproducibility.json、review/negative_path_checks.json及review/R017/资料核查与最小接口建议.md，再查代码差异和逐数组证据。

本包仅文本：代码、MD、CSV、JSON、XML及patch，无图片、网格、NPZ、3.7GB原采集或运行环境。完整模型/两批全部原始日志及图仍在本机完整工程；此包用于审阅，不能单独运行MuJoCo全工程。

两批各三次真正新积分，完整1ms有符号99列CSV共30000行在第一批csv_1ms。第二批同列CSV逐文件SHA完全一致，仅为缩小交付包去重；两批生成身份、验证、完成状态、指标及CSV_schema独立保留，关联见review/data_deduplication.json。49字段NPZ的294项逐数组比较见review/两批及旧结果逐数组复现.csv，NPZ留本机；包内99列不包含每轴全部布尔标志和公共二阶导数，这些在本机NPZ，原输入导数也在inputs完整文本表。本包不把包装去重当作结果自动复用。

标准库自检：python check_engineering_review.py 核查包.zip；核对全包SHA、两批CSV身份、完整行数/有符号通道定义、共同输入及两批全部分窗指标、控制代价和事件统计，不积分。Python -O隔离桩检查证明软件分支，不能当物理模型实机验收。

每批generation.json记录积分时源码/模型/环境/Git身份；review/积分时源码补存保存积分后调整前的历史查看保护源码。积分后增加图表损坏检查和桩验证，不修改动力学/控制/输入，也未额外积分。辅助资料核查、说明和打包代码在积分后完成，不宣称属于当时生成清单。

R017证据来自原C微动/补测；关节command_q为内部getTargetQ。不能辨识当前1ms补偿执行关系；原文件未全量重新哈希的范围见evidence_manifest.json。不得据此新增无依据噪声或宣称CB3已验证。本轮未发布稳定版。
'''
(R/'GPT核查入口.md').write_text(guide,encoding='utf-8');add(R/'GPT核查入口.md')
doc=Path('C:/Users/kinli/Desktop/科研代码工作总览/PROJECTS/流程参考/B组_MuJoCo_SFC_V1.3与V1.2_流程快照.md');add(doc,'reference/B组流程参考_含本轮第11节.md')
guard=read(R/'review/B文档保存保护.json')
if sha(doc)!=guard['staged_sha256']:raise RuntimeError('B reference not saved as prepared')
save(R/'review/B文档保存验收.json',dict(passed=True,target=str(doc),after_sha256=sha(doc),sections1to10_sha256=hashlib.sha256(doc.read_bytes()[:20446]).hexdigest()))
add(R/'review/B文档保存验收.json')
out=R/'F组工程修复_GPT文本核查.zip'
manifest=dict(text_only=True,images=0,meshes=0,npz=0,data_policy='Both actual batches preserved locally; exact duplicate CSVs packaged once',files={n:sha(p) for n,p in files.items()})
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_LZMA) as z:
    for n,p in sorted(files.items()):z.write(p,n)
    z.writestr('package_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(dict(path=str(out),size_MiB=out.stat().st_size/1048576,files=len(files)+1,sha256=sha(out)),ensure_ascii=False))
