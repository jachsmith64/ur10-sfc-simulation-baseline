"""Focused text-only audit package; all six necessary case traces retain full1ms precision."""
from pathlib import Path
import sys,json,hashlib,zipfile,difflib
ROOT=Path(__file__).resolve().parent
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(ROOT.parent/'F组首次接入_20261010/.runtime'))
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    old=Path(json.loads((ROOT/'review/source_version.json').read_text(encoding='utf-8'))['accepted_complete_root'])
    with np.load(ROOT/'inputs/public_slow_correction.npz') as z,np.load(ROOT/'inputs/frozen_common.npz') as c:
        cols={'t_s':c['t']};schema={}
        for key in z.files:
            a=z[key];names=[key] if a.ndim==1 else [key+'_J'+str(j+1) for j in range(a.shape[1])]
            for j,n in enumerate(names):cols[n]=a if a.ndim==1 else a[:,j]
            schema[key]=dict(dtype=str(a.dtype),shape=list(a.shape),columns=names)
        mat=np.column_stack(list(cols.values()));p=ROOT/'inputs/public_slow_full_1ms.csv'
        np.savetxt(p,mat,delimiter=',',fmt='%.17g',header=','.join(cols),comments='')
        assert np.array_equal(np.loadtxt(p,delimiter=',',skiprows=1),mat)
        write(ROOT/'review/public_input_text_schema.json',schema)
    # Source full1ms text is inherited unchanged; schema likewise, not a new NPZ container identity.
    assert sha(ROOT/'inputs/frozen_common_full_1ms.csv')==sha(old/'inputs/frozen_common_full_1ms.csv')
    patch='';changed=[];added=[]
    code=[p for p in sorted(ROOT.glob('*.py')) if p.name not in ['check_review.py','pack_review.py']]
    for p in code:
        origin=old/p.name
        if origin.exists():
            if sha(origin)==sha(p):continue
            changed.append(p.name);before=origin.read_text(encoding='utf-8-sig').splitlines(True)
        else:added.append(p.name);before=[]
        patch+=''.join(difflib.unified_diff(before,p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='accepted_F10/'+p.name if origin.exists() else '/dev/null',tofile='F10_public_slow/'+p.name))
    assert set(changed)=={'run_short.py','position_logging.py','sweep.py'},changed
    (ROOT/'review/本轮代码差异.patch').write_text(patch,encoding='utf-8')
    write(ROOT/'review/code_changes.json',dict(changed_existing=changed,added=added,
        execution_change='One optional frozen position correction added before existing anti-jitter compensation; core controller/dynamics/limits unchanged.',
        guard_change='R026 validates generated and cached upstream results before dependent formal use.',
        old_originals_preserved=True))
    (ROOT/'核查包说明.md').write_text('''# 本轮文本核查包

先读简报.md、GPT核查入口.md；仅标准库运行 `python check_slow_review.py`。不运行仿真或连接机器人。

本机完整工程：C:/Users/kinli/Documents/ChatGPT/ur10实验/F组_预定慢漂移补偿_20261012。
已完成3条新积分（A0构造复用、线性、强SFC），最多6条。双击该完整目录的打开结果.cmd看图。run_slow.py为唯一新仿真入口，可复用缓存；旧矩阵命令行已禁用。核查ZIP不能代替完整工程运行MuJoCo。

outputs/compact_1ms含6条10,000行日志（新三方案、旧S3参照），17g往返无损；无50ms抽稀。必要列包括原始physical/effective Y、反馈速度、事件力、抗抖请求力、请求/实现抗抖力矩范数、六关节补偿、最终六关节位置命令和限幅。完整q/qdot、逐关节力矩、中间NPZ仍在本机。inputs仅原始未缩放背景/导数、冻结g、公共预定位置量/导数及原事件外力；逻辑哈希可从文本精确重建。容器NPZ哈希不可用重建文件冒充。

公共位置量与抗抖力/力矩代价分开。初始非零位置指令保留原初态，冷瞬态单列；停后50点只是短尾代表。主曲线原始、不减均值，旧A0−b仅诊断。离线预定纠偏不代表在线漂移处理完成；R027暂缓。

review/text_package_manifest.json冻结除自身/补充检查结果外全部包内文件。text_audit_result.json是补充检查结果；外层ZIP另有SHA256和CRC。ZIP无图片、HTML、模型网格、NPZ、旧长日志。XML仅供接口核查，完整网格留本机。旧short_report.py仅导入stats/table/LABELS，勿运行其历史15-case主入口。
''',encoding='utf-8')
    allowed={'.py','.md','.csv','.json','.xml','.patch','.txt'}
    chosen=set(code)|{ROOT/n for n in ['简报.md','GPT核查入口.md','核查包说明.md','f_config.json','assets_ur10e/ur10e.xml','assets_ur10e/scene.xml']}
    excluded={'text_package_manifest.json','archive_verification.json','text_audit_result.json'}
    chosen.update(p for p in (ROOT/'review').glob('*') if p.name not in excluded)
    chosen.update(ROOT/'inputs'/n for n in ['frozen_common_full_1ms.csv','public_slow_full_1ms.csv','Fd_event.csv','fixed_position_test_parameters.json','仿真误差参数.json'])
    chosen.update((ROOT/'outputs/compact_1ms').glob('*.csv'))
    chosen.update((ROOT/'outputs/full_1ms').glob('PC__*.json'))
    chosen.update((ROOT/'reference_old_S3').glob('*.json'))
    baseline_names=['frozen_version.json','implementation_freeze.json','compiled_model.json','frozen_input_text_schema.json','source_evidence.json',
        'event_reconstruction_window.json','用户确认运动边界.md','停止边界原始证据.csv','A_B原始事件分解_1ms.csv','事件来源到响应链.csv']
    chosen.update(ROOT/'baseline_review'/n for n in baseline_names)
    chosen={p for p in chosen if p.is_file() and p.suffix.lower() in allowed}
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(chosen)]
    manifest=ROOT/'review/text_package_manifest.json';write(manifest,dict(version='F10-PRESCHEDULED-SLOW-v1',files=files,
        full_1ms_case_rows=60000,only_text=True,excluded=['images','HTML','all NPZ','meshes','old full long logs'],
        supplemental_not_in_own_input_manifest=['review/text_audit_result.json']))
    chosen.add(manifest)
    import check_slow_review
    result=check_slow_review.main();audit=ROOT/'review/text_audit_result.json';write(audit,result);chosen.add(audit)
    dest=ROOT.parent/'F组预定慢漂移补偿_精简文本核查_20261012.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(chosen):z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert all(Path(x.filename).suffix.lower() in allowed for x in z.infolist())
        for p in chosen:assert hashlib.sha256(z.read(ROOT.name+'/'+p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
        count=len(z.infolist());raw=sum(x.file_size for x in z.infolist())
    info=dict(archive=str(dest),bytes=dest.stat().st_size,MiB=dest.stat().st_size/1024**2,sha256=sha(dest),files=count,uncompressed_bytes=raw,
        CRC_and_all_member_hashes_pass=True,compression='DEFLATE',text_only=True,case_rows=60000,new_integrations=3)
    write(ROOT/'review/archive_verification.json',info);dest.with_suffix('.zip.sha256.txt').write_text(info['sha256']+'  '+dest.name+'\n',encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
