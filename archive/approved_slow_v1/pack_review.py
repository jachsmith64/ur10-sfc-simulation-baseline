"""Text-only, lossless1ms package. No images, meshes, old or full intermediate NPZ."""
from pathlib import Path
import sys,json,hashlib,zipfile,difflib,csv
ROOT=Path(__file__).resolve().parent
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
runtime=ROOT.parent/'F组首次接入_20261010/.runtime'
if runtime.exists():sys.path.insert(0,str(runtime))
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    old=ROOT.parent/'F组_位置执行等效测试_20261011'
    # Each full frozen input column is retained exactly as text; this recovers logical identities.
    with np.load(ROOT/'inputs/frozen_common.npz') as z:
        cols={};schema={}
        for key in z.files:
            a=z[key];names=[key] if a.ndim==1 else [key+'_J'+str(j+1) for j in range(a.shape[1])]
            for j,n in enumerate(names):cols[n]=a if a.ndim==1 else a[:,j]
            schema[key]=dict(dtype=str(a.dtype),shape=list(a.shape),columns=names)
        p=ROOT/'inputs/frozen_common_full_1ms.csv';matrix=np.column_stack(list(cols.values()))
        np.savetxt(p,matrix,delimiter=',',fmt='%.17g',header=','.join(cols),comments='')
        assert np.array_equal(np.loadtxt(p,delimiter=',',skiprows=1),matrix)
        write(ROOT/'review/frozen_input_text_schema.json',schema)
    with np.load(ROOT/'reference_B_packet/plant_frf.npz') as z:
        np.savetxt(ROOT/'review/旧B离线频响_原始.csv',np.column_stack([z['freq_hz'],z['gain_re'],z['gain_im']]),delimiter=',',fmt='%.17g',header='freq_hz,gain_real_um_per_N,gain_imag_um_per_N',comments='')
        write(ROOT/'review/旧B频响元数据.json',dict(source_sha256=sha(ROOT/'reference_B_packet/plant_frf.npz'),dc_um_per_n=float(z['dc_um_per_n'].item())))
    patch=''
    changed=[];added=[]
    for p in sorted(ROOT.glob('*.py')):
        orig=old/p.name
        if orig.exists():
            if sha(orig)==sha(p):continue
            changed.append(p.name);before=orig.read_text(encoding='utf-8-sig').splitlines(True)
        else:added.append(p.name);before=[]
        patch+=''.join(difflib.unified_diff(before,p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='accepted_F/'+p.name if orig.exists() else '/dev/null',tofile='F10/'+p.name))
    (ROOT/'review/本轮完整代码差异.patch').write_text(patch,encoding='utf-8')
    assert set(changed)=={'position_logging.py','pack_review.py'},changed
    write(ROOT/'review/code_changes.json',dict(changed_existing=changed,added=added,core_formulas_and_execution_unchanged=True,
        distinction='Only position_logging.py changes a pre-existing simulation-related file. pack_review.py replaces an older packaging helper of the same name; it does not execute/integrate dynamics.'))
    (ROOT/'核查包说明.md').write_text('''# 文本验收包

入口：GPT核查入口.md、简报.md。核查脚本：`python check_review.py`，仅用Python标准库，无机器人或仿真连接。

15条case CSV各10,000行，完整1ms、17位有效数字，逐列float64读回完全相等。包含t、phase、目标Y、physical/effective误差、effective速度、事件外力、控制请求力、请求/实现附加力矩范数、6关节最终补偿和饱和标记；无需50ms抽稀。完整请求/实现6关节力矩、actual q/qdot、反事实MuJoCo状态的NPZ保留本机outputs/full_1ms。

inputs/frozen_common_full_1ms.csv保留完整10秒冻结输入，字段shape/dtype见review/frozen_input_text_schema.json。check_review.py独立重建逻辑input_id并核对15条共享输入、误差叠加、控制公式、补偿/力矩、饱和、240个分段/通道统计。NPZ容器哈希仍在本机，不能以重建NPZ的新容器哈希冒充原文件哈希。

保留原C参数/g1输入、原始A三次D1、旧B源分解与高精度补证片段，源哈希在review JSON。旧B现存原保存CSV是200Hz/6g；“旧B背景已验收F_1ms”是此前验收F重放同一旧B输入的日志，明确区分，未重跑B一致性。

ZIP无图片、网格、旧62秒NPZ，无法脱离本机完整工程独立执行MuJoCo。完整工程路径见简报；双击本机打开结果.cmd可看原始图。只为核查提供文本，不削减峰值统计采样率。

review/text_package_manifest.json包含除自身外所有上传文件的字节数/SHA256；本机另有ZIP外层SHA及CRC核验。仿真冻结在review/frozen_version.json，执行源码在review/implementation_freeze.json；报告/打包工具另由上传清单冻结。
''',encoding='utf-8')
    allowed={'.py','.md','.csv','.json','.xml','.patch','.txt'}
    chosen=set(ROOT.glob('*.py'))|{ROOT/'简报.md',ROOT/'GPT核查入口.md',ROOT/'核查包说明.md',ROOT/'f_config.json',ROOT/'assets_ur10e/ur10e.xml',ROOT/'assets_ur10e/scene.xml'}
    chosen.update(ROOT.glob('review/*'))
    chosen.update(ROOT.glob('inputs/*'))
    chosen.update(ROOT.glob('sources_A/*/*.csv'))
    chosen.update(ROOT.glob('outputs/compact_1ms/*.csv'))
    chosen.update(ROOT.glob('outputs/full_1ms/S*.json'))
    chosen={p for p in chosen if p.is_file() and p.suffix.lower() in allowed and p.name not in ['text_package_manifest.json','archive_verification.json','text_package_check.json']}
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(chosen)]
    manifest=ROOT/'review/text_package_manifest.json';write(manifest,dict(kind='text-only full1ms audit package',version='F10-1ms-Acal-event-v1',files=files,excluded=['images','meshes','all NPZ','old62s full logs','whole6joint intermediate state arrays']))
    chosen.add(manifest)
    import check_review
    check=check_review.main();write(ROOT/'review/text_package_check.json',check);chosen.add(ROOT/'review/text_package_check.json')
    # Check result is supplemental and outside its own input file list, avoiding a recursive hash.
    dest=ROOT.parent/'F组10秒往返_精简文本核查_20261012.zip'
    # ZIP method14 is lossless and is readable by Python's standard zipfile module.
    # Numeric CSV compresses more tightly than DEFLATE without removing any samples.
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_LZMA) as z:
        for p in sorted(chosen):z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert all(Path(x.filename).suffix.lower() in allowed for x in z.infolist())
        for item in files:assert hashlib.sha256(z.read(ROOT.name+'/'+item['path'])).hexdigest()==item['sha256']
        count=len(z.infolist());uncompressed=sum(i.file_size for i in z.infolist())
    info=dict(archive=str(dest),bytes=dest.stat().st_size,MiB=dest.stat().st_size/1024**2,sha256=sha(dest),files=count,uncompressed_bytes=uncompressed,
        archive_crc_pass=True,all_member_hashes_pass=True,text_only=True,compression='ZIP_LZMA method14; Python standard zipfile supported',full_1ms_rows=150000,no_images_meshes_old_full_logs=True)
    write(ROOT/'review/archive_verification.json',info)
    dest.with_suffix('.zip.sha256.txt').write_text(info['sha256']+'  '+dest.name+'\n',encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
