"""Redraw the approved original curves from the attached17-digit CSV; no simulation."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT.parent/'local_records/认可图_重画.png');parser.add_argument('--font',type=Path)
    args=parser.parse_args()
    source=ROOT/'F慢纠偏_三方案原始响应_数据.csv'
    identity=json.loads((ROOT/'认可图_版本身份.json').read_text(encoding='utf-8'))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=identity['plot_data_sha256']:raise RuntimeError('Approved CSV hash changed')
    data=np.genfromtxt(source,delimiter=',',names=True,encoding='utf-8-sig');t=data['t_s']
    if len(data)!=10000 or not np.array_equal(t,np.arange(10000)*.001):raise RuntimeError('Expected original10000 pre-step1ms points')
    font=args.font
    if font:
        fm.fontManager.addfont(str(font));font_name=fm.FontProperties(fname=font).get_name()
    else:
        fonts={x.name for x in fm.fontManager.ttflist}
        font_name=next((x for x in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','Source Han Sans SC'] if x in fonts),'DejaVu Sans')
        if font_name=='DejaVu Sans':print('Chinese font unavailable; pass --font path/to/NotoSansCJKsc-Regular.otf for readable Chinese labels.')
    plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'A0':'#8b939c','Alinear':'#2272a8','ASFC_strong':'#d04d4d'}
    labels={'A0':'公共纠偏，无附加抗抖','Alinear':'公共纠偏＋线性','ASFC_strong':'公共纠偏＋强SFC'}
    fig,axs=plt.subplots(2,1,figsize=(11.4,7.2),gridspec_kw={'height_ratios':[1.45,1]},layout='constrained')
    for ctrl in colors:axs[0].plot(t,data[ctrl+'_effective_e_um'],lw=1.1,color=colors[ctrl],label=labels[ctrl])
    for lo,hi,label in [(0,.8,'初始化'),(1.285,1.926,'A启动'),(4.58,5.643,'换向'),(8.667,9.999,'减速／停止')]:
        axs[0].axvspan(lo,hi,color='#7aa7cf' if lo else '#d4a66d',alpha=.12);axs[0].text((lo+hi)/2,107,label,ha='center',fontsize=9)
    axs[0].set(xlim=(0,10),ylim=(-115,122),ylabel='实际effective Y偏差 / 微米',xlabel='时间 / 秒',title='共同预定慢纠偏条件：完整10秒原始响应（零线仍为原名义轨迹）')
    axs[0].legend(ncol=3,loc='lower left',fontsize=9)
    m=(t>=2.226)&(t<=4.38)
    for ctrl in colors:axs[1].plot(data['along_mm'][m],data[ctrl+'_effective_e_um'][m],lw=1.2,color=colors[ctrl])
    axs[1].set(ylabel='原始Y偏差 / 微米',xlabel='普通去程名义沿程位置 / 毫米',title='普通去程局部：中心接近零，波纹仍保留')
    for ax in axs:ax.axhline(0,color='#333',lw=.8);ax.grid(alpha=.18)
    output=args.output.resolve()
    if output==ROOT/'F慢纠偏_三方案原始响应.png':raise RuntimeError('Do not overwrite the approved original PNG')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)
    print(json.dumps(dict(output=str(output),source_csv_sha256=identity['plot_data_sha256'],rows=len(data),raw_no_mean_removal=True,new_simulations=0,original_png_unchanged=True),ensure_ascii=False,indent=2))
if __name__=='__main__':
    import sys
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
