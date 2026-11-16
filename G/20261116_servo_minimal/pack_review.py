"""Package review plus optional Git recovery and verify after extraction."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PREPARED=ROOT.parent/"20261116_A_multispeed/prepared_A"


def pack(delivery,bundle):
    rel=ROOT.relative_to(REPO).as_posix()+"/"
    tracked=subprocess.check_output(["git","-C",str(REPO),"ls-files","-z","--",rel]).decode().split("\0")
    members={f.removeprefix(rel):REPO/f for f in tracked if f and not f.endswith(".gitignore")}
    for path in (ROOT/"outputs").rglob("*"):
        if path.is_file() and path.suffix in (".npz",".json"):
            members[path.relative_to(ROOT).as_posix()]=path
    for path in PREPARED.rglob("*"):
        if path.is_file():
            members["source_arrays/"+path.relative_to(PREPARED).as_posix()]=path
    attach={"velocity_guard_reference.py":ROOT.parent/"20261116_preprocess_timing/run_preprocess.py",
            "source_A/01_工程/position_execution.py":ROOT.parent/"20261116_A_multispeed/v02_portable/source_A/01_工程/position_execution.py",
            "run_campaign.py":ROOT.parent/"20261116_A_multispeed/v02_portable/run_campaign.py",
            "turning/increment_model.py":REPO/"simulation/engineering_baseline_v2/cb3/turning/increment_model.py",
            "assets/ur10_cb3_b05_aligned.xml":REPO/"simulation/engineering_baseline_v2/cb3/assets/ur10_cb3_b05_aligned.xml"}
    members.update({"code_evidence/"+name:file for name,file in attach.items()})
    members["delivery.json"]=delivery
    if bundle:
        members["git_recovery/g_servo_minimal.bundle"]=bundle
    workspace=(ROOT/"local").resolve()
    workspace.mkdir(exist_ok=True)
    output=ROOT/"G_servo_minimal_review_20261116.zip"
    with TemporaryDirectory(prefix="review_",dir=workspace) as p:
        temp=Path(p).resolve()
        assert temp.parent==workspace
        archive=temp/"review.zip"
        manifest={}
        with ZipFile(archive,"w",ZIP_DEFLATED,compresslevel=7) as z:
            for name,path in sorted(members.items()):
                b=path.read_bytes()
                manifest[name]=dict(bytes=len(b),sha256=sha256(b).hexdigest())
                z.writestr(name,b)
            z.writestr("SHA256SUMS.json",json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
        assert archive.stat().st_size<20_000_000,archive.stat().st_size
        unpacked=temp/"unpacked"
        with ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(unpacked)
        subprocess.run([sys.executable,"-B","-X","utf8",str(unpacked/"verify_review.py"),str(unpacked)],check=True)
        copy2(archive,output)
    print(output,output.stat().st_size,sha256(output.read_bytes()).hexdigest())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--delivery",type=Path,required=True)
    p.add_argument("--bundle",type=Path)
    a=p.parse_args()
    pack(a.delivery,a.bundle)
