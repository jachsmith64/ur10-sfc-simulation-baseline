"""Package and validate the compact review, including Git handoff when needed."""
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
PRIOR=ROOT.parent/"20261116_A_multispeed"


def package(delivery,bundle=None):
    rel=ROOT.relative_to(REPO).as_posix()+"/"
    paths=subprocess.check_output(["git","-C",str(REPO),"ls-files","-z","--",rel]).decode().split("\0")
    members={name.removeprefix(rel):REPO/name for name in paths if name and not name.endswith(".gitignore")}
    members.update({p.relative_to(ROOT).as_posix():p for p in (ROOT/"results").rglob("*.npz")})
    for f in (PRIOR/"prepared_A").rglob("*"):
        if f.is_file():
            members["source_arrays/"+f.relative_to(PRIOR/"prepared_A").as_posix()]=f
    for name in ("filter_sensing.py","run_campaign.py","campaign.json",
        "source_A/01_工程/sweep.py","source_A/01_工程/position_execution.py",
        "source_A/01_工程/controller.py","source_A/01_工程/disturbance.py"):
        members["code_evidence/V02/"+name]=PRIOR/"v02_portable"/name
    cb3=REPO/"simulation/engineering_baseline_v2/cb3"
    for key in ("assets/ur10_cb3_b05_aligned.xml","turning/increment_model.py","turning/increment_config.json"):
        members["code_evidence/CB3/"+key]=cb3/key
    members["delivery.json"]=delivery
    if bundle:
        members["git_recovery/g_preprocess_timing.bundle"]=bundle
        # The bundle already contains byte-for-byte compact previous bindings.
        # The review metrics use results and source_arrays directly.
        members={name:f for name,f in members.items() if not
                 (name.startswith("inputs/previous/") and name.endswith(".npz"))}
    workspace=(ROOT/"local").resolve()
    workspace.mkdir(exist_ok=True)
    output=ROOT/"G_preprocess_timing_review_20261116.zip"
    with TemporaryDirectory(prefix="verify_review_",dir=workspace) as temp:
        directory=Path(temp).resolve()
        assert directory.parent==workspace
        archive=directory/"review.zip"
        manifest={}
        with ZipFile(archive,"w",ZIP_DEFLATED,compresslevel=7) as z:
            for name,file in sorted(members.items()):
                content=file.read_bytes()
                manifest[name]={"bytes":len(content),"sha256":sha256(content).hexdigest()}
                z.writestr(name,content)
            z.writestr("SHA256SUMS.json",json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
        assert archive.stat().st_size<20_000_000,archive.stat().st_size
        with ZipFile(archive) as z:
            assert z.testzip() is None
            unpacked=directory/"unpacked"
            z.extractall(unpacked)
        subprocess.run([sys.executable,"-B","-X","utf8",str(unpacked/"verify_review.py"),str(unpacked)],check=True)
        copy2(archive,output)
    print(output,output.stat().st_size,sha256(output.read_bytes()).hexdigest())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--delivery",required=True,type=Path)
    parser.add_argument("--bundle",type=Path)
    p=parser.parse_args()
    package(p.delivery,p.bundle)
