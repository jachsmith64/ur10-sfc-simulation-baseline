"""Verify task inputs and unpack the archived A model and selected B evidence."""
from hashlib import file_digest
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
TASK = ROOT / "task_package/B_相机编码器SFC闭环任务"
A_NAME = "A_timing_8ms_vs_2ms_review_20261109_light.zip"
B_NAME = "B_反馈可用性最小核查_V2_20261110.zip"
SOURCE = ROOT / "source_A"
DATA = ROOT / "source_B"
DEPS = Path(os.environ.get("B_SFC_RUNTIME_DIR", str(ROOT.parent / "event_sfc_20261109/runtime_local")))


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def unpack(path, dest, select):
    dest.mkdir()
    used = set()
    with ZipFile(path) as z:
        assert z.testzip() is None
        for entry in z.infolist():
            rel = PurePosixPath(entry.filename)
            assert rel.parts and not rel.is_absolute() and ".." not in rel.parts and "\\" not in str(rel) and ":" not in str(rel)
            assert str(rel).casefold() not in used
            used.add(str(rel).casefold())
            if entry.is_dir() or not select(entry.filename):
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as src, target.open("xb") as out:
                shutil.copyfileobj(src, out)


def main():
    verified = read(TASK/"manifest.json")["files"]
    for row in verified:
        p = TASK/row["path"]
        assert sha(p) == row["sha256"] and p.stat().st_size == row["bytes"]
    a = TASK/"02_已有材料"/A_NAME
    b = TASK/"02_已有材料"/B_NAME
    unpack(a,SOURCE,lambda _:True)
    unpack(b,DATA,lambda s: s in {
        "manifest.json", "inputs/v2/inputs/sources/path_mapping_and_origin.json",
        "02_结果/series/path-six-v1-r1.csv", "02_结果/series/path-six-v3-r1.csv",
        "inputs/native_rtde/path-six-v1-r1.csv", "inputs/native_rtde/path-six-v3-r1.csv",
    })
    b_entries = {x["path"]:x for x in read(DATA/"manifest.json")}
    for p in DATA.rglob("*"):
        if p.is_file() and p.name != "manifest.json":
            assert sha(p) == b_entries[p.relative_to(DATA).as_posix()]["sha256"]
    write(ROOT/"source_receipt.json", dict(a_name=A_NAME,a_sha256=sha(a),b_name=B_NAME,b_sha256=sha(b),
        task_members=verified,a_extracted={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob("*") if p.is_file()}))


if __name__ == "__main__":
    main()
