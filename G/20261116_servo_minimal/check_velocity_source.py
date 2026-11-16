"""Record target-velocity fields found in the B cabinet exports."""
from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path
import re

TRIALS=("X_P05_R01","X_P05_R02","X_P10_R01")


def check(data_root):
    result={}
    for name in TRIALS:
        case=data_root/name
        entries=[]
        for f in sorted(case.iterdir()):
            if not f.name.endswith(("_UR10.txt","_RUNLOG.txt")):
                continue
            raw=f.read_bytes()
            robot=[r for line in raw.decode("utf-8-sig").splitlines() if line.strip()
                   if (r:=json.loads(line)).get("kind")=="ROBOT"]
            keys=Counter(key for row in robot for key in row)
            target_velocity={k:n for k,n in keys.items() if ("target" in k.lower() or "desired" in k.lower())
                             and re.search("speed|vel|qd|deriv",k,re.I)}
            entries.append(dict(name=f.name,bytes=len(raw),sha256=sha256(raw).hexdigest(),
                robot_rows=len(robot),target_velocity_keys=target_velocity,all_robot_keys=dict(keys)))
        if not entries or any(e["target_velocity_keys"] for e in entries):
            raise RuntimeError(f"Unverified source or unexpectedly available target-velocity channel: {name}")
        result[name]=entries
    return dict(trials=result,used_velocity_source="analytic slopes of existing original-host IK nominal representation",
                actual_speed_use="X evaluation only",raw_paths_relative_to_supplied_original_root=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.write_text(json.dumps(check(a.data_root),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
