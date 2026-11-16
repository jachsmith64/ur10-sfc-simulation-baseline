"""Portable E-local dependency loading. Never imports old research directories."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(PACKAGE/'.e_deps'),str(PACKAGE/'.runtime')]
DLL_HANDLES=[]
if os.name=='nt':
    for p in (PACKAGE/'.e_deps').rglob('*.dll'):
        DLL_HANDLES.append(os.add_dll_directory(str(p.parent)))
