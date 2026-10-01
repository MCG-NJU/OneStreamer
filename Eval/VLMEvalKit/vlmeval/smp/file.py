import os, os.path as osp, json, pickle, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        return super().default(obj)
def LMUDataRoot():
    root=os.environ['LMUData']; Path(root).mkdir(parents=True,exist_ok=True); return root
def get_file_extension(path): return str(path).split('.')[-1]
def get_intermediate_file_path(path, middle, fmt=None):
    p=Path(path); return str(p.with_name(p.stem+'_'+middle+'.'+(fmt or p.suffix[1:])))
def load(path):
    ext=Path(path).suffix
    if ext=='.json': return json.loads(Path(path).read_text())
    if ext=='.jsonl': return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if ext in ('.pkl','.pickle'):
        with open(path,'rb') as f: return pickle.load(f)
    if ext in ('.xlsx','.xls'): return pd.read_excel(path)
    return pd.read_csv(path,sep='\t' if ext=='.tsv' else ',')
def dump(data,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    ext=Path(path).suffix
    if ext=='.json': Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,cls=NumpyEncoder)); return
    if ext in ('.pkl','.pickle'):
        with open(path,'wb') as f: pickle.dump(data,f)
        return
    if ext in ('.xlsx','.xls'): data.to_excel(path,index=False); return
    data.to_csv(path,sep='\t' if ext=='.tsv' else ',',index=False)
