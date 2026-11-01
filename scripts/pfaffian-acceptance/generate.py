#!/usr/bin/env python3
"""Common binary matrices for the external faer#171 comparison, no timed generation."""
from pathlib import Path
import array, hashlib, json, math, sys
if len(sys.argv) != 2:
    raise SystemExit('usage: generate.py OUTPUT')
root=Path(sys.argv[1]);root.mkdir(parents=True, exist_ok=True)
mask=(1<<64)-1
records=[]
def sample(i,j,seed):
    x=(i*1000003+j+seed*0x9e3779b97f4a7c15)&mask
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
    x=((x^(x>>27))*0x94d049bb133111eb)&mask
    x^=x>>31
    return ((x>>11)/(1<<53)-.5)*4
for n in [2,64,128,256,512,1024]:
    for dtype in ['f64','c64']:
        data=array.array('d')
        for j in range(n):
            for i in range(n):
                if n==2:
                    v=0 if i==j else (2.5 if i<j else -2.5)
                    im=0 if i==j else (1.25 if i<j else -1.25)
                elif i==j:
                    v=im=0.
                else:
                    p,q=max(i,j),min(i,j);sign=1 if i>j else -1
                    v=sign*sample(p,q,171)/math.sqrt(n)
                    im=sign*sample(p,q,242)/math.sqrt(n)
                data.append(v)
                if dtype=='c64':data.append(im)
        if sys.byteorder!='little':data.byteswap()
        path=root/f'n{n}-{dtype}.bin';path.write_bytes(data.tobytes())
        if n==2:
            assert data[2 if dtype=='f64' else 4]==2.5
        records.append({'n':n,'dtype':dtype,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'manifest.json').write_text(json.dumps({'seed':171,'distribution':'uniform[-2,2]/sqrt(n), independent complex components; testn2 upper2.5+1.25i','layout':'little-endian column-major, complex interleaved','matrices':records},indent=2)+'\n')
print('Generated',len(records),'exact common matrices and hash manifest')
