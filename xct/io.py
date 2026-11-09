"""Global field metadata in PLY comments; original per-point columns stay unchanged."""
import json
import os
from pathlib import Path
from plyfile import PlyData
from xct.field import configure_field


def field_metadata(path):
    with open(path,'rb') as f:
        for line in f:
            if line.strip()==b'end_header':break
            if line.startswith(b'comment xct_field '):
                return json.loads(line[len(b'comment xct_field '):])
    return {}


def save_ct_ply(pc,path):
    path=Path(path)
    pc.save_ply(str(path))
    ply=PlyData.read(path)
    metadata=dict(version=3,backend=getattr(pc,'ct_backend','standard'),activation=getattr(pc,'ct_activation','relu'),beta=getattr(pc,'ct_beta',1.),sh_degree=pc.max_sh_degree)
    if getattr(pc, 'volume_geometry', None) is not None:metadata['volume_geometry']=pc.volume_geometry
    ply.comments=list(ply.comments)+['xct_field '+json.dumps(metadata,separators=(',',':'))]
    temporary=path.with_suffix('.with_metadata.ply')
    ply.write(temporary)
    os.replace(temporary,path)


def configure_from_ply(pc,path,fallback=None):
    meta=field_metadata(path) or (fallback or {})
    if meta.get('volume_geometry') is not None:pc.volume_geometry=meta['volume_geometry']
    return configure_field(pc,meta.get('activation','relu'),meta.get('beta',1.),meta.get('backend','standard'))
