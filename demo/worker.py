"""X-Gaussian inference worker for built-in and uploaded radiative Gaussian PLYs."""
import argparse, contextlib, copy, gc, gzip, hashlib, json, math, os, sys, time, traceback, uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from assets import ROOT, REPO, CACHE, catalog
sys.path.insert(0,str(REPO))
from render import load_model
from scene.gaussian_model import GaussianModel_Xray
from xct.field import query, render, attenuation
from xct.io import field_metadata, configure_from_ply
from utils.graphics_utils import getProjectionMatrix
CATALOG=catalog()
IDENTITIES=json.loads((ROOT/'checkpoint-identities.json').read_text())
SPECS={s['scene']:s for s in json.loads((ROOT/'static/assets/scene-manifest.json').read_text())['scenes']}

class Engine:
    def __init__(self,device,shared_gpu=False):
        if not shared_gpu:torch.cuda.set_device(device)
        torch.set_num_threads(2)
        self.models={}
        self.scene=None;self.pc=None;self.catalog=dict(CATALOG)

    def model(self,scene):
        if scene not in self.catalog:raise ValueError('Unknown scene; choose an example or load a PLY first.')
        loaded=self.scene!=scene
        if loaded:
            self.pc=None;self.scene=None;gc.collect();torch.cuda.empty_cache()
            item=self.catalog[scene]
            if item.get('uploaded'):
                self.pc=GaussianModel_Xray(item['sh_degree'])
                self.pc.load_ply(item['checkpoint'])
                configure_from_ply(self.pc,item['checkpoint'],dict(volume_geometry=item['geometry']))
                self.config={}
            else:
                if scene in self.models:self.pc,self.config=self.models[scene]
                else:self.pc,self.config=load_model(item['checkpoint'])
            self.scene=scene
        return self.pc,loaded

    def camera(self,scene,azimuth,elevation,align_display=True):
        item=self.catalog[scene];cfg=item['scanner']
        a,e=np.deg2rad([float(azimuth),float(elevation)])
        if not np.isfinite([a,e]).all() or abs(e)>np.deg2rad(85):raise ValueError('Invalid camera angles')
        radial=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
        right=np.array([-np.sin(a),np.cos(a),0.]);forward=-radial
        if align_display:
            up=np.asarray(item['rotation']).T @ np.array([0.,1.,0.])
            candidate=np.cross(forward,up)
            if np.linalg.norm(candidate)>1e-6:right=candidate/np.linalg.norm(candidate)
        down=np.cross(forward,right)
        pose=np.eye(4);pose[:3,:3]=np.stack([right,down,forward],axis=1)
        pose[:3,3]=np.asarray(cfg['offOrigin'])+float(cfg['DSO'])*radial
        view=torch.tensor(np.linalg.inv(pose),dtype=torch.float32,device='cuda').T.contiguous()
        fx=2*np.arctan2(cfg['sDetector'][1]/2,cfg['DSD']);fy=2*np.arctan2(cfg['sDetector'][0]/2,cfg['DSD'])
        projection=getProjectionMatrix(.01,100.,fx,fy).T.cuda()
        return SimpleNamespace(image_height=int(cfg['nDetector'][0]),image_width=int(cfg['nDetector'][1]),
            mode=1,FoVx=fx,FoVy=fy,world_view_transform=view,full_proj_transform=view@projection,
            camera_center=torch.linalg.inv(view)[3,:3])

    def volume_file(self,scene):
        pc,_=self.model(scene);item=self.catalog[scene];g=item['geometry']
        source=Path(item['checkpoint']).stat()
        directory=CACHE/'volumes';directory.mkdir(exist_ok=True)
        path=directory/f'{scene}-{source.st_size}-{source.st_mtime_ns}.npy'
        cached=path.exists() and path.with_suffix('.json').exists()
        if not cached:
            torch.cuda.synchronize();start=time.perf_counter()
            full=query(pc,g['offOrigin'],g['nVoxel'],g['sVoxel'])['vol']
            if not torch.isfinite(full).all():raise FloatingPointError('The queried CT field contains nonfinite values.')
            torch.cuda.synchronize();query_ms=(time.perf_counter()-start)*1000
            temporary=path.with_name(path.stem+'.partial.npy');np.save(temporary,full.cpu().numpy());os.replace(temporary,path)
            del full
            path.with_suffix('.json').write_text(json.dumps(dict(query_ms=query_ms,source=item['checkpoint'],geometry=g)))
        else:query_ms=json.loads(path.with_suffix('.json').read_text())['query_ms']
        return path,query_ms,cached

    @torch.inference_mode()
    def register(self,path,label):
        path=Path(path).resolve()
        if not path.is_relative_to((CACHE/'uploads').resolve()) or path.suffix.lower()!='.ply':
            raise ValueError('Please upload an X-Gaussian .ply checkpoint.')
        if not 0<path.stat().st_size<=512*1024**2:raise ValueError('The PLY must be smaller than 512 MB.')
        try:ply=PlyData.read(path)
        except Exception as exc:raise ValueError('This file is not a readable PLY point cloud.') from exc
        if not ply.elements or ply.elements[0].name!='vertex':raise ValueError('The PLY needs a Gaussian vertex element.')
        v=ply.elements[0];names=set(v.data.dtype.names)
        required=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
        missing=[n for n in required if n not in names]
        if missing:raise ValueError('This is not an X-Gaussian checkpoint. Missing Gaussian attributes: '+', '.join(missing[:5])+'.')
        if not 0<len(v)<=2000000:raise ValueError('This demo supports PLYs with 1 to 2,000,000 Gaussians.')
        rest=[n for n in names if n.startswith('f_rest_')]
        degree=int(round(math.sqrt(1+len(rest)/3)))-1
        if degree not in range(4) or len(rest)!=3*((degree+1)**2-1):raise ValueError('Unsupported radiative feature layout.')
        for name in required+rest:
            if not np.isfinite(v[name]).all():raise ValueError(f'The Gaussian attribute {name} contains nonfinite values.')
        scales=np.stack([v[f'scale_{i}'] for i in range(3)],1)
        rotations=np.stack([v[f'rot_{i}'] for i in range(4)],1)
        if np.any(scales>20) or np.any(scales<-35) or np.any(np.linalg.norm(rotations,axis=1)<1e-10):
            raise ValueError('The PLY contains invalid Gaussian scales or rotations.')
        meta=field_metadata(path)
        if meta.get('sh_degree',degree)!=degree:raise ValueError('PLY feature degree and metadata disagree.')
        digest=hashlib.sha256(path.read_bytes()).hexdigest();scene='upload-'+digest[:20]
        preset=IDENTITIES.get(digest)
        geometry=meta.get('volume_geometry') or None
        inferred=not bool(geometry)
        if geometry is None:
            xyz=np.stack([v[k] for k in ['x','y','z']],1)
            low=xyz.min(0);high=xyz.max(0)
            padding=max(float((high-low).max())*.025,float(np.exp(np.median(scales)))*2,1e-4)
            extent=np.maximum(high-low+2*padding,1e-4)
            geometry=dict(nVoxel=np.maximum(16,np.rint(extent/extent.max()*256).astype(int)).tolist(),
                          sVoxel=extent.tolist(),offOrigin=((low+high)/2).tolist())
        try:
            count=np.asarray(geometry['nVoxel'],dtype=float);size=np.asarray(geometry['sVoxel'],dtype=float);center=np.asarray(geometry['offOrigin'],dtype=float)
        except Exception as exc:raise ValueError('Invalid CT grid metadata in the PLY.') from exc
        if any(a.shape!=(3,) for a in [count,size,center]) or not np.isfinite(np.concatenate([count,size,center])).all():
            raise ValueError('Invalid CT grid metadata in the PLY.')
        if np.any(count<2) or np.any(count!=count.astype(int)) or np.prod(count)>512**3 or np.any(size<=0):
            raise ValueError('The CT grid must be positive and no larger than 512³ voxels.')
        geometry=dict(nVoxel=count.astype(int).tolist(),sVoxel=size.tolist(),offOrigin=center.tolist())
        if preset:item=copy.deepcopy(CATALOG[preset])
        else:
            radius=float(np.linalg.norm(size)/2);dso=4*radius;dsd=2*dso
            detector=float(size.max()*2.4)
            item=dict(scanner=dict(nDetector=[512,512],nVoxel=geometry['nVoxel'],mode='cone',DSO=dso,DSD=dsd,
                      offOrigin=center.tolist(),offDetector=[0.,0.],sVoxel=size.tolist(),sDetector=[detector,detector],
                      dVoxel=(size/count).tolist()),rotation=[[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]],
                      default_azimuth=45.,default_elevation=15.,exposure=6.,iteration=None)
        item.update(name=label,group='uploaded',checkpoint=str(path),model=str(path.parent),geometry=geometry,
                    uploaded=True,sh_degree=degree,inferred_geometry=inferred,gaussians=len(v),preset=preset)
        self.catalog[scene]=item
        if preset:
            spec=copy.deepcopy(SPECS[preset]);spec.update(scene=scene,name=label)
            files={}
        else:
            self.model(scene)
            directory=path.parent;spec_file=directory/'viewer.json';field=directory/f'{scene}.bin.gz'
            if spec_file.exists() and field.exists():spec=json.loads(spec_file.read_text())
            else:
                volume_path,_,_=self.volume_file(scene)
                resolution=geometry['nVoxel']
                factor=max(resolution)/256
                if factor>1:
                    resolution=[max(2,int(round(n/factor))) for n in resolution]
                    volume=query(self.pc,geometry['offOrigin'],resolution,geometry['sVoxel'])['vol'].cpu().numpy()
                else:volume=np.load(volume_path,mmap_mode='r')
                maximum=float(volume.max())
                if maximum<=0:raise ValueError('The PLY has no positive radiative attenuation field.')
                quantized=np.rint(np.clip(volume/maximum,0,1)*4095).astype(np.uint16).transpose(2,1,0)
                packed=np.empty((*quantized.shape,2),np.uint8);packed[...,0]=quantized>>8;packed[...,1]=quantized&255
                with gzip.open(field,'wb',compresslevel=4) as stream:stream.write(packed.tobytes())
                direction=np.array([np.cos(np.deg2rad(15))*np.cos(np.deg2rad(45)),np.cos(np.deg2rad(15))*np.sin(np.deg2rad(45)),np.sin(np.deg2rad(15))])
                displayed=np.asarray(item['rotation'])@direction
                spec=dict(scene=scene,name=label,dimensions=resolution,source_native_dimensions=geometry['nVoxel'],
                          size=geometry['sVoxel'],scale=maximum,rotation=item['rotation'],target=[0,0,0],
                          threshold=maximum*.12,exposure=6.,yaw=float(np.rad2deg(np.arctan2(displayed[0],displayed[2]))),
                          pitch=float(np.rad2deg(np.arccos(displayed[1]))),field_path=field.name,bytes=field.stat().st_size)
                spec_file.write_text(json.dumps(spec))
            files={field.name:str(field)}
        return dict(scene=scene,item=item,spec=spec,files=files)


    def slice_cached(self,scene,req,volume_path,query_ms=0,cached=True):
        item=self.catalog[scene]
        axis=int(req.get('axis',2));fraction=float(req.get('position',50))/100
        if axis not in [0,1,2] or not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Invalid CT slice')
        geometry=item['geometry'];native=np.array(geometry['nVoxel']);size=np.array(geometry['sVoxel']);center=np.array(geometry['offOrigin'],dtype=float)
        index=int(round(fraction*(native[axis]-1)));center[axis]+=((index+.5)/native[axis]-.5)*size[axis]
        native_volume=np.load(volume_path,mmap_mode='r')
        raw=np.take(native_volume,index,axis=axis)
        if not np.isfinite(raw).all():raise FloatingPointError('Nonfinite CT slice')
        remaining=[i for i in range(3) if i!=axis];physical=size[remaining]
        maximum=float(np.percentile(raw,99.5));display=np.uint8(np.clip(raw/max(maximum,1e-12),0,1)*255)
        image=Image.fromarray(np.flipud(display.T).copy());scale=600/max(physical)
        image=image.resize(tuple(max(1,int(round(v*scale))) for v in physical),Image.Resampling.BILINEAR)
        path=CACHE/f'{scene}-{uuid.uuid4().hex[:12]}-slice.png';image.save(path)
        result=dict(image=str(path),shape=list(raw.shape),axis=axis,index=index,total=int(native[axis]),coordinate=float(center[axis]),
                    compute_ms=query_ms,backend='CUDA RGVQ',cached=cached,volume=str(volume_path),position=100*index/max(1,native[axis]-1))

        return result

    @torch.inference_mode()
    def handle(self,req):
        if req['operation']=='register':return self.register(req['path'],req.get('label','Uploaded point cloud'))
        if req['operation']=='volume':
            path,ms,cached=self.volume_file(req['scene']);return dict(volume=str(path),compute_ms=ms,cached=cached)
        started=time.perf_counter();scene=req['scene'];pc,loaded=self.model(scene);item=self.catalog[scene]
        torch.cuda.synchronize();operation=req['operation'];op_start=time.perf_counter()
        if operation=='points':
            source=Path(item['checkpoint']).stat()
            directory=CACHE/'points';directory.mkdir(exist_ok=True)
            path=directory/f'{scene}-{source.st_size}-{source.st_mtime_ns}.bin'
            metadata=path.with_suffix('.json')
            if not path.exists() or not metadata.exists():
                xyz=pc.get_xyz.detach().cpu().numpy()
                density=attenuation(pc).squeeze(1).detach().cpu().numpy()
                if not np.isfinite(xyz).all() or not np.isfinite(density).all():raise ValueError('Nonfinite Gaussian points')
                reference=max(float(np.percentile(density,99)),1e-12)
                strength=(np.log1p(np.clip(density/reference,0,1)*4)/np.log(5)).astype(np.float32)
                packed=np.column_stack([xyz,strength]).astype('<f4')
                packed.tofile(path)
                metadata.write_text(json.dumps(dict(count=len(xyz),bytes=path.stat().st_size,
                    encoding='float32 little-endian: x,y,z,display_strength',geometry=item['geometry'])))
            result=json.loads(metadata.read_text());result.update(file=str(path),backend='Gaussian point centers')
        elif operation=='mesh':
            from xct.mesh import export_mesh,mask_resolution
            volume_path,_,_=self.volume_file(scene)
            source=Path(item['checkpoint']).stat();g=item['geometry']
            directory=CACHE/'meshes';directory.mkdir(exist_ok=True)
            path=directory/f'{scene}-{source.st_size}-{source.st_mtime_ns}.glb';metadata=path.with_suffix('.json')
            if not path.exists() or not metadata.exists():
                name=item.get('preset') if item.get('uploaded') else scene
                roi=None
                if name in ['abdomen','pancreas']:
                    resolution=mask_resolution(g['nVoxel'])
                    if resolution!=g['nVoxel']:roi=query(pc,g['offOrigin'],resolution,g['sVoxel'])['vol'].cpu().numpy()
                info=export_mesh(np.load(volume_path,mmap_mode='r'),path,g['offOrigin'],g['sVoxel'],
                                 scene=name,roi_volume=roi,reference_geometry=g)
                metadata.write_text(json.dumps(info))
            result=json.loads(metadata.read_text());result.update(file=str(path),backend='Native CT marching cubes')
        elif operation=='render':
            camera=self.camera(scene,req['azimuth'],req['elevation'])
            raw=render(camera,pc)['render'].squeeze(0);torch.cuda.synchronize();compute_ms=(time.perf_counter()-op_start)*1000
            raw=raw.cpu().numpy()
            if not np.isfinite(raw).all():raise FloatingPointError('Nonfinite X-ray projection')
            exposure=float(req.get('exposure',4.))
            if not .1<=exposure<=30:raise ValueError('Invalid exposure')
            normalized=np.maximum(raw,0)/max(float(raw.max()),1e-12)
            display=np.power(-np.expm1(-normalized*exposure*.18)/max(-np.expm1(-exposure*.18),1e-12),1.35)
            if req.get('invert',False):display=1-display
            npy=CACHE/f'{scene}-{uuid.uuid4().hex[:12]}-projection.npy';png=npy.with_suffix('.png')
            np.save(npy,raw);Image.fromarray(np.uint8(np.clip(display*255,0,255))).save(png)
            result=dict(image=str(png),download=str(npy),shape=list(raw.shape),minimum=float(raw.min()),maximum=float(raw.max()),
                azimuth=float(req['azimuth']),elevation=float(req['elevation']),compute_ms=compute_ms,backend='CUDA Gaussian rasterization')
        elif operation=='slice':
            volume_path,query_ms,cached=self.volume_file(scene)
            result=self.slice_cached(scene,req,volume_path,query_ms,cached)
        else:raise ValueError('Unknown operation')
        result.update(scene=scene,gaussians=int(pc.get_xyz.shape[0]),loaded_model=loaded,total_ms=(time.perf_counter()-started)*1000)
        return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',type=int,default=0);args=parser.parse_args();engine=Engine(args.device)
    print(json.dumps(dict(ready=True,device=torch.cuda.get_device_name())),flush=True)
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):result=engine.handle(json.loads(line))
            print(json.dumps(dict(ok=True,result=result)),flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr);print(json.dumps(dict(ok=False,error=f'{type(exc).__name__}: {exc}')),flush=True)
