"""Local X-Gaussian Gradio demo with visual scene selection and PLY uploads."""
import argparse,atexit,copy,hashlib,html,json,os,shutil,subprocess,threading,sys,struct
from pathlib import Path
from assets import ROOT, REPO, CACHE, catalog, field_file, download
os.environ['GRADIO_ANALYTICS_ENABLED']='False';os.environ['GRADIO_TEMP_DIR']=str(CACHE/'gradio')
import gradio as gr
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
CATALOG=catalog()
SPECS={s['scene']:s for s in json.loads((ROOT/'static/assets/scene-manifest.json').read_text())['scenes']}
AXES={'Axial (Z)':2,'Coronal (Y)':1,'Sagittal (X)':0};ORDER=['chest','abdomen','foot','head','pancreas','pine','seashell','walnut']
POINTS={}
FIELDS={name:Path(path) for item in CATALOG.values() for name,path in item['fields'].items()}
MESHES={}
THUMBNAILS=[(str(ROOT/'static/assets/thumbnails'/f'{scene}.webp'),CATALOG[scene]['name']) for scene in ORDER]

class Worker:
    def __init__(self,python,device):
        self.lock=threading.Lock();self.log=open(CACHE/'worker.log','a',buffering=1)
        self.process=subprocess.Popen([python,'-u',str(ROOT/'worker.py'),'--device',str(device)],cwd=REPO,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        first=self.process.stdout.readline()
        if not first:raise RuntimeError('CUDA worker did not start; see worker.log')
        self.info=json.loads(first);print('CUDA worker ready:',self.info,flush=True);atexit.register(self.close)
    def call(self,operation,scene=None,**params):
        with self.lock:
            if self.process.poll() is not None:raise gr.Error('The CUDA worker stopped. See worker.log.')
            self.process.stdin.write(json.dumps(dict(operation=operation,scene=scene,**params))+'\n');self.process.stdin.flush()
            line=self.process.stdout.readline()
            if not line:raise gr.Error('The CUDA worker stopped. See worker.log.')
            result=json.loads(line)
            if not result.get('ok'):raise gr.Error(result.get('error','Unable to load this point cloud.'))
            return result['result']
    def available(self):return self.process.poll() is None
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill()
        self.log.close()

HEADER='''<div class="app-header"><svg class="brand-icon" viewBox="0 0 40 40" fill="none" aria-hidden="true"><path d="M9 7L31 33M31 7L9 33" stroke="currentColor" stroke-width="2.3"/><ellipse cx="20" cy="20" rx="17" ry="7" transform="rotate(-30 20 20)" stroke="currentColor" opacity=".45"/><circle cx="20" cy="20" r="4" fill="currentColor"/></svg><div><h1>X-Gaussian · ECCV 2024</h1><p>Interactive X-ray &amp; CT exploration</p></div></div>'''
HEAD='''<script>
function setGradioSlider(label,value){const input=document.querySelector('input[aria-label="range slider for '+label+'"]');if(!input)return;Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,value);input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));}
function currentSliceImage(){return document.querySelector('#slice-panel img')?.src;}
window.addEventListener('message',e=>{
 if(e.origin!==location.origin||e.source!==document.querySelector('#volume-frame')?.contentWindow)return;
 const c=window.__xgaussianControls;
 if(e.data?.kind==='xgaussian-ready'&&c&&c.scene===e.data.scene)e.source.postMessage({...c,cameraChanged:true,sliceImage:currentSliceImage()},location.origin);
 if(e.data?.kind==='xgaussian-slice-drag'&&c&&c.scene===e.data.scene&&Number.isFinite(e.data.position))setGradioSlider('Slice position (%)',Math.max(0,Math.min(100,e.data.position)));
});
let controlSizeObserver;
function fitControlPanel(){
 const a=document.querySelector('#top-row'),b=document.querySelector('#bottom-row'),c=document.querySelector('#controls'),first=document.querySelector('#volume-panel'),last=document.querySelector('#mesh-panel'),layout=document.querySelector('#main-layout');if(!a||!b||!c||!first||!last||!layout)return;
 const fit=()=>{if(innerWidth<=760){c.style.removeProperty('height');c.style.removeProperty('max-height');c.style.removeProperty('margin-top');return;}const height=last.getBoundingClientRect().bottom-first.getBoundingClientRect().top;const margin=first.getBoundingClientRect().top-layout.getBoundingClientRect().top;c.style.setProperty('margin-top',margin+'px','important');const value=height+'px';if(c.style.height!==value){c.style.setProperty('height',value,'important');c.style.setProperty('max-height',value,'important');}};
 if(!controlSizeObserver){controlSizeObserver=new ResizeObserver(fit);controlSizeObserver.observe(a);controlSizeObserver.observe(b);window.addEventListener('resize',fit);}fit();
}
new MutationObserver(fitControlPanel).observe(document.documentElement,{childList:true,subtree:true});
let lastSliceImage='';new MutationObserver(()=>{const src=currentSliceImage(),c=window.__xgaussianControls;if(src&&c&&src!==lastSliceImage){lastSliceImage=src;document.querySelector('#volume-frame')?.contentWindow?.postMessage({...c,cameraChanged:false,sliceImage:src},location.origin);}}).observe(document.documentElement,{childList:true,subtree:true,attributes:true,attributeFilter:['src']});
</script>'''
PREVIEW_JS='''(scene,azimuth,elevation,exposure,invert,axis,position,showPlane)=>{
 const frame=document.querySelector('#volume-frame');scene=frame?new URL(frame.src,location.href).searchParams.get('scene'):scene;
 const previous=window.__xgaussianControls;
 const c={kind:'xgaussian-controls',scene,azimuth,elevation,exposure,invert,axis:showPlane?({'Axial (Z)':2,'Coronal (Y)':1,'Sagittal (X)':0}[axis]):-1,position,sliceImage:document.querySelector('#slice-panel img')?.src};
 c.cameraChanged=!previous||previous.scene!==scene||Math.abs(previous.azimuth-azimuth)>.001||Math.abs(previous.elevation-elevation)>.001;
 window.__xgaussianControls=c;document.querySelector('#volume-frame')?.contentWindow?.postMessage(c,location.origin);return [];
}'''

def iframe(scene,view_mode='Point cloud',axis='Coronal (Y)',position=50):
    name=html.escape(CATALOG[scene]['name'],quote=True);display='points' if view_mode=='Point cloud' else 'xray';version=POINTS.get(scene,{}).get('count',0)
    return f'<iframe class="volume-frame" id="volume-frame" src="/viewer/index.html?scene={scene}&display={display}&plane_axis={AXES[axis]}&slice_position={position}&v={version}" title="{name} interactive Gaussian model"></iframe>'

def cached_mesh(scene):
    item=CATALOG[scene];source=Path(item['checkpoint']).stat();path=CACHE/'meshes'/f'{scene}-{source.st_size}-{source.st_mtime_ns}.glb'
    if path.exists() and path.with_suffix('.json').exists():
        info=json.loads(path.with_suffix('.json').read_text());info['file']=str(path);MESHES[scene]=info;return info
    if not item.get('uploaded'):
        if scene in MESHES:return MESHES[scene]
        relative=SPECS[scene]['mesh_path'];preview=REPO/'data'/relative
        if not preview.is_file():preview=download(relative,dataset=True)
        with preview.open('rb') as stream:
            header=stream.read(20);length=struct.unpack_from('<I',header,12)[0];document=json.loads(stream.read(length))
        primitives=[p for m in document['meshes'] for p in m['primitives']]
        info=dict(file=str(preview),vertices=sum(document['accessors'][p['attributes']['POSITION']]['count'] for p in primitives),
                  faces=sum(document['accessors'][p['indices']]['count']//3 for p in primitives))
        MESHES[scene]=info;return info
    return None

def mesh_iframe(scene):
    info=cached_mesh(scene);version=Path(info['file']).stat().st_mtime_ns if info else 'empty'
    return f'<iframe class="volume-frame" id="mesh-frame" src="/mesh-viewer/index.html?scene={scene}&v={version}" title="Interactive exported CT mesh"></iframe>'

def scene_info(scene,gaussians):
    item=CATALOG[scene];grid=' × '.join(map(str,item['geometry']['nVoxel']));extra=' · estimated grid' if item.get('inferred_geometry') else ''
    return f'<div class="scene-info-line"><span class="scene-name">{html.escape(item["name"])}</span><span class="scene-detail">{gaussians:,} Gaussians · {grid} voxels{extra}</span></div>'

def render_meta(result):
    shape=' × '.join(map(str,result['shape']))
    return f'<div class="render-meta"><span><b>{shape}</b> pixels</span><span><b>{result["compute_ms"]:.1f} ms</b> CUDA</span><span>{result["azimuth"]:.1f}° / {result["elevation"]:.1f}°</span></div>'

def build_app(worker):
    def render_projection(scene,azimuth,elevation,exposure,invert):
        result=worker.call('render',scene,azimuth=azimuth,elevation=elevation,exposure=exposure,invert=invert)
        return result['image'],result['download'],render_meta(result)
    def render_current(scene,azimuth,elevation,exposure,invert):return (*render_projection(scene,azimuth,elevation,exposure,invert),azimuth,elevation)
    def get_slice(scene,axis,position):
        result=worker.call('slice',scene,axis=AXES[axis],position=position)
        caption=f'<div class="slice-caption">{axis} · slice {result["index"]+1} / {result["total"]}</div>'
        return result['image'],caption,result['volume']
    def load_scene(scene,axis,position,invert,view_mode):
        item=CATALOG[scene];a,e,x=item['default_azimuth'],item['default_elevation'],item['exposure']
        result=worker.call('render',scene,azimuth=a,elevation=e,exposure=x,invert=invert);POINTS[scene]=worker.call('points',scene)
        sliced,caption,volume=get_slice(scene,axis,position);mesh=cached_mesh(scene)
        return (scene,iframe(scene,view_mode,axis,position),a,e,x,result['image'],result['download'],render_meta(result),sliced,caption,
            scene_info(scene,result['gaussians']),volume,gr.update(value=mesh['file'] if mesh else None,visible=bool(mesh)),
            f'{mesh["vertices"]:,} vertices · {mesh["faces"]:,} faces' if mesh else '',mesh_iframe(scene))
    def choose_example(axis,position,invert,view_mode,evt:gr.SelectData):
        index=evt.index[0] if isinstance(evt.index,(list,tuple)) else evt.index
        return load_scene(ORDER[index],axis,position,invert,view_mode)
    def upload_model_impl(path,axis,position,invert,view_mode):
        if not path:raise gr.Error('Choose an X-Gaussian .ply checkpoint.')
        source=Path(path)
        if source.suffix.lower()!='.ply':raise gr.Error('Please upload a .ply checkpoint.')
        if source.stat().st_size>512*1024**2:raise gr.Error('The PLY must be smaller than 512 MB.')
        with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        destination=CACHE/'uploads'/digest/'point_cloud.ply';destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copyfile(source,destination)
        registered=worker.call('register',path=str(destination),label=source.name);scene=registered['scene']
        CATALOG[scene]=registered['item'];SPECS[scene]=registered['spec'];FIELDS.update({k:Path(v) for k,v in registered['files'].items()})
        return (*load_scene(scene,axis,position,invert,view_mode),gr.update(selected_index=None))
    def upload_model(path,axis,position,invert,view_mode):
        try:return upload_model_impl(path,axis,position,invert,view_mode)
        except gr.Error as exc:
            gr.Warning(getattr(exc,'message',str(exc)),duration=6);return tuple(gr.skip() for _ in range(16))
    def export_surface(scene):
        yield gr.update(value='Exporting…',interactive=False),gr.skip(),'Extracting the CT surface…',gr.skip()
        try:
            result=worker.call('mesh',scene);MESHES[scene]=result
            yield gr.update(value='Export Mesh',interactive=True),gr.update(value=result['file'],visible=True),f'{result["vertices"]:,} vertices · {result["faces"]:,} faces',mesh_iframe(scene)
        except gr.Error as exc:
            gr.Warning(getattr(exc,'message',str(exc)),duration=6)
            yield gr.update(value='Export Mesh',interactive=True),gr.skip(),'Mesh export failed.',gr.skip()
    theme=gr.themes.Base(primary_hue='slate',neutral_hue='slate',font=['Arial','Helvetica','sans-serif']).set(
        body_background_fill='#ffffff',body_background_fill_dark='#ffffff',body_text_color='#171717',body_text_color_dark='#171717',
        block_background_fill='#ffffff',block_background_fill_dark='#ffffff',block_border_color='#dce0e4',block_border_color_dark='#dce0e4',
        input_background_fill='#ffffff',input_background_fill_dark='#ffffff',block_label_background_fill='#ffffff',block_label_background_fill_dark='#ffffff',
        block_label_text_color='#515963',block_label_text_color_dark='#515963')
    default=CATALOG['head']
    with gr.Blocks(title='X-Gaussian · ECCV 2024',theme=theme,css=(ROOT/'static/assets/app.css').read_text(),head=HEAD,
                   js="()=>{document.documentElement.classList.remove('dark');document.body.classList.remove('dark');}") as demo:
        scene=gr.State('head');gr.HTML(HEADER,padding=False)
        with gr.Row(equal_height=True,elem_id='scene-selector-row'):
            with gr.Column(scale=0,min_width=118,elem_id='scene-toolbar'):
                gr.HTML('<div class="scene-heading">Choose a scene</div>',padding=False)
                upload=gr.UploadButton('Upload PLY',file_types=['.ply'],type='filepath',file_count='single',elem_id='upload-button')
            with gr.Column(scale=1,min_width=0,elem_id='scene-strip'):
                gallery=gr.Gallery(THUMBNAILS,columns=8,rows=1,height=76,allow_preview=False,selected_index=3,object_fit='contain',
                    show_label=False,container=False,interactive=False,show_download_button=False,show_fullscreen_button=False,elem_id='scene-gallery')
        with gr.Row(equal_height=True,elem_id='main-layout'):

            with gr.Column(scale=4,min_width=500,elem_id='workspace'):
                with gr.Row(equal_height=False,elem_id='top-row'):
                    with gr.Column(scale=1,min_width=180,elem_id='model-card'):
                        gr.HTML('<div class="panel-heading">Gaussian Model <span>Point cloud</span></div>',padding=False)
                        viewport=gr.HTML(iframe('head'),elem_id='volume-panel',padding=False)
                    with gr.Column(scale=1,min_width=180,elem_id='ct-card'):
                        gr.HTML('<div class="panel-heading">CT Slice <span>Linked to model</span></div>',padding=False)
                        slice_image=gr.Image(label='CT slice',show_label=False,type='filepath',interactive=False,elem_id='slice-panel',show_download_button=True,show_fullscreen_button=True)
                        gr.Markdown('Drag the cyan handle in GS, or adjust Slice position on the right.',elem_classes='helper')
                with gr.Row(equal_height=False,elem_id='bottom-row'):
                    with gr.Column(scale=1,min_width=180,elem_id='projection-card'):
                        gr.HTML('<div class="panel-heading">X-ray Projection <span>CUDA</span></div>',padding=False)
                        projection=gr.Image(label='Rendered projection',show_label=False,type='filepath',interactive=False,elem_id='projection-panel',show_download_button=True,show_fullscreen_button=True)
                        stats=gr.HTML('<div class="render-meta">Loading…</div>',padding=False,elem_id='render-meta')
                    with gr.Column(scale=1,min_width=180,elem_id='mesh-card'):
                        gr.HTML('<div class="panel-heading">Exported Mesh <span>Interactive</span></div>',padding=False)
                        mesh_preview=gr.HTML(mesh_iframe('head'),elem_id='mesh-panel',padding=False)
                        mesh_status=gr.Markdown('',elem_classes='helper',elem_id='mesh-status')
            with gr.Column(scale=1,min_width=225,elem_id='controls'):
                info=gr.HTML('<div class="scene-info-line">Loading Gaussian model…</div>',elem_id='scene-info',padding=False)
                gr.HTML('<div class="control-heading">CT slice</div>',padding=False)
                axis=gr.Radio(list(AXES),value='Coronal (Y)',show_label=False,label='CT slice plane',elem_id='slice-axis')
                position=gr.Slider(0,100,value=50,step=.1,label='Slice position (%)')
                show_plane=gr.Checkbox(value=True,label='Show slice plane')
                slice_caption=gr.HTML('<div class="slice-caption">Querying CT volume…</div>',padding=False,elem_id='slice-caption')
                gr.HTML('<div class="control-heading">Projection</div>',padding=False)
                render_button=gr.Button('Render X-ray projection',variant='primary',elem_id='render-button')
                gr.HTML('<div class="control-heading">Mesh</div>',padding=False)
                mesh_button=gr.Button('Export Mesh',elem_id='export-mesh')
                mesh_download=gr.DownloadButton('Download mesh (.glb)',visible=False,elem_id='download-mesh')
                with gr.Accordion('View settings',open=False,elem_id='view-settings'):
                    view_mode=gr.Radio(['Point cloud','X-ray'],value='Point cloud',label='Model view',elem_id='model-view')
                    azimuth=gr.Slider(0,360,value=default['default_azimuth'],step=.1,label='Azimuth (°)')
                    elevation=gr.Slider(-80,80,value=default['default_elevation'],step=.1,label='Elevation (°)')
                    exposure=gr.Slider(.3,25,value=default['exposure'],step=.1,label='Display exposure')
                    invert=gr.Checkbox(value=False,label='Invert X-ray appearance')
                with gr.Accordion('Downloads',open=False,elem_id='downloads'):
                    download=gr.DownloadButton('Projection (.npy)',elem_id='download-projection')
                    volume_download=gr.DownloadButton('CT volume (.npy)',elem_id='download-volume')
        output=[scene,viewport,azimuth,elevation,exposure,projection,download,stats,slice_image,slice_caption,info,volume_download,mesh_download,mesh_status,mesh_preview]
        preview_inputs=[scene,azimuth,elevation,exposure,invert,axis,position,show_plane]
        for component in [azimuth,elevation,exposure,invert,axis,position,show_plane]:component.input(fn=None,inputs=preview_inputs,outputs=[],js=PREVIEW_JS,queue=False)
        for event in [gallery.select(choose_example,[axis,position,invert,view_mode],output,concurrency_id='cuda',concurrency_limit=1,show_progress='minimal'),
                      upload.upload(upload_model,[upload,axis,position,invert,view_mode],output+[gallery],concurrency_id='cuda',concurrency_limit=1,show_progress='minimal'),
                      demo.load(load_scene,[scene,axis,position,invert,view_mode],output,concurrency_id='cuda',concurrency_limit=1,show_progress='minimal'),
                      view_mode.change(iframe,[scene,view_mode,axis,position],[viewport],queue=False,show_progress='hidden')]:
            event.then(fn=None,inputs=preview_inputs,outputs=[],js=PREVIEW_JS,queue=False)
        mesh_button.click(export_surface,[scene],[mesh_button,mesh_download,mesh_status,mesh_preview],concurrency_id='cuda',concurrency_limit=1,show_progress='hidden')
        render_button.click(render_current,[scene,azimuth,elevation,exposure,invert],[projection,download,stats,azimuth,elevation],
            js="""(scene,a,e,exposure,invert)=>{const v=document.querySelector('#volume-frame')?.contentWindow?.__xray;if(v?.ready&&v.getScannerView){const c=v.getScannerView();a=c.azimuth;e=Math.max(-80,Math.min(80,c.elevation));}return [scene,a,e,exposure,invert];}""",
            concurrency_id='cuda',concurrency_limit=1,show_progress='minimal')
        for event in [position.change,axis.change]:event(get_slice,[scene,axis,position],[slice_image,slice_caption,volume_download],concurrency_id='cuda',concurrency_limit=1,trigger_mode='always_last',show_progress='hidden')
    demo.queue(max_size=16,default_concurrency_limit=1);app=FastAPI()
    @app.get('/health')
    def health():return dict(ok=worker.available(),scenes=ORDER,gpu=worker.info.get('device'))
    @app.get('/api/scene/{scene_id}')
    def scene_data(scene_id:str):
        if scene_id not in SPECS:raise HTTPException(404)
        return dict(scene=SPECS[scene_id],asset_base='/fields/',points_url='/points/'+scene_id if scene_id in POINTS else None,point_count=POINTS.get(scene_id,{}).get('count'),point_geometry=CATALOG[scene_id]['geometry'])
    @app.get('/api/mesh/{scene_id}')
    def mesh_info(scene_id:str):
        if scene_id not in SPECS:raise HTTPException(404)
        info=cached_mesh(scene_id);spec=SPECS[scene_id]
        return dict(ready=bool(info),url='/meshes/'+scene_id,orbit=f'{spec["yaw"]}deg {spec["pitch"]}deg 135%',name=spec['name'])
    @app.get('/meshes/{scene_id}')
    def mesh_data(scene_id:str):
        info=MESHES.get(scene_id)
        if not info:raise HTTPException(404)
        return FileResponse(info['file'],media_type='model/gltf-binary')
    @app.get('/points/{scene_id}')
    def point_data(scene_id:str):
        if scene_id not in POINTS:raise HTTPException(404)
        return FileResponse(POINTS[scene_id]['file'],media_type='application/octet-stream')
    @app.get('/fields/{name}')
    def field(name:str):
        if name not in FIELDS:raise HTTPException(404)
        path=FIELDS[name]
        if not path.is_file():path=field_file(name)
        return FileResponse(path,media_type='application/octet-stream')
    app.mount('/viewer',StaticFiles(directory=ROOT/'static/viewer'),name='viewer')
    app.mount('/mesh-viewer',StaticFiles(directory=ROOT/'static/mesh-viewer'),name='mesh-viewer')
    app.mount('/demo-assets',StaticFiles(directory=ROOT/'static/assets'),name='assets')
    return gr.mount_gradio_app(app,demo,path='/',allowed_paths=[str(CACHE),str(REPO/'data/meshes'),str(ROOT/'static/assets/thumbnails')],show_error=True,show_api=False,max_file_size='512mb')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--device',type=int,default=0);parser.add_argument('--cuda-python',default=sys.executable)
    args=parser.parse_args();worker=Worker(args.cuda_python,args.device);app=build_app(worker)
    print(f'Demo: http://127.0.0.1:{args.port}',flush=True);uvicorn.run(app,host=args.host,port=args.port,log_level='info')
