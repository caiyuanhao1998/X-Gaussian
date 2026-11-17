function showError(error){const status=document.getElementById('status');status.hidden=false;status.textContent='Unable to open this X-ray view: '+error.message;document.getElementById('retry').hidden=false;console.error(error);}
try{
const query=new URLSearchParams(location.search);
const requested=query.get('scene')||'head';
const sceneResponse=await fetch('/api/scene/'+encodeURIComponent(requested));
if(!sceneResponse.ok)throw new Error('Scene preview unavailable.');
const sceneData=await sceneResponse.json();
const entry=sceneData.scene;
const index={scenes:[entry],asset_base:sceneData.asset_base};
const native=query.get('quality')==='native';
const spec={...entry,...(!native&&entry.preview?entry.preview:{})};
const sceneMenu=document.getElementById('scene');
for(const s of index.scenes){const option=document.createElement('option');option.value=s.scene;option.textContent=s.name;option.selected=s.scene===entry.scene;sceneMenu.append(option);}
sceneMenu.addEventListener('change',()=>{const params=new URLSearchParams({scene:sceneMenu.value});location.search=params.toString();});
const qualityMenu=document.getElementById('quality');
if(entry.preview){document.getElementById('detail-control').hidden=false;qualityMenu.value=native?'native':'preview';qualityMenu.addEventListener('change',()=>{const params=new URLSearchParams({scene:entry.scene});if(qualityMenu.value==='native')params.set('quality','native');location.search=params.toString();});}
document.getElementById('resolution').textContent=spec.dimensions.join(' × ');
document.getElementById('field-download').href=index.asset_base+(entry.display_cleanup?.source_field_path||entry.field_path);

const $=id=>document.getElementById(id);
$('title').textContent=spec.name;document.title=`${spec.name} · X-ray · X-Gaussian`;
const canvas=$('view');canvas.setAttribute('aria-label',`Interactive ${spec.name} X-ray view`);
const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true,powerPreference:'high-performance'});
if(!gl)throw new Error('This viewer needs a browser with WebGL 2 support.');
const vertex=`#version 300 es
precision highp float;
out vec2 uv;
void main(){vec2 q=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));uv=q;gl_Position=vec4(q*2.-1.,0.,1.);}`;
const fragment=`#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 uv;out vec4 outputColor;
uniform sampler3D density;
uniform vec3 size,eye,right,up,forward;
uniform mat3 inverseRotation;
uniform float aspect,tanFov,scale,exposure,contrast,threshold;
uniform int steps;
uniform int sliceAxis;uniform float slicePosition;
uniform bool invert,cool;
void main(){
 vec3 ro=inverseRotation*eye;
 vec3 rd=normalize(inverseRotation*(forward+(uv.x*2.-1.)*aspect*tanFov*right+(uv.y*2.-1.)*tanFov*up));
 vec3 safe=vec3(rd.x==0.?1e-7:rd.x,rd.y==0.?1e-7:rd.y,rd.z==0.?1e-7:rd.z);
 vec3 t1=(-.5*size-ro)/safe,t2=(.5*size-ro)/safe;
 vec3 lo=min(t1,t2),hi=max(t1,t2);
 float entry=max(max(lo.x,lo.y),max(lo.z,0.));float exit=min(min(hi.x,hi.y),hi.z);
 float A=0.;
 if(exit>entry){
  float ds=(exit-entry)/float(steps);vec3 p=ro+rd*(entry+.5*ds);vec3 delta=rd*ds;
  for(int i=0;i<512;i++){
   if(i>=steps)break;
   vec2 sampleValue=texture(density,p/size+.5).rg;
   float mu=dot(sampleValue,vec2(65280.,255.))*scale/4095.;
   mu*=smoothstep(threshold,threshold*1.25+1e-5,mu);
   A+=mu*ds;p+=delta;
  }
 }
 float brightness=pow(clamp(1.-exp(-exposure*A),0.,1.),contrast);
 vec3 background=invert?vec3(.97,.98,.99):vec3(0.);
 vec3 tint=invert?vec3(.025,.035,.05):(cool?vec3(.69,.86,1.):vec3(.96,.975,1.));
 vec3 color=mix(background,tint,brightness);
 if(sliceAxis>=0 && abs(rd[sliceAxis])>1e-5){
  float t=(slicePosition*size[sliceAxis]-ro[sliceAxis])/rd[sliceAxis];
  vec3 q=(ro+rd*t)/size;
  if(t>0. && all(lessThanEqual(abs(q),vec3(.502)))){
   float edge=0.;for(int j=0;j<3;j++){if(j!=sliceAxis)edge=max(edge,smoothstep(.482,.5,abs(q[j])));}
   color=mix(color,vec3(.22,.77,.86),.06+edge*.65);
  }
 }
 outputColor=vec4(color,1.);
}`;
function compile(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
const program=gl.createProgram();gl.attachShader(program,compile(gl.VERTEX_SHADER,vertex));gl.attachShader(program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const uniforms={};for(const n of ['density','size','eye','right','up','forward','inverseRotation','aspect','tanFov','scale','exposure','contrast','threshold','steps','invert','cool','sliceAxis','slicePosition'])uniforms[n]=gl.getUniformLocation(program,n);
const sub=(a,b)=>a.map((x,i)=>x-b[i]);const plus=(a,b)=>a.map((x,i)=>x+b[i]);const mul=(a,s)=>a.map(x=>x*s);const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];const norm=a=>mul(a,1/Math.hypot(...a));
const originalTarget=spec.target.slice();const diagonal=Math.hypot(...spec.size);
const initialDistance=spec.camera_radius?spec.camera_radius/Math.sin(17.5*Math.PI/180)*.9:diagonal*.5/Math.sin(17.5*Math.PI/180)*.78;
let yaw=spec.yaw*Math.PI/180,polar=spec.pitch*Math.PI/180,distance=initialDistance,target=[0,0,0];
let sliceAxis=-1,slicePosition=0.;
let exposure=spec.exposure,contrast=1.45,threshold=spec.threshold,invert=false,cool=false,ready=false,autoRotate=false,lastTime=0,frame=0,settleTimer=0,drawCount=0;
$('rotation').textContent='Start rotation';$('rotation').setAttribute('aria-pressed','false');$('exposure').value=exposure;$('contrast').value=contrast;$('structure').value=threshold/spec.scale*100;
function labelControls(){$('exposure-value').textContent=`${exposure.toFixed(1)}×`;$('contrast-value').textContent=contrast.toFixed(2);$('structure-value').textContent=`${Math.round(threshold/spec.scale*100)}%`;}
labelControls();
function camera(){const eye=plus(target,[distance*Math.sin(polar)*Math.sin(yaw),distance*Math.cos(polar),distance*Math.sin(polar)*Math.cos(yaw)]);const f=norm(sub(target,eye)),r=norm(cross(f,[0,1,0])),u=cross(r,f);return {eye,f,r,u};}
function draw(moving=false){
 if(!ready)return;
 const rect=canvas.getBoundingClientRect();const maxEdge=moving?380:640;const factor=Math.min(1,maxEdge/Math.max(rect.width,rect.height));
 const width=Math.max(1,Math.round(rect.width*factor)),height=Math.max(1,Math.round(rect.height*factor));
 if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
 gl.viewport(0,0,width,height);gl.useProgram(program);const c=camera();
 gl.uniform3fv(uniforms.eye,c.eye);gl.uniform3fv(uniforms.right,c.r);gl.uniform3fv(uniforms.up,c.u);gl.uniform3fv(uniforms.forward,c.f);
 gl.uniform1f(uniforms.aspect,width/height);gl.uniform1f(uniforms.tanFov,Math.tan(17.5*Math.PI/180));
 gl.uniform1f(uniforms.exposure,exposure);gl.uniform1f(uniforms.contrast,contrast);gl.uniform1f(uniforms.threshold,threshold);
 gl.uniform1i(uniforms.invert,invert);gl.uniform1i(uniforms.cool,cool);gl.uniform1i(uniforms.steps,moving?128:320);gl.uniform1i(uniforms.sliceAxis,sliceAxis);gl.uniform1f(uniforms.slicePosition,slicePosition);
 gl.drawArrays(gl.TRIANGLES,0,3);drawCount++;
}
function schedule(moving=false){if(!ready)return;if(autoRotate){if(!frame)frame=requestAnimationFrame(animate);return;}if(frame)cancelAnimationFrame(frame);frame=requestAnimationFrame(()=>{frame=0;draw(moving);});if(moving){clearTimeout(settleTimer);settleTimer=setTimeout(()=>schedule(false),180);}}
function animate(time){if(!autoRotate||!ready){frame=0;lastTime=0;return;}if(lastTime)yaw+=(time-lastTime)*.00011;lastTime=time;draw(true);frame=requestAnimationFrame(animate);}
function stopRotation(){autoRotate=false;lastTime=0;if(frame)cancelAnimationFrame(frame);frame=0;$('rotation').textContent='Start rotation';$('rotation').setAttribute('aria-pressed','false');}
function reset(){stopRotation();yaw=spec.yaw*Math.PI/180;polar=spec.pitch*Math.PI/180;distance=initialDistance;target=[0,0,0];autoRotate=true;$('rotation').textContent='Pause rotation';$('rotation').setAttribute('aria-pressed','true');lastTime=0;schedule();}
$('rotation').addEventListener('click',()=>{autoRotate=!autoRotate;$('rotation').textContent=autoRotate?'Pause rotation':'Start rotation';$('rotation').setAttribute('aria-pressed',String(autoRotate));if(autoRotate){lastTime=0;frame=requestAnimationFrame(animate);}else{stopRotation();schedule();}});
$('reset').addEventListener('click',reset);
for(const n of ['exposure','contrast','structure'])$(n).addEventListener('input',()=>{if(n==='exposure')exposure=+$(n).value;if(n==='contrast')contrast=+$(n).value;if(n==='structure')threshold=+$(n).value/100*spec.scale;labelControls();stopRotation();schedule(true);});
$('invert').addEventListener('change',()=>{invert=$('invert').checked;schedule();});$('cool').addEventListener('change',()=>{cool=$('cool').checked;schedule();});
$('download').addEventListener('click',()=>{draw();const a=document.createElement('a');a.download=`x-gaussian-${spec.scene}-xray.png`;a.href=canvas.toDataURL('image/png');a.click();});
const pointers=new Map();let drag=null,pinchDistance=0;
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{e.preventDefault();stopRotation();canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};if(pointers.size===2){const p=[...pointers.values()];pinchDistance=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);}});
canvas.addEventListener('pointermove',e=>{if(!pointers.has(e.pointerId))return;const previous=pointers.get(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);if(pointers.size===2){const p=[...pointers.values()],d=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);if(pinchDistance>0)distance=Math.max(initialDistance*.25,Math.min(initialDistance*3,distance*pinchDistance/Math.max(d,1)));pinchDistance=d;}else{const dx=e.clientX-previous[0],dy=e.clientY-previous[1];if(drag.pan){const c=camera(),k=distance*Math.tan(17.5*Math.PI/180)*2/canvas.clientHeight;target=plus(target,plus(mul(c.r,-dx*k),mul(c.u,dy*k)));}else{yaw-=dx*.008;polar=Math.max(.08,Math.min(Math.PI-.08,polar-dy*.006));}}schedule(true);});
function release(e){pointers.delete(e.pointerId);if(!pointers.size){drag=null;schedule(false);}pinchDistance=0;}
canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
canvas.addEventListener('wheel',e=>{e.preventDefault();stopRotation();distance=Math.max(initialDistance*.25,Math.min(initialDistance*3,distance*Math.exp(e.deltaY*.001)));schedule(true);},{passive:false});
new ResizeObserver(()=>schedule()).observe(canvas);
window.__xray={ready:false,getState:()=>({yaw,polar,distance,target:target.slice(),exposure,contrast,threshold,invert,cool,autoRotate,drawCount,width:canvas.width,height:canvas.height}),reset,draw:()=>draw(false),metadata:{scene:spec.scene,dimensions:spec.dimensions,nativeDimensions:entry.source_native_dimensions||entry.dimensions,source:'Gaussian-derived CT',projection:'ray-integrated display density; exposure, contrast and structure window are visualization controls'}};
try{
 const response=await fetch(index.asset_base+spec.field_path);
 if(!response.ok)throw new Error(`Volume download failed (${response.status}).`);
 let received=0;const total=spec.bytes||Number(response.headers.get('content-length'))||0;
 const reader=response.body.getReader();
 const compressed=new ReadableStream({async pull(controller){const part=await reader.read();if(part.done){controller.close();return;}received+=part.value.byteLength;$('status').textContent=total?`Loading ${spec.name} · ${Math.min(99,Math.round(received/total*100))}%`:`Loading ${spec.name} · ${(received/1048576).toFixed(1)} MB`;controller.enqueue(part.value);},cancel(){reader.cancel();}});
 if(!globalThis.DecompressionStream)throw new Error('Please use a recent version of Chrome, Edge, Firefox or Safari.');
 const buffer=await new Response(compressed.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 const expected=spec.dimensions.reduce((a,b)=>a*b,1)*2;if(buffer.byteLength!==expected)throw new Error('Incomplete volume data.');
 const texture=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,texture);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
 gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
 for(const axis of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,axis,gl.CLAMP_TO_EDGE);
 gl.texImage3D(gl.TEXTURE_3D,0,gl.RG8,...spec.dimensions,0,gl.RG,gl.UNSIGNED_BYTE,new Uint8Array(buffer));
 const error=gl.getError();if(error!==gl.NO_ERROR)throw new Error(`Unable to create the volume texture (${error}).`);
 gl.uniform1i(uniforms.density,0);gl.uniform3fv(uniforms.size,spec.size);gl.uniformMatrix3fv(uniforms.inverseRotation,false,new Float32Array(spec.rotation.flat()));gl.uniform1f(uniforms.scale,spec.scale);
 ready=true;draw();$('status').hidden=true;window.__xray.ready=true;document.body.dataset.ready='true';schedule();
}catch(error){$('status').textContent='Unable to open this X-ray viewer: '+error.message;window.__xray.error=error.message;throw error;}


function scannerView(){
 const c=camera();const v=norm(sub(c.eye,target));
 const r=spec.rotation;const w=[0,1,2].map(i=>r[0][i]*v[0]+r[1][i]*v[1]+r[2][i]*v[2]);
 return {azimuth:(Math.atan2(w[1],w[0])*180/Math.PI+360)%360,elevation:Math.asin(Math.max(-1,Math.min(1,w[2])))*180/Math.PI};
}
function setScannerView(a,e){
 stopRotation();target=[0,0,0];
 a*=Math.PI/180;e*=Math.PI/180;
 const w=[Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e)];
 const d=spec.rotation.map(row=>row.reduce((v,x,i)=>v+x*w[i],0));
 yaw=Math.atan2(d[0],d[2]);polar=Math.acos(Math.max(-1,Math.min(1,d[1])));schedule(true);
}
window.__xray.getScannerView=scannerView;
window.__xray.setScannerView=setScannerView;
window.addEventListener('message',event=>{
 if(event.origin!==location.origin||event.data?.kind!=='xgaussian-controls')return;
 const c=event.data;
 if(c.scene&&c.scene!==spec.scene)return;
 if(c.cameraChanged!==false&&Number.isFinite(c.azimuth)&&Number.isFinite(c.elevation))setScannerView(c.azimuth,c.elevation);
 if(Number.isFinite(c.exposure))exposure=c.exposure;
 if(typeof c.invert==='boolean')invert=c.invert;
 if(Number.isFinite(c.axis))sliceAxis=c.axis;
 if(Number.isFinite(c.position)){const roundVoxel=x=>{const low=Math.floor(x);return x-low===.5?(low%2?low+1:low):Math.round(x);};const n=(sceneData.point_geometry?.nVoxel||entry.dimensions)[sliceAxis];slicePosition=n?(roundVoxel(c.position/100*(n-1))+.5)/n-.5:c.position/100-.5;}
 labelControls();schedule(true);
});
window.parent.postMessage({kind:'xgaussian-ready',scene:spec.scene},location.origin);

document.addEventListener('visibilitychange',()=>{if(document.hidden){if(frame)cancelAnimationFrame(frame);frame=0;lastTime=0;}else schedule();});

}catch(error){showError(error);}
