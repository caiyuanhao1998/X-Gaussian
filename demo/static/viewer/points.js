// Render the actual Gaussian centers as GL_POINTS; no mesh or CT surface is used.
const $=id=>document.getElementById(id);
try{
 const params=new URLSearchParams(location.search),scene=params.get('scene')||'head',thumbnail=params.get('thumbnail')==='1';
 if(thumbnail)document.body.classList.add('thumbnail-mode');
 const response=await fetch('/api/scene/'+encodeURIComponent(scene));
 if(!response.ok)throw new Error('Point cloud metadata unavailable.');
 const data=await response.json(),spec=data.scene,g=data.point_geometry;
 if(!data.points_url||!g)throw new Error('Point cloud is not ready. Please reload the scene.');
 const canvas=$('view'),gl=canvas.getContext('webgl2',{alpha:false,antialias:true,preserveDrawingBuffer:true});
 if(!gl)throw new Error('WebGL 2 is required for the point cloud viewer.');
 const vertex=`#version 300 es
 precision highp float;
 layout(location=0) in vec4 point;
 uniform mat3 rotation;
 uniform vec3 center,eye,right,up,forward;
 uniform float aspect,tanFov,nearClip,farClip,pointSize;
 out float strength;
 void main(){vec3 p=rotation*(point.xyz-center)-eye;float z=dot(p,forward);
 gl_Position=vec4(dot(p,right)/(aspect*tanFov),dot(p,up)/tanFov,(farClip+nearClip)/(farClip-nearClip)*z-2.*farClip*nearClip/(farClip-nearClip),z);
 gl_PointSize=pointSize;strength=point.w;}`;
 const fragment=`#version 300 es
 precision highp float;in float strength;out vec4 color;
 uniform bool whiteBackground,lineMode;
 uniform float exposure;
 void main(){if(lineMode){color=vec4(.28,.78,.87,.9);return;}
 float radius=length(gl_PointCoord-vec2(.5));if(radius>.5)discard;
 float s=log(1.+max(.3,exposure)*strength)/log(1.+max(.3,exposure));
 float gray=whiteBackground?(.18+.3*(1.-s)):(.63+.35*s);
 float alpha=(.15+.7*s)*(1.-smoothstep(.28,.5,radius));color=vec4(vec3(gray),alpha);}`;
 function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
 const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);
 if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
 gl.useProgram(program);const u={};for(const n of ['rotation','center','eye','right','up','forward','aspect','tanFov','nearClip','farClip','pointSize','whiteBackground','lineMode','exposure'])u[n]=gl.getUniformLocation(program,n);
 const rotation=spec.rotation,center=g.offOrigin,size=g.sVoxel;
 gl.uniformMatrix3fv(u.rotation,false,new Float32Array([0,1,2].flatMap(j=>[0,1,2].map(i=>rotation[i][j]))));
 gl.uniform3fv(u.center,center);
 const norm=v=>{const d=Math.hypot(...v);return v.map(x=>x/(d||1));},cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],sub=(a,b)=>a.map((x,i)=>x-b[i]);
 const diagonal=Math.hypot(...size),initialDistance=diagonal*.5/Math.sin(17.5*Math.PI/180)*.86;
 let yaw=spec.yaw*Math.PI/180,polar=spec.pitch*Math.PI/180,distance=initialDistance,target=[0,0,0],exposure=spec.exposure||6,invert=false,autoRotate=false;
 let ready=false,drawCount=0,frame=0,lastTime=0,sliceAxis=thumbnail?-1:Number(params.get('plane_axis')||'1'),slicePercent=Number(params.get('slice_position')||'50'),slicePosition=0,pointBuffer,count=0;
 let moveSliceMode=false,hasSliceTexture=false,lastSliceImage='',loadedSliceImage='',lastEmit=0;
 const roundVoxel=x=>{const low=Math.floor(x);return x-low===.5?(low%2?low+1:low):Math.round(x);};
 const normalizedPosition=()=>sliceAxis<0?0:(roundVoxel(slicePercent/100*(g.nVoxel[sliceAxis]-1))+.5)/g.nVoxel[sliceAxis]-.5;
 slicePosition=normalizedPosition();
 $('reset').textContent='Reset';$('rotation').textContent='Rotate';
 const planeBuffer=gl.createBuffer();
 const planeVertex=`#version 300 es
 precision highp float;layout(location=0) in vec4 point;uniform mat3 rotation;uniform vec3 center,size,eye,right,up,forward;uniform float aspect,tanFov,nearClip,farClip;uniform int planeAxis;out vec2 sliceUV;
 void main(){vec3 uvw=(point.xyz-center)/size+.5;sliceUV=planeAxis==0?uvw.yz:(planeAxis==1?uvw.xz:uvw.xy);vec3 p=rotation*(point.xyz-center)-eye;float z=dot(p,forward);gl_Position=vec4(dot(p,right)/(aspect*tanFov),dot(p,up)/tanFov,(farClip+nearClip)/(farClip-nearClip)*z-2.*farClip*nearClip/(farClip-nearClip),z);}`;
 const planeFragment=`#version 300 es
 precision highp float;in vec2 sliceUV;uniform sampler2D sliceTexture;uniform bool hasTexture;out vec4 color;
 void main(){float value=hasTexture?texture(sliceTexture,sliceUV).r:.25;color=vec4(mix(vec3(.10,.60,.73),vec3(.84,.97,1.),value),.08+.36*value);}`;
 const planeProgram=gl.createProgram();gl.attachShader(planeProgram,shader(gl.VERTEX_SHADER,planeVertex));gl.attachShader(planeProgram,shader(gl.FRAGMENT_SHADER,planeFragment));gl.linkProgram(planeProgram);if(!gl.getProgramParameter(planeProgram,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(planeProgram));
 const pu={};for(const n of ['rotation','center','size','eye','right','up','forward','aspect','tanFov','nearClip','farClip','planeAxis','sliceTexture','hasTexture'])pu[n]=gl.getUniformLocation(planeProgram,n);
 const sliceTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,sliceTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([0,0,0,255]));gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
 function loadSliceTexture(url){if(!url||url===lastSliceImage)return;lastSliceImage=url;const image=new Image();image.onload=()=>{if(url!==lastSliceImage)return;gl.bindTexture(gl.TEXTURE_2D,sliceTexture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);hasSliceTexture=true;loadedSliceImage=url;schedule();};image.src=url;}

 function camera(){const eye=target.map((x,i)=>x+[distance*Math.sin(polar)*Math.sin(yaw),distance*Math.cos(polar),distance*Math.sin(polar)*Math.cos(yaw)][i]);
  const f=norm(sub(target,eye)),r=norm(cross(f,Math.abs(f[1])>.9999?[0,0,1]:[0,1,0])),up=cross(r,f);return {eye,f,r,up};}
 function attribute(buffer){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,4,gl.FLOAT,false,16,0);}
 function draw(){if(!ready)return;const rect=canvas.getBoundingClientRect(),factor=Math.min(devicePixelRatio||1,1.5,1100/Math.max(rect.width,rect.height));
  canvas.width=Math.max(1,Math.round(rect.width*factor));canvas.height=Math.max(1,Math.round(rect.height*factor));
  gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(thumbnail||invert?1:0,thumbnail||invert?1:0,thumbnail||invert?1:0,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);const c=camera();gl.uniform3fv(u.eye,c.eye);gl.uniform3fv(u.right,c.r);gl.uniform3fv(u.up,c.up);gl.uniform3fv(u.forward,c.f);
  gl.uniform1f(u.aspect,canvas.width/canvas.height);gl.uniform1f(u.tanFov,Math.tan(17.5*Math.PI/180));gl.uniform1f(u.nearClip,diagonal*.002);gl.uniform1f(u.farClip,initialDistance*30);
  gl.uniform1f(u.pointSize,(thumbnail?1.5:1.7)*factor);gl.uniform1i(u.whiteBackground,thumbnail||invert);gl.uniform1f(u.exposure,exposure);gl.uniform1i(u.lineMode,false);
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.enable(gl.DEPTH_TEST);gl.depthMask(true);
  attribute(pointBuffer);gl.drawArrays(gl.POINTS,0,count);
  if(sliceAxis>=0){
   const axes=[0,1,2].filter(i=>i!==sliceAxis),vertices=[];
   for(const signs of [[-1,-1],[1,-1],[1,1],[-1,1]]){const p=center.slice();p[sliceAxis]+=slicePosition*size[sliceAxis];for(let j=0;j<2;j++)p[axes[j]]+=signs[j]*size[axes[j]]*.5;vertices.push(...p,1);}
   const middle=center.slice();middle[sliceAxis]+=slicePosition*size[sliceAxis];const normalTip=middle.slice();normalTip[sliceAxis]+=size[sliceAxis]*.12;vertices.push(...middle,1,...normalTip,1);
   gl.disable(gl.DEPTH_TEST);gl.depthMask(false);gl.bindBuffer(gl.ARRAY_BUFFER,planeBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.DYNAMIC_DRAW);attribute(planeBuffer);
   gl.useProgram(planeProgram);gl.uniformMatrix3fv(pu.rotation,false,new Float32Array([0,1,2].flatMap(j=>[0,1,2].map(i=>rotation[i][j]))));gl.uniform3fv(pu.center,center);gl.uniform3fv(pu.size,size);gl.uniform3fv(pu.eye,c.eye);gl.uniform3fv(pu.right,c.r);gl.uniform3fv(pu.up,c.up);gl.uniform3fv(pu.forward,c.f);gl.uniform1f(pu.aspect,canvas.width/canvas.height);gl.uniform1f(pu.tanFov,Math.tan(17.5*Math.PI/180));gl.uniform1f(pu.nearClip,diagonal*.002);gl.uniform1f(pu.farClip,initialDistance*30);gl.uniform1i(pu.planeAxis,sliceAxis);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,sliceTexture);gl.uniform1i(pu.sliceTexture,0);gl.uniform1i(pu.hasTexture,hasSliceTexture);gl.drawArrays(gl.TRIANGLE_FAN,0,4);
   gl.useProgram(program);gl.uniform1i(u.lineMode,true);gl.drawArrays(gl.LINE_LOOP,0,4);gl.drawArrays(gl.LINES,4,2);
   gl.uniform1f(u.pointSize,14*factor);gl.drawArrays(gl.POINTS,4,1);gl.uniform1i(u.lineMode,false);gl.uniform1f(u.pointSize,8*factor);gl.drawArrays(gl.POINTS,4,1);
   gl.depthMask(true);
  }
  drawCount++;
 }
 function schedule(){if(frame||!ready)return;frame=requestAnimationFrame(time=>{frame=0;if(autoRotate){if(lastTime)yaw+=(time-lastTime)*.00011;lastTime=time;}draw();if(autoRotate)schedule();});}
 function stopRotation(){autoRotate=false;lastTime=0;$('rotation').textContent='Rotate';$('rotation').setAttribute('aria-pressed','false');}
 function reset(){stopRotation();yaw=spec.yaw*Math.PI/180;polar=spec.pitch*Math.PI/180;distance=initialDistance;target=[0,0,0];schedule();}
 function scannerView(){const c=camera(),v=norm(sub(c.eye,target)),w=[0,1,2].map(i=>rotation[0][i]*v[0]+rotation[1][i]*v[1]+rotation[2][i]*v[2]);return {azimuth:(Math.atan2(w[1],w[0])*180/Math.PI+360)%360,elevation:Math.asin(Math.max(-1,Math.min(1,w[2])))*180/Math.PI};}
 function setScannerView(a,e){stopRotation();target=[0,0,0];a*=Math.PI/180;e*=Math.PI/180;const w=[Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e)],d=rotation.map(row=>row.reduce((v,x,i)=>v+x*w[i],0));yaw=Math.atan2(d[0],d[2]);polar=Math.acos(Math.max(-1,Math.min(1,d[1])));schedule();}
 const state={ready:false,getState:()=>({yaw,polar,distance,target:target.slice(),exposure,autoRotate,drawCount,width:canvas.width,height:canvas.height,pointCount:count,sliceAxis,slicePosition,slicePercent,hasSliceTexture,loadedSliceImage,moveSliceMode,handle:sliceHandle()}),
  getScannerView:scannerView,setScannerView,reset,draw,metadata:{scene,source:'Original Gaussian centers',primitive:'GL_POINTS',pointCount:data.point_count,dimensions:g.nVoxel,nativeDimensions:g.nVoxel}};
 window.__xray=state;window.__points=state;
 $('rotation').addEventListener('click',()=>{autoRotate=!autoRotate;lastTime=0;$('rotation').textContent=autoRotate?'Pause':'Rotate';$('rotation').setAttribute('aria-pressed',String(autoRotate));schedule();});$('reset').addEventListener('click',reset);

 function sliceHandle(){if(sliceAxis<0)return null;const c=camera(),relative=[0,0,0];relative[sliceAxis]=slicePosition*size[sliceAxis];const p=rotation.map(row=>row.reduce((v,x,i)=>v+x*relative[i],0)),q=sub(p,c.eye),depth=q.reduce((v,x,i)=>v+x*c.f[i],0),rect=canvas.getBoundingClientRect(),t=Math.tan(17.5*Math.PI/180);return {x:(1+q.reduce((v,x,i)=>v+x*c.r[i],0)/(depth*t*rect.width/rect.height))*rect.width/2,y:(1-q.reduce((v,x,i)=>v+x*c.up[i],0)/(depth*t))*rect.height/2};}
 function emitSlice(force=false){const now=performance.now();if(!force&&now-lastEmit<45)return;lastEmit=now;window.parent.postMessage({kind:'xgaussian-slice-drag',scene,position:slicePercent},location.origin);}
 function moveSlice(value,emit=true){slicePercent=Math.max(0,Math.min(100,value));slicePosition=normalizedPosition();schedule();if(emit)emitSlice();}
 $('slice-move').addEventListener('click',()=>{if(sliceAxis<0)return;moveSliceMode=!moveSliceMode;$('slice-move').setAttribute('aria-pressed',String(moveSliceMode));canvas.style.cursor=moveSliceMode?'ns-resize':'grab';stopRotation();});
 let drag=null;canvas.addEventListener('contextmenu',e=>e.preventDefault());
 canvas.addEventListener('pointerdown',e=>{e.preventDefault();stopRotation();canvas.setPointerCapture(e.pointerId);const rect=canvas.getBoundingClientRect(),h=sliceHandle(),hit=h&&Math.hypot(e.clientX-rect.left-h.x,e.clientY-rect.top-h.y)<18;
  drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,startPercent:slicePercent,slice:sliceAxis>=0&&(moveSliceMode||hit||e.altKey),pan:e.button===2||e.shiftKey};});
 canvas.addEventListener('pointermove',e=>{if(!drag){const rect=canvas.getBoundingClientRect(),h=sliceHandle();canvas.style.cursor=(moveSliceMode||(h&&Math.hypot(e.clientX-rect.left-h.x,e.clientY-rect.top-h.y)<18))?'ns-resize':'grab';return;}
  const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
  if(drag.slice){const c=camera(),normal=rotation.map(row=>row[sliceAxis]),sx=normal.reduce((v,x,i)=>v+x*c.r[i],0),sy=-normal.reduce((v,x,i)=>v+x*c.up[i],0),length2=sx*sx+sy*sy,rect=canvas.getBoundingClientRect();let delta;
   if(length2<.12)delta=-(e.clientY-drag.startY)/rect.height*100;
   else{const pixelsPerUnit=rect.height/(2*Math.tan(17.5*Math.PI/180)*distance);delta=((e.clientX-drag.startX)*sx+(e.clientY-drag.startY)*sy)/(length2*pixelsPerUnit*size[sliceAxis])*100;}
   moveSlice(drag.startPercent+delta);return;
  }
  if(drag.pan){const c=camera(),k=distance*Math.tan(17.5*Math.PI/180)*2/canvas.clientHeight;target=target.map((x,i)=>x+(-dx*c.r[i]+dy*c.up[i])*k);}else{yaw-=dx*.008;polar=Math.max(.03,Math.min(Math.PI-.03,polar-dy*.006));}schedule();});
 function finishDrag(){if(drag?.slice)emitSlice(true);drag=null;schedule();}
 canvas.addEventListener('pointerup',finishDrag);canvas.addEventListener('pointercancel',finishDrag);
 canvas.addEventListener('wheel',e=>{e.preventDefault();stopRotation();if(sliceAxis>=0&&(moveSliceMode||e.shiftKey)){const n=g.nVoxel[sliceAxis],index=Math.round(slicePercent/100*(n-1))+Math.sign(e.deltaY);moveSlice(index/(n-1)*100);emitSlice(true);}else{distance=Math.max(initialDistance*.15,Math.min(initialDistance*4,distance*Math.exp(e.deltaY*.001)));schedule();}},{passive:false});
 new ResizeObserver(schedule).observe(canvas);
 window.addEventListener('message',event=>{if(event.origin!==location.origin||event.data?.kind!=='xgaussian-controls')return;const c=event.data;if(c.scene&&c.scene!==scene)return;
  if(c.cameraChanged!==false&&Number.isFinite(c.azimuth)&&Number.isFinite(c.elevation))setScannerView(c.azimuth,c.elevation);if(Number.isFinite(c.exposure))exposure=c.exposure;if(typeof c.invert==='boolean')invert=c.invert;
  if(Number.isFinite(c.axis))sliceAxis=c.axis;if(Number.isFinite(c.position))moveSlice(c.position,false);else slicePosition=normalizedPosition();$('slice-move').disabled=sliceAxis<0;if(c.sliceImage)loadSliceTexture(c.sliceImage);schedule();});
 const pointResponse=await fetch(data.points_url);if(!pointResponse.ok)throw new Error('Unable to load Gaussian centers.');
 const array=new Float32Array(await pointResponse.arrayBuffer());if(array.length!==data.point_count*4)throw new Error('Incomplete point cloud.');
 count=data.point_count;pointBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pointBuffer);gl.bufferData(gl.ARRAY_BUFFER,array,gl.STATIC_DRAW);
 ready=true;state.ready=true;draw();$('status').hidden=true;document.body.dataset.ready='true';window.parent.postMessage({kind:'xgaussian-ready',scene},location.origin);
}catch(error){$('status').hidden=false;$('status').textContent=error.message;window.__pointError=error.message;console.error(error);}
