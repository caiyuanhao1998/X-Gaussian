const viewer=document.getElementById('mesh'),status=document.getElementById('status');
const scene=new URLSearchParams(location.search).get('scene')||'head';
window.__mesh={ready:false,scene};
try{
 const response=await fetch('/api/mesh/'+encodeURIComponent(scene));if(!response.ok)throw new Error('Mesh unavailable.');
 const meta=await response.json();
 if(!meta.ready){status.textContent='Click “Export Mesh” below to generate a mesh for this model.';window.__mesh.empty=true;}
 else{
  await customElements.whenDefined('model-viewer');viewer.cameraOrbit=meta.orbit;viewer.src=meta.url;
  viewer.addEventListener('load',()=>{viewer.resetTurntableRotation(0);viewer.jumpCameraToGoal();status.hidden=true;window.__mesh.ready=true;window.__mesh.metadata=meta;});
  viewer.addEventListener('error',()=>{status.hidden=false;status.textContent='Unable to load the exported mesh.';window.__mesh.error=true;});
  document.getElementById('rotation').addEventListener('click',e=>{viewer.autoRotate=!viewer.autoRotate;e.target.textContent=viewer.autoRotate?'Pause':'Rotate';});
  document.getElementById('reset').addEventListener('click',()=>{viewer.cameraOrbit=meta.orbit;viewer.cameraTarget='auto auto auto';viewer.resetTurntableRotation(0);viewer.jumpCameraToGoal();});
 }
}catch(error){status.textContent=error.message;window.__mesh.error=error.message;console.error(error);}
