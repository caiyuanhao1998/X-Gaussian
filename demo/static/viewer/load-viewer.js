const params=new URLSearchParams(location.search);
const meta=await (await fetch('/api/scene/'+encodeURIComponent(params.get('scene')||'head'))).json();
const xray=params.get('display')==='xray'||!meta.points_url;document.body.classList.add(xray?'volume-mode':'points-mode');
await import(xray?'./viewer.js?v=4':'./points.js?v=4');
