/* Local point-cloud review: no network libraries and no tracker mutations. */
'use strict';
const el=id=>document.getElementById(id);
const cases=JSON.parse(el('data').textContent);
const cloudMeta=JSON.parse(el('cloudMeta').textContent);
const fmt=x=>x==null?'—':(100*x).toFixed(1)+'%';
const cloudCache=new Map(),pendingClouds=new Map();
let renderVersion=0,timer=null,playVersion=0;

function idColor(id){
  const h=(id*137.508%360)/60,s=.75,l=.68,c=(1-Math.abs(2*l-1))*s;
  const x=c*(1-Math.abs(h%2-1)),m=l-c/2;
  const rgb=h<1?[c,x,0]:h<2?[x,c,0]:h<3?[0,c,x]:h<4?[0,x,c]:h<5?[x,0,c]:[c,0,x];
  return rgb.map(v=>v+m);
}
function cssColor(rgb){return 'rgb('+rgb.map(v=>Math.round(v*255)).join(',')+')';}
function viewBasis(){
  const a=+el('azimuth').value*Math.PI/180,e=+el('elevation').value*Math.PI/180;
  return {u:[Math.cos(a),Math.sin(a),0],
    v:[-Math.sin(a)*Math.sin(e),Math.cos(a)*Math.sin(e),Math.cos(e)],
    depth:[Math.sin(a)*Math.cos(e),-Math.cos(a)*Math.cos(e),Math.sin(e)]};
}
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
function projector(f,W,H,basis){
  const scale=W/(+el('extent').value),origin=f.sensor_xyz;
  const relative=p=>[W/2+dot(basis.u,p)*scale,H/2-dot(basis.v,p)*scale];
  return {relative,world:(x,y,z)=>relative([x-origin[0],y-origin[1],z-origin[2]]),scale};
}
function boxCorners(b){
  const c=Math.cos(b.yaw),s=Math.sin(b.yaw),points=[];
  for(const z of [-1,1])for(const [u,v] of [[1,1],[-1,1],[-1,-1],[1,-1]])
    points.push([b.x+u*b.length/2*c-v*b.width/2*s,b.y+u*b.length/2*s+v*b.width/2*c,b.z+z*b.height/2]);
  return points;
}
function drawBox(ctx,b,projection,color,dashed,text){
  const corners=boxCorners(b).map(p=>projection.world(...p));
  ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.setLineDash(dashed?[6,4]:[]);ctx.beginPath();
  for(const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]){
    ctx.moveTo(...corners[a]);ctx.lineTo(...corners[b]);
  }
  ctx.stroke();ctx.setLineDash([]);
  if(text){const [x,y]=projection.world(b.x,b.y,b.z+b.height/2);ctx.font='bold 13px system-ui';
    ctx.lineWidth=3;ctx.strokeStyle='#101821';ctx.strokeText(text,x+4,y-4);ctx.fillStyle=color;ctx.fillText(text,x+4,y-4);}
}
function metric(p){
  const s=p.score.observed_tracks,r=p.score.dropout_recovery;
  return '观测精度 '+fmt(s.precision)+' · 召回 '+fmt(s.recall)+' · IDF1 '+fmt(s.idf1)+' · ID 切换 '+s.id_changes+'\n'+
    (r?'中断后同 ID '+r.same_id+' / '+r.eligible+'；换 ID '+r.changed_id+'；有返回检测但无匹配轨迹 '+r.no_track_after_return+'\n':'')+
    '仅跟踪计算 P99 '+p.timing.toFixed(2)+' ms';
}
window.registerPointcloud=(sequence,index,encoded)=>{
  const key=sequence+'/'+index,meta=cloudMeta?.frames?.[sequence]?.[index];
  try{
    if(!meta)throw new Error('点云元数据缺失');
    const binary=atob(encoded),bytes=new Uint8Array(binary.length);
    for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
    if(bytes.length!==meta.display_points*8)throw new Error('点云长度与元数据不一致');
    const cloud={bytes,xyz:new Int16Array(bytes.buffer),intensity:new Uint16Array(bytes.buffer),meta};
    cloudCache.set(key,cloud);
    while(cloudCache.size>6)cloudCache.delete(cloudCache.keys().next().value);
    pendingClouds.get(key)?.resolve(cloud);
  }catch(error){pendingClouds.get(key)?.reject(error);}
  finally{pendingClouds.delete(key);}
};
function loadCloud(sequence,index){
  const key=sequence+'/'+index;
  if(cloudCache.has(key))return Promise.resolve(cloudCache.get(key));
  if(pendingClouds.has(key))return pendingClouds.get(key).promise;
  if(!cloudMeta?.frames?.[sequence]?.[index])return Promise.resolve(null);
  let resolve,reject;
  const promise=new Promise((ok,bad)=>{resolve=ok;reject=bad;});
  pendingClouds.set(key,{promise,resolve,reject});
  const script=document.createElement('script');
  script.src='pointclouds/'+sequence+'/'+String(index).padStart(6,'0')+'.js';
  script.onload=()=>{if(pendingClouds.has(key)){pendingClouds.get(key).reject(new Error('点云文件未注册数据'));pendingClouds.delete(key);}script.remove();};
  script.onerror=()=>{pendingClouds.get(key)?.reject(new Error('无法读取本地帧点云，请保留 pointclouds 文件夹'));pendingClouds.delete(key);script.remove();};
  document.head.append(script);
  return promise;
}

const VERTEX_SHADER=`
attribute vec3 a_xyz;
attribute float a_intensity;
uniform vec3 u_axis_x, u_axis_y, u_axis_depth;
uniform vec2 u_clip_scale;
uniform float u_point_size;
uniform int u_color_mode, u_box_count;
uniform vec4 u_box_centers[24], u_box_sizes[24];
uniform vec3 u_box_colors[24];
varying vec3 v_color;
void main(){
  vec3 p=a_xyz*0.01;
  gl_Position=vec4(dot(u_axis_x,p)*u_clip_scale.x,dot(u_axis_y,p)*u_clip_scale.y,-dot(u_axis_depth,p)/400.0,1.0);
  gl_PointSize=u_point_size;
  float height=clamp((p.z+3.0)/6.0,0.0,1.0);
  v_color=mix(vec3(0.26,0.42,0.56),vec3(0.90,0.72,0.32),height);
  if(u_color_mode==1)v_color=vec3(0.25+0.75*a_intensity);
  if(u_color_mode==2){
    for(int i=0;i<24;i++){
      if(i<u_box_count){
        vec3 d=p-u_box_centers[i].xyz;
        float c=u_box_centers[i].w,s=u_box_sizes[i].w;
        vec3 local=vec3(d.x*c+d.y*s,-d.x*s+d.y*c,d.z);
        if(all(lessThanEqual(abs(local),u_box_sizes[i].xyz))){v_color=u_box_colors[i];break;}
      }
    }
  }
}`;
const FRAGMENT_SHADER=`precision mediump float;varying vec3 v_color;void main(){gl_FragColor=vec4(v_color,1.0);}`;
function makeRenderer(canvas){
  const gl=canvas.getContext('webgl',{alpha:false,antialias:false});
  if(!gl)return {canvas,ctx:canvas.getContext('2d'),fallback:true};
  function shader(kind,source){const s=gl.createShader(kind);gl.shaderSource(s,source);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
  const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,VERTEX_SHADER));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,FRAGMENT_SHADER));gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
  const uniforms={};
  for(const name of ['u_axis_x','u_axis_y','u_axis_depth','u_clip_scale','u_point_size','u_color_mode','u_box_count',
    'u_box_centers[0]','u_box_sizes[0]','u_box_colors[0]'])uniforms[name]=gl.getUniformLocation(program,name);
  return {canvas,gl,program,uniforms,buffer:gl.createBuffer(),lastCloud:null,
    xyz:gl.getAttribLocation(program,'a_xyz'),intensity:gl.getAttribLocation(program,'a_intensity')};
}
const renderers={};
function renderer(id){
  if(!renderers[id]){
    const canvas=el(id+'Cloud');
    try{renderers[id]=makeRenderer(canvas);}catch(error){
      // A context that compiled unsuccessfully cannot be reused as 2D.
      const replacement=canvas.cloneNode();canvas.replaceWith(replacement);
      renderers[id]={canvas:replacement,ctx:replacement.getContext('2d'),fallback:true,error:error.message};
    }
  }
  return renderers[id];
}
function drawCloud(id,cloud,f,basis){
  const r=renderer(id),W=r.canvas.width,H=r.canvas.height;
  const boxes=f.tracks.filter(t=>!t.predicted_only).slice(0,24);
  const colors=boxes.map(t=>idColor(t.track_id));
  const full=el('fullPoints').checked,mode=+el('pointColor').value;
  let step=cloud?(full?1:Math.max(1,Math.ceil(cloud.meta.display_points/30000))):1;
  if(r.fallback){
    const ctx=r.ctx;ctx.fillStyle='#172330';ctx.fillRect(0,0,W,H);
    if(!cloud)return {points:0,fallback:true};
    step=Math.max(step,Math.ceil(cloud.meta.display_points/30000));
    const projection=projector(f,W,H,basis);let count=0;
    for(let i=0;i<cloud.meta.display_points;i+=step){
      const p=[cloud.xyz[i*4]*.01,cloud.xyz[i*4+1]*.01,cloud.xyz[i*4+2]*.01];
      const height=Math.max(0,Math.min(1,(p[2]+3)/6));
      let color=mode===1?cssColor(Array(3).fill(.25+.75*cloud.intensity[i*4+3]/65535)):
        cssColor([.26+(.90-.26)*height,.42+(.72-.42)*height,.56+(.32-.56)*height]);
      if(mode===2)for(let j=0;j<boxes.length;j++){
        const b=boxes[j],dx=p[0]-(b.x-f.sensor_xyz[0]),dy=p[1]-(b.y-f.sensor_xyz[1]),dz=p[2]-(b.z-f.sensor_xyz[2]);
        const c=Math.cos(b.yaw),s=Math.sin(b.yaw);
        if(Math.abs(dx*c+dy*s)<=b.length/2&&Math.abs(-dx*s+dy*c)<=b.width/2&&Math.abs(dz)<=b.height/2){color=cssColor(colors[j]);break;}
      }
      const [x,y]=projection.relative(p);ctx.fillStyle=color;ctx.fillRect(x,y,1.5,1.5);count++;
    }
    return {points:count,fallback:true};
  }
  const gl=r.gl,u=r.uniforms;gl.viewport(0,0,W,H);gl.clearColor(23/255,35/255,48/255,1);
  gl.enable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if(!cloud)return {points:0,fallback:false};
  gl.useProgram(r.program);gl.bindBuffer(gl.ARRAY_BUFFER,r.buffer);
  if(r.lastCloud!==cloud){gl.bufferData(gl.ARRAY_BUFFER,cloud.bytes,gl.STATIC_DRAW);r.lastCloud=cloud;}
  step=Math.min(step,31); // WebGL 1 attribute stride is at most 255 bytes.
  gl.enableVertexAttribArray(r.xyz);gl.vertexAttribPointer(r.xyz,3,gl.SHORT,false,step*8,0);
  gl.enableVertexAttribArray(r.intensity);gl.vertexAttribPointer(r.intensity,1,gl.UNSIGNED_SHORT,true,step*8,6);
  gl.uniform3fv(u.u_axis_x,basis.u);gl.uniform3fv(u.u_axis_y,basis.v);gl.uniform3fv(u.u_axis_depth,basis.depth);
  gl.uniform2f(u.u_clip_scale,2/(+el('extent').value),2*W/H/(+el('extent').value));
  gl.uniform1f(u.u_point_size,+el('pointSize').value);gl.uniform1i(u.u_color_mode,mode);gl.uniform1i(u.u_box_count,boxes.length);
  const centers=new Float32Array(24*4),sizes=new Float32Array(24*4),palette=new Float32Array(24*3);
  boxes.forEach((b,i)=>{centers.set([b.x-f.sensor_xyz[0],b.y-f.sensor_xyz[1],b.z-f.sensor_xyz[2],Math.cos(b.yaw)],i*4);
    sizes.set([b.length/2,b.width/2,b.height/2,Math.sin(b.yaw)],i*4);palette.set(colors[i],i*3);});
  gl.uniform4fv(u['u_box_centers[0]'],centers);gl.uniform4fv(u['u_box_sizes[0]'],sizes);gl.uniform3fv(u['u_box_colors[0]'],palette);
  const count=Math.ceil(cloud.meta.display_points/step);gl.drawArrays(gl.POINTS,0,count);
  return {points:count,fallback:false};
}
function drawOverlay(id,f,history,index,basis,cloud){
  const canvas=el(id),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
  const projection=projector(f,W,H,basis);ctx.clearRect(0,0,W,H);
  const sx=f.sensor_xyz[0],sy=f.sensor_xyz[1],ground=f.sensor_xyz[2]-1.7,extent=+el('extent').value;
  ctx.strokeStyle='#566a7a55';ctx.lineWidth=1;
  for(let x=Math.floor((sx-extent)/10)*10;x<=sx+extent;x+=10){ctx.beginPath();ctx.moveTo(...projection.world(x,sy-extent,ground));ctx.lineTo(...projection.world(x,sy+extent,ground));ctx.stroke();}
  for(let y=Math.floor((sy-extent)/10)*10;y<=sy+extent;y+=10){ctx.beginPath();ctx.moveTo(...projection.world(sx-extent,y,ground));ctx.lineTo(...projection.world(sx+extent,y,ground));ctx.stroke();}
  const angle=cloud?.meta.sensor_heading_rad??0,c=Math.cos(angle),s=Math.sin(angle);
  ctx.fillStyle='white';ctx.beginPath();
  [[2*c,2*s,0],[-c-s,-s+c,0],[-c+s,-s-c,0]].forEach((p,i)=>i?ctx.lineTo(...projection.relative(p)):ctx.moveTo(...projection.relative(p)));
  ctx.closePath();ctx.fill();
  if(el('det').checked)f.detections.forEach(b=>drawBox(ctx,b,projection,'#b4bcc4',false,''));
  if(el('gt').checked)f.truth.forEach(b=>drawBox(ctx,b,projection,'#ffe08a',false,'G'+b.gt_id+' '+b.label));
  for(const t of f.tracks){
    const color=cssColor(idColor(t.track_id));ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();let connected=false;
    for(let k=Math.max(0,index-20);k<=index;k++){
      const old=history[k].tracks.find(o=>o.track_id===t.track_id);if(!old){connected=false;continue;}
      const p=projection.world(old.x,old.y,old.z);connected?ctx.lineTo(...p):ctx.moveTo(...p);connected=true;
    }
    ctx.stroke();drawBox(ctx,t,projection,color,t.predicted_only,'ID '+t.track_id+' '+t.label+(t.predicted_only?' 预测':''));
  }
  ctx.fillStyle=f.hidden?'#ffb0a0':'#e6edf3';ctx.font='14px system-ui';
  ctx.fillText(f.hidden?'检测被人工隐藏；真实点云仍正常显示':'观测 '+f.tracks.filter(t=>!t.predicted_only).length+' / 预测 '+f.tracks.filter(t=>t.predicted_only).length,14,24);
  ctx.fillText('本帧恢复 '+f.diagnostics.recovered+' / 模糊拒绝 '+f.diagnostics.ambiguous_recoveries,14,46);
  ctx.fillStyle='#d0d8df';ctx.fillText('10 m 网格 · 框内点着色不等于点级真值',14,H-16);
}
async function render(){
  const version=++renderVersion,c=cases[+el('case').value],a=c.profiles.short_memory,b=c.profiles.stable;
  const n=a.frames.length,i=Math.min(+el('frame').value,n-1),basis=viewBasis();
  el('frame').max=n-1;el('frame').value=i;
  el('time').textContent='帧 '+i+' / '+(n-1)+' · '+(i*.1).toFixed(1)+' s';
  el('leftInfo').textContent=metric(a);el('rightInfo').textContent=metric(b);
  el('viewInfo').textContent='俯仰 '+el('elevation').value+'° / 方向 '+el('azimuth').value+'° / 横向范围 '+el('extent').value+' m';
  drawCloud('left',null,a.frames[i],basis);drawCloud('right',null,b.frames[i],basis);
  drawOverlay('left',a.frames[i],a.frames,i,basis,null);drawOverlay('right',b.frames[i],b.frames,i,basis,null);
  if(!el('points').checked){el('cloudInfo').textContent='点云显示关闭';return;}
  if(!cloudMeta){el('cloudInfo').textContent='当前页面没有点云资源；请运行 pointcloud_review 生成';return;}
  el('cloudInfo').textContent='正在读取本帧原始点云…';
  try{
    const cloud=await loadCloud(c.sequence,i);if(version!==renderVersion)return;
    if(!cloud){el('cloudInfo').textContent='官方点云包缺少本帧：'+c.sequence+'/'+String(i).padStart(6,'0')+'；仅显示框，未补造点云';return;}
    for(const f of [a.frames[i],b.frames[i]])if(f.sensor_xyz.some((v,j)=>Math.abs(v-cloud.meta.sensor_xyz[j])>.001))
      throw new Error('点云与检测框的车辆位姿不一致，已拒绝叠加');
    const stats=drawCloud('left',cloud,a.frames[i],basis);drawCloud('right',cloud,b.frames[i],basis);
    drawOverlay('left',a.frames[i],a.frames,i,basis,cloud);drawOverlay('right',b.frames[i],b.frames,i,basis,cloud);
    el('cloudInfo').textContent='原始 '+cloud.meta.original_points.toLocaleString()+' 点 / 本次绘制 '+stats.points.toLocaleString()+
      (stats.fallback?' · 兼容模式，最多 3 万点':' · '+(el('fullPoints').checked?'全量显示':'采样预览，可勾选全量'))+
      ' · 同帧 '+c.sequence+'/'+String(i).padStart(6,'0')+' · 坐标仅显示时量化到 1 cm';
    if(i+1<n)loadCloud(c.sequence,i+1).catch(()=>{});
  }catch(error){if(version===renderVersion)el('cloudInfo').textContent='点云显示失败：'+error.message;}
}
cases.forEach((c,i)=>{
  const o=document.createElement('option');o.value=i;
  const interval=c.profiles.stable.hidden_interval;
  o.textContent=c.sequence+' / '+(c.scenario==='natural'?'自然检测':
    c.scenario.startsWith('stress_start_')?'失败案例：中断起始帧 '+c.scenario.slice(13):'3 秒中断：起始帧 '+interval.start_frame);
  el('case').append(o);
});
function stop(){clearTimeout(timer);timer=null;playVersion++;el('play').textContent='播放';}
async function advance(token){
  if(timer===null||token!==playVersion)return;
  const start=performance.now(),v=+el('frame').value;
  if(v>=+el('frame').max){stop();return;}
  el('frame').value=v+1;await render();
  if(timer!==null&&token===playVersion)timer=setTimeout(()=>advance(token),Math.max(1,100-(performance.now()-start)));
}
el('play').onclick=()=>{if(timer!==null)stop();else{el('play').textContent='暂停';const token=++playVersion;timer=setTimeout(()=>advance(token),100);}};
el('jump').onclick=()=>{stop();const frames=cases[+el('case').value].profiles.stable.frames;
  const first=frames.findIndex(f=>f.hidden);let end=first<0?0:first;while(end<frames.length&&frames[end].hidden)end++;
  el('frame').value=Math.min(end,frames.length-1);render();};
el('topView').onclick=()=>{el('elevation').value=90;el('azimuth').value=0;render();};
el('obliqueView').onclick=()=>{el('elevation').value=55;el('azimuth').value=0;render();};
el('case').onchange=()=>{stop();el('frame').value=0;render();};
for(const id of ['frame','gt','det','points','fullPoints','pointColor','pointSize','elevation','azimuth','extent'])
  el(id).addEventListener('input',()=>render());
el('points').checked=!!cloudMeta;el('elevation').value=cloudMeta?55:90;
render();
