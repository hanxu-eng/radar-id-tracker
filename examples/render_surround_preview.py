"""Scientific full-surround BEV snapshot from original points and saved tracks."""
import argparse
import colorsys
import json
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from radar_id_tracker.kitti_adapter import calibration_transforms,load_poses
from radar_id_tracker.pointcloud_review import read_scan
from radar_id_tracker.box_geometry import corners_xy,iou3d
from radar_id_tracker.kitti_evaluate import load_truth
from radar_id_tracker.lidar_tracker import BoxTrack,BoxDetection
from radar_id_tracker.lidar_scoped_evaluate import in_scope


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--sequence',default='0001');p.add_argument('--frame',type=int,default=120)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--comparison',choices=('detector','input','quality','heads','coverage'),default='detector')
    p.add_argument('--show-errors',action='store_true',help='offline front-ROI truth and TP/FP/missed coloring')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.output.parent.resolve()/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    poses=load_poses(args.dataset_root/'training/oxts'/f'{args.sequence}.txt')
    _,lidar_to_imu=calibration_transforms(args.dataset_root/'training/calib'/f'{args.sequence}.txt')
    inverse=np.linalg.inv(poses[args.frame] @ lidar_to_imu)
    ignored=[]
    if args.show_errors:
        ignored=[box for box in load_truth(args.dataset_root,args.sequence)[1][args.frame] if in_scope(box,inverse)]
    points=read_scan(args.dataset_root/'training/velodyne'/args.sequence/f'{args.frame:06d}.bin')
    displayed=points[(np.abs(points[:,0])<51.2)&(np.abs(points[:,1])<51.2)]
    plt.rcParams.update({'text.color':'#e6edf3','axes.labelcolor':'#b1becb',
                        'xtick.color':'#b1becb','ytick.color':'#b1becb','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(15,8),layout='constrained',facecolor='#101821')
    evidence=[]
    titles=('CenterPoint original input','CenterPoint adapted input (experimental)') if args.comparison=='input' else ('Published PointRCNN + tracker','Raw CenterPoint + tracker (experimental)')
    if args.comparison=='quality':titles=('CenterPoint original scores','Current-scan box quality (experimental)')
    if args.comparison=='heads':titles=('CenterPoint original heads','Front head adaptation (experimental)')
    if args.comparison=='coverage':titles=('CenterPoint original heads','Expanded training coverage (experimental)')
    rows=json.loads((args.run_root/'benchmark.json').read_text())
    for ax,profile,title in zip(axes,('baseline','candidate'),titles):
        ax.set_facecolor('#172330');ax.axvspan(-51.2,0,color='#25314d',alpha=.5)
        ax.scatter(displayed[:,0],displayed[:,1],s=.15,c='#7a91a5',alpha=.45,rasterized=True)
        metadata=next(r for r in rows if r['sequence']==args.sequence and r['scenario']=='natural' and r['profile']==profile)
        run=Path(metadata.get('run_dir',args.run_root/args.sequence/'natural'/profile))
        frame=json.loads((run/'tracks.jsonl').read_text().splitlines()[args.frame])
        row=json.loads((run/'detections.jsonl').read_text().splitlines()[args.frame])
        score=json.loads((run/'evaluation.json').read_text())['observed_tracks']
        detail=json.loads((run/'evaluation_frames.jsonl').read_text().splitlines()[args.frame])
        matched=detail['observed_tracks']['matches']
        tracks=[BoxTrack(**t) for t in frame['tracks']]
        annotations=[];rear=0
        for track in tracks:
            center=inverse @ [track.x,track.y,track.z,1.]
            if (np.abs(center[:2])>51.2).any():continue
            rear+=int(center[0]<0)
            world_corners=np.column_stack([corners_xy(track),np.full(4,track.z),np.ones(4)])
            xy=(world_corners @ inverse.T)[:,:2]
            color=colorsys.hls_to_rgb((track.track_id*137.508%360)/360,.7,.75)
            status='rear/unscored' if center[0]<0 else 'unscored'
            if in_scope(track,inverse) and not track.predicted_only:
                status='TP' if track.track_id in matched.values() else 'FP'
                if status=='FP' and any(track.label==box.label and iou3d(track,box)>=.25 for box in ignored):status='IGNORED'
                if args.show_errors:color='#62e6a7' if status=='TP' else '#ff7676'
                if args.show_errors and status=='IGNORED':color='#b1becb'
            ax.add_patch(Polygon(xy,closed=True,fill=False,edgecolor=color,linewidth=1.6,
                                 linestyle='--' if track.predicted_only else '-'))
            local=(points[:,:3] @ np.linalg.inv(inverse)[:3,:3].T+np.linalg.inv(inverse)[:3,3]-[track.x,track.y,track.z])
            c,s=np.cos(track.yaw),np.sin(track.yaw)
            local=local @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
            inside=(np.abs(local)<=np.array([track.length,track.width,track.height])/2).all(1)
            colored=points[inside]
            ax.scatter(colored[:,0],colored[:,1],s=.7,c=[color],rasterized=True)
            annotations.append((track,center,color,status))
            evidence.append(dict(profile=profile,track_id=track.track_id,label=track.label,
                confidence=track.confidence,lidar_center=center[:3].tolist(),raw_points_in_box=int(inside.sum()),
                rear=bool(center[0]<0),predicted_only=track.predicted_only,diagnostic_status=status))
        if args.show_errors:
            missed=[]
            for truth in detail['truth']:
                box=BoxDetection(**{k:v for k,v in truth.items() if k!='gt_id'})
                corners=np.column_stack([corners_xy(box),np.full(4,box.z),np.ones(4)])
                xy=(corners @ inverse.T)[:,:2]
                ax.add_patch(Polygon(xy,closed=True,fill=False,edgecolor='#ffe08a',linewidth=1,linestyle=':'))
                if truth['gt_id'] not in matched:
                    center=inverse @ [box.x,box.y,box.z,1.]
                    missed.append((truth['gt_id'],center))
            for number,(gt_id,center) in enumerate(missed):
                # Place missed-target labels in separate rows so nearby GT remain readable.
                ax.annotate(f'MISS G{gt_id}',center[:2],xytext=(5,-33-number*3),
                    color='#ffe08a',fontsize=8,
                    arrowprops=dict(arrowstyle='-',color='#ffe08a',alpha=.6,lw=.6))
        # Limit labels only; every saved in-range track box is still drawn.
        for rear_side in (True,False):
            selected=sorted([item for item in annotations if bool(item[1][0]<0)==rear_side],
                            key=lambda item:-item[0].confidence)[:5]
            for number,(track,center,color,status) in enumerate(selected):
                text=f'ID {track.track_id} {track.label} {track.confidence:.2f}'
                if args.show_errors:text+=' '+status
                if track.predicted_only:text+=' (predicted)'
                ax.annotate(text,center[:2],xytext=(-49 if rear_side else 8,47-number*5),
                    color=color,fontsize=8,bbox=dict(facecolor='#101821',edgecolor=color,alpha=.9,pad=2),
                    arrowprops=dict(arrowstyle='-',color=color,alpha=.5,lw=.7))
        ax.scatter([0],[0],c='white',s=35,marker='>');ax.axvline(0,color='#6981a0',linewidth=.7)
        for sign in (-1,1):ax.plot([0,50*np.cos(np.deg2rad(40))],[0,sign*50*np.sin(np.deg2rad(40))],color='#ffe08a',alpha=.5,linestyle=':')
        ax.text(-49,-49,'REAR: no KITTI ground truth',color='#c3d0eb',fontsize=10)
        ax.set(xlim=(-51.2,51.2),ylim=(-51.2,51.2),aspect='equal',xlabel='KITTI LiDAR X / forward (m)',ylabel='KITTI LiDAR Y / left (m)')
        ax.grid(alpha=.12);ax.set_title(title+f'\nSequence front P {score["precision"]:.1%} / R {score["recall"]:.1%}; rear displayed tracks: {rear}',pad=10)
    role_note=''
    if args.comparison in ('heads','coverage'):
        role=next(r['role'] for r in json.loads((args.run_root/'benchmark.json').read_text()) if r['sequence']==args.sequence)
        role_note=' | TRAINING REPLAY' if role=='training_replay' else ' | HELD-OUT VALIDATION'
    fig.suptitle(f'Original KITTI pointcloud + stable IDs | sequence {args.sequence}, frame {args.frame}{role_note}\n'+
        ('Front 3D IoU >= 0.25: green TP / red FP / yellow truth and MISS / gray ignored; rear unscored' if args.show_errors else
         'Front scores: Car/Pedestrian, +/-40 deg and 50m, local 3D IoU >= 0.25; rear boxes need independent verification'),fontsize=12)
    fig.savefig(args.output,dpi=150,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.12)
    args.output.with_suffix('.json').write_text(json.dumps(dict(sequence=args.sequence,frame=args.frame,
        raw_points=len(points),displayed_points=len(displayed),boxes=evidence),indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
