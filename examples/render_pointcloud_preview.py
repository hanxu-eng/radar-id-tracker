"""Optional Matplotlib snapshot of original scans and tracked 3D boxes."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/kitti_tracking'))
    p.add_argument('--run-root',type=Path,default=Path('outputs/kitti_final'))
    p.add_argument('--sequence',default='0000');p.add_argument('--frame',type=int,default=81)
    p.add_argument('--scenario',default='dropout_3s')
    p.add_argument('--output',type=Path,default=Path('outputs/kitti_final/pointcloud_preview.png'))
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(args.output.parent.resolve()/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from mpl_toolkits.mplot3d import proj3d

    poses=load_poses(args.dataset_root/'training/oxts'/f'{args.sequence}.txt')
    _,lidar_to_imu=calibration_transforms(args.dataset_root/'training/calib'/f'{args.sequence}.txt')
    transform=poses[args.frame]@lidar_to_imu
    original=read_scan(args.dataset_root/'training/velodyne'/args.sequence/f'{args.frame:06d}.bin')
    xyz=original[:,:3]@transform[:3,:3].T
    crop=(np.abs(xyz[:,0]-5)<20)&(np.abs(xyz[:,1]+10)<20)&(xyz[:,2]>-2.5)&(xyz[:,2]<5)
    cloud=xyz[crop][::2]
    plt.rcParams.update({'font.size':10,'text.color':'#e6edf3','axes.labelcolor':'#b1becb',
                         'xtick.color':'#b1becb','ytick.color':'#b1becb'})
    fig=plt.figure(figsize=(13,6),facecolor='#101821',layout='constrained')
    edge_indices=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    evidence=[]
    stable_eval=json.loads((args.run_root/args.sequence/args.scenario/'stable/evaluation.json').read_text())
    recovery=stable_eval.get('dropout_recovery')
    reference=recovery['targets'][0] if recovery and recovery['targets'] else None
    for panel,profile in enumerate(('short_memory','stable'),1):
        run=args.run_root/args.sequence/args.scenario/profile
        f=json.loads((run/'tracks.jsonl').read_text().splitlines()[args.frame])
        ax=fig.add_subplot(1,2,panel,projection='3d',facecolor='#101821')
        ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],s=.45,c='#758ca3',alpha=.6,depthshade=False,rasterized=True)
        annotations=[]
        for t in f['tracks']:
            center=np.array([t['x'],t['y'],t['z']])-transform[:3,3]
            c=np.cos(t['yaw']);s=np.sin(t['yaw']);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            local=(xyz-center)@rotation
            inside=(np.abs(local)<=np.array([t['length'],t['width'],t['height']])/2).all(axis=1)
            color=colorsys.hls_to_rgb((t['track_id']*137.508%360)/360,.68,.75)
            object_points=xyz[inside]
            ax.scatter(object_points[:,0],object_points[:,1],object_points[:,2],s=2,c=[color],depthshade=False,rasterized=True)
            corners=np.array([[u*t['length']/2,v*t['width']/2,z*t['height']/2]
                              for z in (-1,1) for u,v in ((1,1),(-1,1),(-1,-1),(1,-1))])@rotation.T+center
            ax.add_collection3d(Line3DCollection([[corners[a],corners[b]] for a,b in edge_indices],colors=[color],linewidths=1.8,
                                                linestyles='dashed' if t['predicted_only'] else 'solid'))
            annotations.append((center+np.array([0,0,t['height']/2]),
                                f"ID {t['track_id']} | {t['label']}",color))
            evidence.append(dict(profile=profile,track_id=t['track_id'],inside_raw_points=int(inside.sum())))
        ax.scatter([0],[0],[0],c='white',marker='^',s=35)
        ax.set(xlim=(-15,25),ylim=(-30,10),zlim=(-2.5,5),xlabel='World X relative to LiDAR (m)',
               ylabel='World Y relative to LiDAR (m)',zlabel='Z (m)')
        ax.set_box_aspect((40,40,12));ax.view_init(elev=35,azim=-65)
        for number,(location,label,color) in enumerate(annotations):
            x,y,_=proj3d.proj_transform(*location,ax.get_proj())
            ax.annotate(label,(x,y),xytext=(.62,.85-number*.14),textcoords='axes fraction',color=color,
                        fontsize=12,fontweight='bold',zorder=1000,
                        bbox=dict(boxstyle='round,pad=.4',facecolor='#101821',edgecolor=color),
                        arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.set_facecolor('#172330');axis.pane.set_edgecolor('#304356')
            axis._axinfo['grid']['color']='#304356'
        ax.tick_params(colors='#b1becb')
        details=json.loads((run/'evaluation_frames.jsonl').read_text().splitlines()[args.frame])
        pid=details['observed_tracks']['matches'].get(reference['gt_id']) if reference else None
        title=('1.5 s' if profile=='short_memory' else '4.0 s')+' retention'
        if pid is not None:
            title+=f': ID {pid} '+('restored' if pid==reference['before_id'] else f"(previous ID {reference['before_id']})")
        else:title+=': no observed match to reference target'
        ax.set_title(title,color='#e6edf3',pad=12)
    fig.suptitle(f'Original KITTI scan + 3D tracking | Sequence {args.sequence}, frame {args.frame} ({args.frame*.1:.1f} s)\n'
                 'Same raw scan in both panels; boxes and point colors come from the tracker',color='#e6edf3',fontsize=13)
    fig.savefig(args.output,dpi=150,facecolor=fig.get_facecolor())
    (args.output.with_suffix('.json')).write_text(json.dumps(dict(original_points=len(original),preview_points=len(cloud),box_point_counts=evidence),indent=2)+'\n')
    print(args.output);print(evidence)


if __name__=='__main__':main()
