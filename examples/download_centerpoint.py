"""Download the single supported official checkpoint with a pinned SHA256.

Uses only standard-library HTTP; does not import PyTorch or execute model code.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

NAME='centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
SHA256='191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d'
URL=('https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/'
     'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/'+NAME)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/models/centerpoint'))
    p.add_argument('--references',action='store_true',help='download hash-pinned source classes for optional tensor audit')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/NAME
    if not target.exists():
        partial=target.with_suffix('.part')
        try:
            with urlopen(URL,timeout=120) as source,partial.open('wb') as destination:
                size=0
                while data:=source.read(1024*1024):
                    size+=len(data)
                    if size>30*1024*1024:raise ValueError('checkpoint exceeds expected size bound')
                    destination.write(data)
            if hashlib.sha256(partial.read_bytes()).hexdigest()!=SHA256:
                raise ValueError('checkpoint checksum mismatch')
            partial.replace(target)
        finally:partial.unlink(missing_ok=True)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('existing checkpoint checksum mismatch; refusing to overwrite')
    manifest=dict(url=URL,path=str(target.resolve()),sha256=SHA256,bytes=target.stat().st_size,
        architecture='CenterPoint pillar02 SECOND SECFPN circle-NMS',training_dataset='nuScenes',
        source_config='https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py')
    (args.output_dir/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.references:
        # Load hashes as data, without importing the GPU verification script.
        import ast
        source=ast.parse(Path(__file__).with_name('verify_centerpoint_port.py').read_text())
        hashes=next(ast.literal_eval(node.value) for node in source.body if isinstance(node,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='HASHES' for t in node.targets))
        for name,expected in hashes.items():
            url='https://raw.githubusercontent.com/open-mmlab/mmdetection3d/fe25f7a51d36e3702f961e198894580d83c4387b/'+name
            with urlopen(url,timeout=60) as response:data=response.read(1024*1024)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('reference checksum mismatch: '+name)
            destination=args.output_dir/'reference'/name
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
        manifest['reference_commit']='fe25f7a51d36e3702f961e198894580d83c4387b'
        manifest['reference_sha256']=hashes
        (args.output_dir/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
