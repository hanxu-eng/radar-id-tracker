"""Fetch selected original KITTI tracking scans with HTTP ZIP range reads.

The official archive is about 35 GiB. Read its ZIP64 directory, then download
only existing replay sequences; never fetch or extract unrelated sequences.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

URL='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip'
MAX_RANGE=32*1024*1024


class HttpRangeFile(io.RawIOBase):
    def __init__(self,url=URL):
        self.url=url; self.position=0
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as response:
            self.size=int(response.headers['Content-Length']); self.etag=response.headers['ETag']

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self,offset,whence=0):
        new=(0 if whence==0 else self.position if whence==1 else self.size)+offset
        if not 0 <= new <= self.size: raise ValueError('invalid remote ZIP seek')
        self.position=new; return new

    def fetch(self,start,length):
        if not 0 <= start <= self.size or not 0 <= length <= min(MAX_RANGE,self.size-start):
            raise ValueError('invalid HTTP range length')
        if not length: return b''
        request=urllib.request.Request(self.url,headers={
            'Range':f'bytes={start}-{start+length-1}','If-Match':self.etag,
            'User-Agent':'radar-id-tracker-pointcloud-review'})
        last_error=None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request,timeout=60) as response:
                    expected=f'bytes {start}-{start+length-1}/{self.size}'
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('server did not honor byte range; refusing full archive download')
                    content=response.read(length+1)
                    if len(content)!=length: raise OSError('incomplete HTTP range')
                    return content
            except OSError as error:
                last_error=error
        raise last_error

    def read(self,length=-1):
        length=self.size-self.position if length<0 else min(length,self.size-self.position)
        data=self.fetch(self.position,length);self.position+=len(data);return data


def extract_member(info,chunk,chunk_offset):
    offset=info.header_offset-chunk_offset
    header=chunk[offset:offset+30]
    if len(header)!=30 or header[:4]!=b'PK\x03\x04': raise ValueError('invalid ZIP local header')
    flags,method=struct.unpack_from('<HH',header,6)
    name_length,extra_length=struct.unpack_from('<HH',header,26)
    name=chunk[offset+30:offset+30+name_length].decode('utf-8' if flags&2048 else 'cp437')
    if name!=info.filename or method!=info.compress_type or flags&1:
        raise ValueError('unexpected ZIP member header')
    start=offset+30+name_length+extra_length
    compressed=chunk[start:start+info.compress_size]
    if len(compressed)!=info.compress_size: raise ValueError('truncated ZIP member payload')
    if method==zipfile.ZIP_STORED: data=compressed
    elif method==zipfile.ZIP_DEFLATED: data=zlib.decompress(compressed,-15)
    else: raise ValueError(f'unsupported ZIP compression {method}')
    if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:
        raise ValueError('ZIP scan size/CRC mismatch')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('datasets/kitti_tracking'))
    parser.add_argument('--sequences',nargs='+',default=['0000','0001','0005'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error('workers must be 1..8')
    if any(len(s)!=4 or not s.isdecimal() for s in args.sequences): parser.error('invalid sequence ID')
    remote=HttpRangeFile()
    with zipfile.ZipFile(remote) as archive:
        directory={info.filename:info for info in archive.infolist()}
    selected=[];records=[];missing=[]
    for sequence in args.sequences:
        count=len((args.dataset_root/'training/oxts'/f'{sequence}.txt').read_text().splitlines())
        for frame in range(count):
            name=f'training/velodyne/{sequence}/{frame:06d}.bin'
            if name not in directory:
                missing.append(name)
                continue
            info=directory[name]
            if info.file_size%16 or not 0 < info.file_size <= 8*1024*1024:
                raise ValueError(f'invalid scan size: {name}')
            target=args.dataset_root/name
            if target.exists() and target.stat().st_size==info.file_size:
                data=target.read_bytes()
                if zlib.crc32(data)&0xffffffff==info.CRC:
                    records.append(dict(member=name,bytes=len(data),crc32=info.CRC,
                                        sha256=hashlib.sha256(data).hexdigest()))
                    continue
            selected.append(info)
    selected.sort(key=lambda info:info.header_offset)
    # Batch contiguous scans. Each group has a small header allowance and stays
    # below MAX_RANGE; selected paths alone are ever written.
    groups=[]
    for info in selected:
        end=min(remote.size,info.header_offset+info.compress_size+65536)
        if (not groups or info.header_offset-groups[-1][1]>65536 or end-groups[-1][0]>MAX_RANGE):
            groups.append([info.header_offset,end,[info]])
        else:
            groups[-1][1]=max(groups[-1][1],end);groups[-1][2].append(info)
    print(f'archive {remote.size/1024**3:.2f} GiB; selected {len(selected)} scans '
          f'{sum(i.file_size for i in selected)/1024**3:.2f} GiB; {len(groups)} ranges',flush=True)

    def fetch_group(group):
        start,end,infos=group;chunk=remote.fetch(start,end-start);result=[]
        for info in infos:
            data=extract_member(info,chunk,start)
            target=args.dataset_root/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
            part=target.with_suffix('.bin.part');part.write_bytes(data);part.replace(target)
            result.append(dict(member=info.filename,bytes=len(data),crc32=info.CRC,
                               sha256=hashlib.sha256(data).hexdigest()))
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(fetch_group,group) for group in groups]
        for completed,task in enumerate(as_completed(tasks),1):
            records.extend(task.result())
            print(f'ranges {completed}/{len(groups)}, verified scans {len(records)}',flush=True)
    manifest=dict(source_url=URL,archive_bytes=remote.size,archive_etag=remote.etag,
                  raw_lidar_points_downloaded=True,format='little-endian float32 x,y,z,reflectance',
                  dataset_license='CC BY-NC-SA 3.0',sequences=args.sequences,
                  missing_archive_members=missing,
                  scans=sorted(records,key=lambda row:row['member']))
    (args.dataset_root/'pointcloud_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    existing=args.dataset_root/'manifest.json'
    if existing.exists():
        overview=json.loads(existing.read_text())
        overview['raw_lidar_points_downloaded']=True
        overview['pointcloud_manifest']='pointcloud_manifest.json'
        existing.write_text(json.dumps(overview,indent=2)+'\n')


if __name__=='__main__':main()
