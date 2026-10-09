"""Offline review of original scans, box detections, identities and GT."""
import json
from pathlib import Path


def build_review(root: Path, rows: list[dict]) -> Path:
    box_keys=('x','y','z','length','width','height','yaw','label','track_id','predicted_only','gt_id')
    # Display precision only; scoring JSONL preserves full numeric precision.
    def display_box(box):
        return {k:round(v,4) if isinstance(v,float) else v for k,v in box.items() if k in box_keys}
    cases = []
    for sequence,scenario in sorted({(r['sequence'],r['scenario']) for r in rows}):
        profiles = {}
        for row in rows:
            if (row['sequence'],row['scenario']) != (sequence,scenario):
                continue
            directory = Path(row.get('run_dir',root/sequence/scenario/row['profile']))
            read = lambda name: [json.loads(line) for line in (directory/name).read_text().splitlines()]
            detections,tracks,details = read('detections.jsonl'),read('tracks.jsonl'),read('evaluation_frames.jsonl')
            frames = [dict(sensor_xyz=[round(v,4) for v in d['sensor_xyz']],
                           detections=[display_box(b) for b in d['detections']],
                           tracks=[display_box(b) for b in t['tracks']],diagnostics=t['diagnostics'],
                           truth=[display_box(b) for b in e['truth']],
                           matches=e['observed_tracks']['matches'],hidden=d['synthetic_detector_dropout'])
                      for d,t,e in zip(detections,tracks,details)]
            profiles[row['profile']] = dict(frames=frames,score=row['evaluation'],timing=row['run']['p99_tracker_ms'],
                                           hidden_interval=row['run']['hidden_interval'])
        cases.append(dict(sequence=sequence,scenario=scenario,profiles=profiles))
    payload = json.dumps(cases,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    stress_note = ''
    if (root/'stress_grid.json').exists():
        stress=json.loads((root/'stress_grid.json').read_text())
        p=stress['paired_totals']; t=stress['totals']['stable']
        stress_note=(f'<p>多位置压力测试：{t["trials"]} 个中断位置，{p["eligible"]} 次共同可评恢复机会'
                     f'（{p["unique_physical_targets"]} 个不同目标）。同 ID 恢复：短时保留 '
                     f'{p["short_same_id"]}/{p["eligible"]}，4 秒保留 {p["stable_same_id"]}/{p["eligible"]}。'
                     f'稳定方案换 ID {t["changed_id"]} 次，其中误接其他旧身份 {t["wrong_retained_identity"]} 次；'
                     f'返回检测存在但未输出匹配轨迹 {t["no_track_after_return"]} 次。下拉框包含失败案例。</p>')
    cloud_path=root/'pointclouds/manifest.json'
    cloud=json.loads(cloud_path.read_text()) if cloud_path.exists() else None
    cloud_payload=json.dumps(cloud,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    script=Path(__file__).with_name('lidar_review.js').read_text(encoding='utf-8')
    if cloud and (root/'review.html').exists() and not (root/'review_boxes_only.html').exists():
        (root/'review_boxes_only.html').write_bytes((root/'review.html').read_bytes())
    html=Path(__file__).with_name('lidar_review.html').read_text(encoding='utf-8')
    html=html.replace('CLOUD_PAYLOAD',cloud_payload).replace('PAYLOAD',payload)
    html=html.replace('STRESS_NOTE',stress_note).replace('MAIN_JS',script)
    path = root/'review.html'; path.write_text(html,encoding='utf-8')
    return path


def add_stress_reviews(root: Path, rows: list[dict]) -> Path:
    """Include every failed grid recovery, alongside natural and first tests."""
    stress=json.loads((root/'stress_grid.json').read_text())
    selected={(r['sequence'],r['start_frame']) for r in stress['trials']
              if r['profile']=='stable' and r['recovery']['same_id']<r['recovery']['eligible']}
    reviews=list(rows)
    for sequence,start in sorted(selected):
        for profile in ('short_memory','stable'):
            directory=root/'stress_grid'/sequence/f'start_{start}'/profile
            reviews.append(dict(sequence=sequence,scenario=f'stress_start_{start}',profile=profile,
                                run=json.loads((directory/'summary.json').read_text()),
                                evaluation=json.loads((directory/'evaluation.json').read_text()),
                                run_dir=str(directory)))
    return build_review(root,reviews)
