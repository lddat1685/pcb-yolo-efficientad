"""Disposable implementer smoke checks, independent of VERIFIER's tests/."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
from pcb_lab.data.audit import run_audit
from pcb_lab.data.eda import render_overlays
from pcb_lab.data.manifest import DatasetInputError, PARTITIONS, METADATA_PATHS, sha256_file

workspace = Path(__file__).resolve().parents[2]
base = workspace / 'reports/verification'
names = ['open_circuit', 'short', 'mouse_bite', 'spur', 'spurious_copper', 'pin_hole']
checks = []


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def fixture(case):
    root = base / case / 'dataset'
    rows = []
    holdout = {'partitions': list(PARTITIONS), 'samples': {}, 'groups': {}, 'pair_ids': {}, 'actual_pair_counts': {}}
    for i, p in enumerate(PARTITIONS):
        sid = f'{p}_defect'
        pair = f'pair_{i}'
        group = f'group_{i}'
        image_rel = f'images/{sid}.png'
        label_rel = f'labels/{sid}.txt'
        image_path = root / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new('RGB', (640, 640), (20 + i * 30, 40 + i, 60 + i)) as image:
            pixel = hashlib.sha256(image.tobytes()).hexdigest()
            image.save(image_path)
        label_path = root / label_rel
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(''.join(f'{c} 0.5 0.5 0.05 0.05\n' for c in range(6)))
        rows.append({'sample_id': sid, 'split': p, 'source_group': group, 'pair_ids': [pair], 'is_defect': True,
                     'width': 640, 'height': 640, 'image': image_rel, 'label': label_rel,
                     'sha256': sha256_file(image_path), 'label_sha256': sha256_file(label_path), 'pixel_sha256': pixel,
                     'boxes': [{'class_id': c, 'class_name': name, 'source_class_id': c+1,
                                'xyxy': [304, 304, 336, 336]} for c, name in enumerate(names)]})
        holdout['samples'][p] = [sid]
        holdout['groups'][p] = [group]
        holdout['pair_ids'][p] = [pair]
        holdout['actual_pair_counts'][p] = 1
    write_json(root / METADATA_PATHS['holdout'], holdout)
    write_json(root / METADATA_PATHS['classes'], {'names': names, 'source_to_class_id': {str(c+1): c for c in range(6)}})
    write_json(root / METADATA_PATHS['protocol'], {'pixel_metrics': None})
    save_rows(root, rows)
    lock(root)
    return root, rows, holdout


def save_rows(root, rows):
    path = root / METADATA_PATHS['manifest']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def lock(root):
    write_json(root / METADATA_PATHS['release'], {'files': {v: sha256_file(root/v) for k,v in METADATA_PATHS.items() if k != 'release'}})


def audit(root):
    return run_audit(root, root.parent / 'out')


def status(report, key):
    return next(c['status'] for c in report['checks'] if c['id'] == key)


root, rows, holdout = fixture('valid')
r = audit(root)
assert r['overall'] == 'PASS' and r['errors'] == []
assert len(r['leakage']['pairs']) == 6
assert run_audit(root, root.parent/'repeat') == r
checks.append('valid four-partition fixture, six pairs, exact repeat')
for args in [('test',), ('fusion',), ('train', 'test')]:
    try:
        render_overlays(Path('nonexistent'), root.parent/'overlays', args)
        raise AssertionError('Forbidden overlay did not raise')
    except ValueError:
        pass
checks.append('forbidden overlay partitions rejected before metadata I/O')

original_open = Image.open
def guard_open(path, *args, **kwargs):
    assert Path(path).name not in {'test_defect.png', 'fusion_defect.png'}
    return original_open(path, *args, **kwargs)
Image.open = guard_open
try:
    paths = render_overlays(root, root.parent/'overlays', per_class=1)
    assert len(paths) == 12
finally:
    Image.open = original_open
checks.append('overlay Image.open spy: only train/calibration, six classes each')

root, rows, holdout = fixture('corrupt')
(root / rows[0]['image']).write_bytes(b'broken image')
(root / rows[1]['label']).write_text('99 0.5 0.5 0.2 0.2\n1 nan 0.5 0.2 0.2\n1 0.1 0.1 -0.1 0.1\n')
(root / rows[2]['label']).unlink()
r = audit(root)
assert r['counts']['total']['images'] == 4
assert r['class_box_counts']['test']['pin_hole'] == 1
for key in ('image_decode','image_sha256','leak_pixel_hash','class_id_valid','box_valid','manifest_paths_exist','label_defect_nonempty'):
    assert status(r,key) == 'FAIL', key
assert r['overall'] == 'FAIL' and all({'sample_id','stage','message'} <= e.keys() for e in r['errors'])
checks.append('corrupt image, bad IDs/NaN/negative boxes, missing label: FAIL and continue')

root, rows, holdout = fixture('leakage')
for row in rows[1:]:
    row['source_group'] = rows[0]['source_group']
    row['pair_ids'] = rows[0]['pair_ids']
    (root/row['image']).write_bytes((root/rows[0]['image']).read_bytes())
save_rows(root,rows)
r = audit(root)
for pair in r['leakage']['pairs']:
    assert all(pair[k] == 1 for k in ('source_group','pair','image_hash','pixel_hash'))
assert status(r, 'release_sha256') == 'FAIL'
assert status(r, 'partition_counts') == 'FAIL'
checks.append('all six pairs detect four leakage types using actual bytes, plus release/membership failures')

root, rows, holdout = fixture('schema')
rows[0]['pair_ids'] = None
rows[0]['boxes'] = [None, {'class_id': 0, 'class_name': 'open_circuit', 'xyxy': [0,0,float('nan'),3]}]
rows[1]['is_defect'] = 'false'
rows[1]['image'] = '../outside.png'
rows.append(None)
save_rows(root, rows)
r = audit(root)
assert status(r, 'manifest_schema') == 'FAIL'
assert status(r, 'manifest_paths_exist') == 'FAIL'
assert r['counts']['total']['images'] == 5
checks.append('malformed individual records and escaping path fail without aborting')

try:
    run_audit(root, root/'reports')
    raise AssertionError('dataset write guard failed')
except DatasetInputError:
    pass
checks.append('dataset output guard')

command = [sys.executable, str(workspace/'scripts/audit_dataset.py')]
valid_root = base/'valid/dataset'
for label, arguments, expected in [
    ('CLI PASS with WARN only', ['--dataset-root',str(valid_root),'--out-dir',str(base/'cli-pass'),'--no-overlays'], 0),
    ('CLI PASS with fewer than five overlays available', ['--dataset-root',str(valid_root),'--out-dir',str(base/'cli-overlay')], 0),
    ('CLI FAIL', ['--dataset-root',str(base/'corrupt/dataset'),'--out-dir',str(base/'cli-fail'),'--no-overlays'], 1),
    ('CLI missing metadata', ['--dataset-root',str(base/'absent'),'--out-dir',str(base/'cli-absent')], 2),
]:
    proc = subprocess.run(command+arguments, capture_output=True, text=True)
    assert proc.returncode == expected, (label, proc.returncode, proc.stderr)
    checks.append(label)
root, _, _ = fixture('bad-json')
(root/METADATA_PATHS['manifest']).write_text('{broken json\n')
proc = subprocess.run(command+['--dataset-root',str(root),'--out-dir',str(base/'cli-json')],capture_output=True,text=True)
assert proc.returncode == 2
checks.append('CLI malformed JSONL returns 2')
write_json(base/'self_check_results.json', {'passed': checks, 'count': len(checks)})
print(json.dumps({'passed':len(checks),'checks':checks}, indent=2))
