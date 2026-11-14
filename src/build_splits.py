"""Create reproducible, content-disjoint splits without copying the raw images."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, io, json, random, re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/processed/split_manifest.csv'
SEED = 42
CLASS_NAMES = ('cat', 'dog')


def build_splits(seed=SEED):
    inspection = json.loads((ROOT / 'reports/data_summary.json').read_text())
    raw = ROOT / inspection['image_directory']
    bad = {x['file'] for x in inspection['corrupt_images']}
    paths = sorted(p for p in raw.rglob('*.jpg') if '__MACOSX' not in p.parts and not p.name.startswith('._') and re.fullmatch(r'(cat|dog)\.\d+\.jpg', p.name, re.I) and str(p.relative_to(ROOT)) not in bad)
    groups = defaultdict(list)
    for i, path in enumerate(paths, 1):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        label = CLASS_NAMES.index(path.name.split('.')[0].lower())
        groups[digest].append({'path': str(path.relative_to(ROOT)), 'label': label, 'class_name': CLASS_NAMES[label], 'sha256': digest})
        if i % 5000 == 0:
            print(f'Hashed {i}/{len(paths)} images', flush=True)
    conflicts = [digest for digest, rows in groups.items() if len({r['label'] for r in rows}) != 1]
    if conflicts:
        raise ValueError('Identical images have conflicting labels; resolve before splitting.')
    rows = []
    for label in range(2):
        keys = sorted(k for k, values in groups.items() if values[0]['label'] == label)
        random.Random(seed + label).shuffle(keys)
        n_train, n_val = int(len(keys) * 0.70), int(len(keys) * 0.15)
        for split, selected in [('train', keys[:n_train]), ('validation', keys[n_train:n_train+n_val]), ('test', keys[n_train+n_val:])]:
            for digest in selected:
                rows.extend(dict(row, split=split) for row in groups[digest])
    rows.sort(key=lambda r: (r['split'], r['label'], r['path']))
    by_split = {split: [r for r in rows if r['split'] == split] for split in ('train', 'validation', 'test')}
    path_sets = {s: {r['path'] for r in v} for s, v in by_split.items()}
    hash_sets = {s: {r['sha256'] for r in v} for s, v in by_split.items()}
    assert len(rows) == len(paths) == len({r['path'] for r in rows})
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not path_sets[a] & path_sets[b]
        assert not hash_sets[a] & hash_sets[b]
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['path', 'label', 'class_name', 'split', 'sha256'])
    writer.writeheader(); writer.writerows(rows)
    manifest_text = out.getvalue()
    if MANIFEST.exists() and MANIFEST.read_text() != manifest_text.replace(chr(13)+chr(10), chr(10)):
        raise FileExistsError('An existing manifest differs; inspect it before replacing it.')
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(manifest_text)
    summary = {
        'seed': seed, 'requested_ratios': {'train': .70, 'validation': .15, 'test': .15},
        'label_mapping': {'cat': 0, 'dog': 1}, 'total_images': len(rows),
        'unique_content_hashes': len(groups), 'duplicate_images': len(paths)-len(groups),
        'excluded_unreadable_images': len(bad), 'path_overlap': 0, 'content_hash_overlap': 0,
        'manifest': str(MANIFEST.relative_to(ROOT)),
        'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'splits': {s: {'count': len(v), 'class_counts': dict(Counter(r['class_name'] for r in v))} for s, v in by_split.items()},
        'notes': 'Exact duplicate content is kept in one split. Images are not copied. Ratios may differ slightly when duplicate groups have multiple members. The test split is reserved for final evaluation and is not used for fitting or model selection.'
    }
    (ROOT / 'reports/split_summary.json').write_text(json.dumps(summary, indent=2)+chr(10))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=SEED)
    build_splits(parser.parse_args().seed)
