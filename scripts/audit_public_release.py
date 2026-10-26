#!/usr/bin/env python3
"""Fail closed when a public source snapshot contains likely study data.

Checks files in the repository directory, *not* permissions or legal clearance.
Use before `git add` and inspect the staged list as a separate step.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_EXTENSIONS = {
    '.pth', '.pt', '.ckpt', '.pkl', '.pickle', '.npy', '.npz',
    '.h5', '.hdf5', '.h5ad', '.parquet', '.csv', '.tsv',
    '.svs', '.ndpi', '.mrxs', '.tiff', '.tif', '.jpg', '.jpeg',
    '.png', '.bmp', '.pdf', '.docx', '.xlsx', '.zip',
}
RESTRICTED_NAMES = re.compile(
    r'(?:^|[_.-])(?:train_fold[1-5]|test_fold[1-5]|trainAB_latent|testAB_latent|'
    r'Gene2vec_0111|hj_scaler_mat|gene_embeddings|pretrain_\d+)(?:[_.-]|$)', re.IGNORECASE
)
TEXT_EXTENSIONS = {'.py', '.md', '.txt', '.yml', '.yaml', '.toml', '.json', '.cfg', '.ini'}
PATIENT_ID = re.compile(r'\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-\S+)?', re.IGNORECASE)
LOCAL_PATH = re.compile(r'(/home/[^\s\'"`]+|/mnt/data/[^\s\'"`]+)')
IGNORED_DIRS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.git', '.venv', 'venv'}


def audit():
    issues = []
    scanned = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] == 'data':
            if str(rel) not in {'data/README.md', 'data/.gitkeep'}:
                issues.append(f'Private data file present: {rel}')
            continue
        scanned += 1
        if path.suffix.lower() in RESTRICTED_EXTENSIONS:
            issues.append(f'Forbidden potential data/artifact in public directory: {rel}')
        if path.suffix.lower() not in {".py", ".md", ".txt"} and RESTRICTED_NAMES.search(path.name):
            issues.append(f'Possible sensitive research file name: {rel}')
        if path.suffix.lower() in TEXT_EXTENSIONS and path.stat().st_size < 1_000_000:
            txt = path.read_text(encoding='utf-8', errors='replace')
            # Prevent accidental embedding of real TCGA patient barcodes or
            # personal machine file paths in public source/doc files.
            if PATIENT_ID.search(txt):
                issues.append(f'Possible TCGA patient barcode in source text: {rel}')
            if path.name != 'audit_public_release.py' and LOCAL_PATH.search(txt):
                issues.append(f'User-specific absolute filesystem path in source text: {rel}')
    return scanned, issues


if __name__ == '__main__':
    count, issues = audit()
    print(f'Public-release audit: inspected {count} files outside data/')
    if issues:
        for issue in issues:
            print('FAIL:', issue)
        print('Remove flagged files and re-run before publishing.')
        sys.exit(1)
    print('PASS: No obvious private data files/barcodes/absolute user paths in the source snapshot.')
    print('Review the files staged by Git before publishing.')
