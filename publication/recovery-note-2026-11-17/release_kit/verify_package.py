"""Verify all manifest-listed files with Python's standard library only."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    errors = []
    for entry in manifest['files']:
        relative = PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in entry['path']:
            errors.append('unsafe path: ' + entry['path']); continue
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            errors.append('missing: ' + entry['path']); continue
        if path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append('changed: ' + entry['path'])
    validation = json.loads((root / 'VALIDATION.json').read_text(encoding='utf-8'))
    archive_manifest = json.loads((root / 'research_results/manifest.json').read_text(encoding='utf-8'))
    expected_counts = dict(source_archive_files=len(archive_manifest['files']),
                           package_manifest_files=len(manifest['files']),
                           zip_files_including_manifest=len(manifest['files']) + 1)
    if validation.get('file_counts') != expected_counts:
        errors.append('VALIDATION.json file_counts disagree with the package and archive manifests')
    if validation.get('source_archive_files_verified') != expected_counts['source_archive_files']:
        errors.append('VALIDATION.json source_archive_files_verified disagrees with the archive manifest')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({'manifest_files': len(manifest['files']), 'file_counts': expected_counts, 'result': 'pass',
                      'meaning': 'file integrity, not independent scientific validation'}))


if __name__ == '__main__':
    main()
