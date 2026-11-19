"""Build and test a self-contained local companion; never publish or change Git."""
from __future__ import annotations

import hashlib
from importlib.metadata import version as installed_version
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = 'a4c8880e469445cb203ee157d52990b2e9b6d5bc'
VERSION = 'v1.0.1'
NOTE_VERSION = '1.0.0'
BUILD_DATE = '2026-11-18'
NAME = 'local-recovery-note-' + VERSION
SERIES = ('minimal_observer_2026-10-08', 'help_timing_2026-10-08', 'repeated_help_2026-10-08')
SCIENTIFIC_FILES = (
    'experiment_local_support_signal.py', 'experiment_minimal_observer.py',
    'experiment_help_timing.py', 'experiment_repeated_help.py',
    'report_minimal_observer.py', 'report_help_timing.py', 'report_repeated_help.py',
    'plot_minimal_observer.py', 'plot_help_timing.py', 'plot_repeated_help.py',
    'check_repeated_help_example.py', 'test_minimal_observer.py',
    'test_help_timing.py', 'test_repeated_help.py',
)
NOTE_FILES = ('note.ru.md', 'README.md', 'REPRODUCE.md', 'quick_demo.py',
              'audit_note.py', 'evidence.json', 'build_pdf.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(path):
    result = subprocess.run(['git', 'show', f'{BASE}:{path}'], cwd=ROOT,
                            capture_output=True, check=True)
    return result.stdout


def main():
    build_parent = ROOT / 'tmp/releases'
    build_parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix='stage-', dir=build_parent))
    stage = stage_parent / NAME
    stage.mkdir()
    origins = {}

    def put(path, data, origin):
        target = stage / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        origins[path] = origin

    def put_text(path, text, origin='new companion documentation, 2026-11-17'):
        put(path, text.encode('utf-8'), {'kind': 'new_companion', 'description': origin})

    for filename in SCIENTIFIC_FILES:
        relative = 'simulations/' + filename
        data = git_bytes(relative)
        live = (ROOT / relative).read_bytes()
        if data.replace(b'\r\n', b'\n') != live.replace(b'\r\n', b'\n'):
            raise RuntimeError('Working scientific file differs from pinned source: ' + relative)
        put(relative, data, {'kind': 'pinned_git', 'revision': BASE, 'path': relative})
    for name in ('minimal_observer', 'help_timing', 'repeated_help'):
        relative = f'manuscripts/v6/{name}_protocol_2026-10-08.md'
        put(relative, git_bytes(relative), {'kind': 'pinned_git', 'revision': BASE, 'path': relative})

    manifest_bytes = (ROOT / 'research_results/manifest.json').read_bytes()
    upstream = json.loads(manifest_bytes)
    expected = {r['path']: r for r in upstream['files']}
    subset = []
    for series in SERIES:
        for source in sorted((ROOT / 'research_results' / series).rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(ROOT / 'research_results').as_posix()
            data = source.read_bytes()
            assert sha(data) == expected[relative]['sha256'], relative
            assert len(data) == expected[relative]['bytes'], relative
            subset.append(expected[relative])
            put('research_results/' + relative, data,
                {'kind': 'archived_result', 'revision': BASE,
                 'path': 'research_results/' + relative, 'original_sha256': sha(data)})
    subset_manifest = dict(checkpoint=upstream['checkpoint'],
                           subset_of=f'{BASE}:research_results/manifest.json',
                           original_manifest_sha256=sha(manifest_bytes), files=subset)
    put_text('research_results/manifest.json', json.dumps(subset_manifest, indent=2) + '\n',
             'subset of original archive manifest; original file entries unchanged')
    put_text('research_results/README.md', '# Included archived results\n\n' + '\n'.join(
        f'- [{s}]({s}/)' for s in SERIES) + '\n\nSee [SOURCE_MAP.md](../SOURCE_MAP.md).\n')

    for filename in NOTE_FILES:
        put('publication/recovery-note-2026-11-17/' + filename, (HERE / filename).read_bytes(),
            {'kind': 'new_companion', 'path': 'publication/recovery-note-2026-11-17/' + filename})
    pdf_path = 'output/pdf/local-recovery-note-ru.pdf'
    put(pdf_path, (ROOT / pdf_path).read_bytes(),
        {'kind': 'new_companion', 'description': 'PDF built from included note.ru.md; same file as separate release asset'})

    for source in sorted((HERE / 'release_kit').rglob('*')):
        if source.is_file():
            relative = source.relative_to(HERE / 'release_kit').as_posix()
            text = source.read_text(encoding='utf-8').replace('{{VERSION}}', VERSION).replace('{{BASE}}', BASE).replace('{{BUILD_DATE}}', BUILD_DATE)
            put_text(relative, text)
    put_text('requirements.txt', 'numpy==2.3.5\n')
    put_text('requirements-plots.txt', '-r requirements.txt\nmatplotlib>=3.8,<4\n')
    put_text('requirements-pdf.txt', 'reportlab>=4,<5\nPillow>=10,<13\n')
    metadata = dict(title='Локальное восстановление ресурсной сети: наблюдение, время помощи и последствия повторных вмешательств',
                    version=VERSION, note_version=NOTE_VERSION, status='prepared_for_publication', prepared_date=BUILD_DATE,
                    display_author_line='SymbiosisK (Katia) & ChatGPT / Codex (OpenAI)',
                    responsible_human_author='SymbiosisK (Katia)', ai_contribution_file='AUTHORSHIP.md',
                    ai_contributors=[{'system': 'Codex', 'provider': 'OpenAI',
                                      'role': 'analysis, verification code, numerical audit, writing and packaging'},
                                     {'system': 'ChatGPT', 'name': 'Лира', 'provider': 'OpenAI',
                                      'role': 'additional technical review and scientific editing recommendations'}],
                    licenses={'code': 'MIT', 'text_figures_data': 'CC-BY-4.0'},
                    scientific_base_revision=BASE, doi=None,
                    metadata_kind='companion package metadata; not a Zenodo API request')
    put_text('release-metadata.json', json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    validation = {'package_version': VERSION, 'note_version': NOTE_VERSION, 'prepared_date': BUILD_DATE,
                  'python': platform.python_version(), 'checks': [],
                  'not_performed': ['full simulation sweeps', 'fresh bootstrap intervals',
                                    'optional Matplotlib plot regeneration', 'independent external review']}
    commands = [
        ['-B', '-I', 'publication/recovery-note-2026-11-17/audit_note.py'],
        ['-B', '-I', 'run_demo.py'],
        ['-B', '-I', '-m', 'unittest', 'discover', '-s', 'simulations', '-p', 'test_*.py', '-v'],
    ]
    for command in commands:
        result = subprocess.run([sys.executable] + command, cwd=stage, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=180)
        record = dict(command='python ' + ' '.join(command), exit_code=result.returncode,
                      stdout=result.stdout, stderr=result.stderr)
        validation['checks'].append(record)
        if result.returncode:
            raise RuntimeError(json.dumps(record, ensure_ascii=False))
    # A separate package root and isolated Python mode prevent imports from the source checkout.
    validation['isolation'] = 'fresh companion directory; Python -I; no source-checkout imports'
    validation['source_archive_files_verified'] = len(subset)
    validation['scientific_modules_from_pinned_git'] = len(SCIENTIFIC_FILES)
    validation['protocols_from_pinned_git'] = 3
    validation['numpy'] = installed_version('numpy')
    validation['archived_numpy'] = json.loads((stage / 'research_results' / SERIES[0] / 'config.json').read_text())['numpy']
    test_output = validation['checks'][-1]['stderr']
    match = re.search(r'Ran (\d+) tests?', test_output)
    validation['unit_tests_passed'] = int(match[1]) if match else None

    # Validate relative links in the new navigation and note. Archived data tables are untouched.
    from urllib.parse import urlsplit, unquote
    link_count = 0
    for path in [p for p in stage.rglob('*.md') if 'research_results' not in p.parts]:
        text = path.read_text(encoding='utf-8')
        assert '{{VERSION}}' not in text and '{{BASE}}' not in text, path
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            resolved = (path.parent / unquote(target.split('#')[0])).resolve()
            if not resolved.exists() or not resolved.is_relative_to(stage.resolve()):
                raise RuntimeError(f'Invalid companion link: {path.relative_to(stage)} -> {target}')
            link_count += 1
    validation['relative_links_verified'] = link_count
    # Two files are still to be written: VALIDATION.json and ORIGIN.json.
    # MANIFEST.json excludes itself, and the ZIP includes it.
    expected_manifest_files = sum(p.is_file() for p in stage.rglob('*')) + 2
    validation['file_counts'] = dict(
        source_archive_files=len(subset),
        package_manifest_files=expected_manifest_files,
        zip_files_including_manifest=expected_manifest_files + 1)
    validation['file_count_definitions'] = dict(
        source_archive_files='Original data and code snapshots in the three research_results series; a subset of package_manifest_files, not the whole companion.',
        package_manifest_files='Every companion file listed in MANIFEST.json, including data, working code, PDF and documentation; MANIFEST.json itself is excluded.',
        zip_files_including_manifest='All packaged files, including MANIFEST.json; no directory entries are added to the ZIP.')
    put_text('VALIDATION.json', json.dumps(validation, ensure_ascii=False, indent=2) + '\n', 'actual isolated package checks')
    put_text('ORIGIN.json', json.dumps(dict(base_revision=BASE, prepared_date=BUILD_DATE, files=origins),
                                      ensure_ascii=False, indent=2) + '\n', 'per-file source classification')

    files = [dict(path=p.relative_to(stage).as_posix(), bytes=p.stat().st_size,
                  sha256=sha(p.read_bytes())) for p in sorted(stage.rglob('*')) if p.is_file()]
    assert len(files) == validation['file_counts']['package_manifest_files']
    (stage / 'MANIFEST.json').write_text(json.dumps(dict(version=VERSION,
        excludes=['MANIFEST.json'], files=files), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = subprocess.run([sys.executable, '-B', '-I', 'verify_package.py'], cwd=stage,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['manifest_files'] == len(files)

    release = ROOT / 'release' / NAME
    release.mkdir(parents=True, exist_ok=True)
    zip_path = release / (NAME + '.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                archive.write(path, NAME + '/' + path.relative_to(stage).as_posix())

    # Verify the actual downloadable ZIP after extraction to another new directory.
    extract_parent = Path(tempfile.mkdtemp(prefix='verify-', dir=build_parent))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            relative = PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
            assert relative.parts[0] == NAME
        archive.extractall(extract_parent)
    extracted = extract_parent / NAME
    checks_after_extraction = []
    for command in (['verify_package.py'], ['run_demo.py']):
        out = subprocess.run([sys.executable, '-B', '-I'] + command, cwd=extracted,
                             capture_output=True, text=True, check=True, timeout=90)
        checks_after_extraction.append(dict(command='python -B -I ' + ' '.join(command), stdout=out.stdout))
    shutil.copyfile(stage / pdf_path, release / 'local-recovery-note-ru.pdf')
    for name in ('ZENODO_UPLOAD.md', 'RELEASE_NOTES.md'):
        shutil.copyfile(stage / name, release / name)
    # The outer guide points into the ZIP, rather than pretending the inner files are loose assets.
    (release / 'README.md').write_text(
        '# Комплект ' + VERSION + ' к заметке ' + NOTE_VERSION + '\n\n'
        '**' + NAME + '**\n\n'
        '1. Для чтения откройте `local-recovery-note-ru.pdf`.\n'
        '2. Распакуйте `' + zip_path.name + '` и начните с `START_HERE.md` внутри архива.\n'
        '3. `ZENODO_UPLOAD.md` содержит подготовленные сведения для размещения.\n\n'
        'Полная авторская строка: SymbiosisK (Katia) & ChatGPT / Codex (OpenAI).\n'
        'MIT для кода, CC BY 4.0 для текста и данных.\n'
        'Пакет проверен отдельно от исходной папки; публикация ещё не выполнена.\n', encoding='utf-8')
    # Keep a separate report for the post-ZIP checks; it is not falsely included in its own archive.
    outer_report = dict(version=VERSION, note_version=NOTE_VERSION, zip_sha256=sha(zip_path.read_bytes()),
                        zip_bytes=zip_path.stat().st_size, zip_members=len(files)+1,
                        extracted_zip_checks=checks_after_extraction,
                        isolated_unit_tests=validation['unit_tests_passed'],
                        source_archive_files_verified=len(subset),
                        package_manifest_files_verified=len(files),
                        zip_files_including_manifest=len(files)+1,
                        relative_links_verified=link_count)
    (release / 'PACKAGE_CHECK.json').write_text(json.dumps(outer_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checksums = ''.join(sha(p.read_bytes()) + '  ' + p.name + '\n'
                        for p in sorted(release.iterdir()) if p.is_file() and p.name != 'SHA256SUMS.txt')
    (release / 'SHA256SUMS.txt').write_text(checksums, encoding='utf-8')
    print(json.dumps(dict(release=str(release), archive=str(zip_path), bytes=zip_path.stat().st_size,
                          files=len(files)+1, unit_tests=validation['unit_tests_passed'],
                          archive_files=len(subset), relative_links=link_count,
                          stage=str(stage_parent), extraction=str(extract_parent)), ensure_ascii=False))


if __name__ == '__main__':
    main()
