"""Syntax-check all dashboard Python files."""
import sys, pathlib
sys.path.insert(0, '.')
errors = []
files = list(pathlib.Path('dashboard').rglob('*.py'))
for p in sorted(files):
    src = p.read_text(encoding='utf-8')
    try:
        compile(src, str(p), 'exec')
    except SyntaxError as e:
        errors.append(f'{p}: {e}')
if errors:
    for e in errors:
        print('SYNTAX ERROR:', e)
else:
    print(f'All {len(files)} dashboard files have valid Python syntax. Ready to run.')
