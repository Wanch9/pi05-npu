"""Install the upstream OpenPI overlay into this isolated venv only."""
from pathlib import Path
import shutil
import sys


def main():
    import transformers

    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate an isolated venv before installing the overlay')
    target = Path(transformers.__file__).resolve().parent
    if not target.is_relative_to(prefix):
        raise RuntimeError('Install transformers inside the venv; refusing to modify shared site-packages')
    if transformers.__version__ != '4.53.2':
        raise RuntimeError('This overlay requires transformers==4.53.2')
    source = Path(__file__).resolve().parents[1] / 'src/openpi/models_pytorch/transformers_replace'
    shutil.copytree(source, target, dirs_exist_ok=True)
    for path in source.rglob('*.py'):
        if path.read_bytes() != (target / path.relative_to(source)).read_bytes():
            raise RuntimeError(f'Overlay verification failed: {path.name}')
    print('OPENPI_TRANSFORMERS_OVERLAY_VERIFIED')


if __name__ == '__main__':
    main()
