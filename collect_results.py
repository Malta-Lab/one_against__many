import os
from pathlib import Path
import shutil

rootdir = Path('checkpoints')

save_path = Path('all_results')
save_path.mkdir(exist_ok=True)

for (root, dirs, file) in os.walk(rootdir):
    for f in file:
        if '.json' in f:
            file_path = Path(root) / f
            Path(save_path / '/'.join(str(file_path).split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            print(file_path)
            shutil.copy(str(file_path), str(save_path / file_path))
            