from pathlib import Path

# negs
for i in Path('data/neg').iterdir():
    if int(i.name.split('-')[-1].split('.')[0]) > 255:
        i.unlink()
