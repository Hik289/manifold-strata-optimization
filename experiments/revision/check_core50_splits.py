from pathlib import Path
from prepare_image_benchmarks import download,extract_zip
r=Path('../data/core50');p=download(['https://vlomonaco.github.io/core50/data/batches_filelists.zip'],r/'archives/batches_filelists.zip');extract_zip(p,r/'splits')
f=[f for f in (r/'splits').rglob('train_batch_00_filelist.txt') if 'nc_inc' in str(f).lower() and 'run0' in str(f).lower()];assert len(f)==1,f
print(f[0],f[0].read_text().splitlines()[:3]);print('Batches',len(list(f[0].parent.glob('train_batch_*_filelist.txt'))))
