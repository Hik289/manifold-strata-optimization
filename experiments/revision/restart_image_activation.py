"""Restart only this project's data-activation watcher after its robustness fix."""
import os,signal,subprocess,sys
from pathlib import Path
root=Path('..').resolve();pid=117672
try:
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
 if b'revision/activate_image_jobs.py' in cmd:os.kill(pid,signal.SIGTERM)
except (FileNotFoundError,ProcessLookupError):pass
with (root/'activate_image_jobs.log').open('a') as log:
 process=subprocess.Popen([sys.executable,'-u','revision/activate_image_jobs.py'],cwd=Path.cwd(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'image_activation.pid').write_text(str(process.pid));print('Data activation watcher',process.pid)
partial=root/'data/imagenet100/images'
if partial.exists():(partial/'INCOMPLETE_NOT_USED.txt').write_text('Preserved partial image-file materialization. The complete benchmark is backed by ../manifest.jsonl using the full imagenet100_arrow cache. This folder is not used for training.\n')
