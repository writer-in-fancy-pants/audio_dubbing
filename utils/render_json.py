import json
from pathlib import Path
path = Path('/Users/ganois/workdir/tools/audio_translate/work_coqui/translated.json')
with path.open('r', encoding='utf-8') as f:
    data = json.load(f)
    print(data)
