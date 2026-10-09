from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re

src=Path(r'C:\Users\Kryo\Downloads\Deep_Sky_Divide_ASTROMANTHAN_v4.pptx')
out=Path(__file__).parent/'source_compatible.pptx'
with ZipFile(src) as z, ZipFile(out,'w',ZIP_DEFLATED) as w:
    for info in z.infolist():
        data=z.read(info.filename)
        if info.filename.startswith('ppt/charts/') and info.filename.endswith('.xml'):
            text=data.decode('utf-8')
            text=re.sub(r'(<c:(?:axId|crossAx)\b[^>]*\bval=")(-\d+)(")',lambda m:m[1]+str(int(m[2])%(2**32))+m[3],text)
            data=text.encode('utf-8')
        w.writestr(info,data)
print(out)
