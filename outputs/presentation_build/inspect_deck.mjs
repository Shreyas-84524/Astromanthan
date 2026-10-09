import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from 'file:///C:/Users/Kryo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const dir='C:/Learning some new stuf/Astromanthan/outputs/presentation_build';
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'/source_compatible.pptx'));
const snap=await p.inspect({select:'$.slides[*]',include:['id','index','layout','elements.id','elements.kind','elements.name','elements.text','elements.position','notes.text']});
await fs.writeFile(dir+'/artifact_inventory.ndjson',snap.ndjson);
console.log(snap.ndjson);
