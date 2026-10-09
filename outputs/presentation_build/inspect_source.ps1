$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$source = 'C:\Users\Kryo\Downloads\Deep_Sky_Divide_ASTROMANTHAN_v4.pptx'
$build = 'C:\Learning some new stuf\Astromanthan\outputs\presentation_build'
$zip = [System.IO.Compression.ZipFile]::OpenRead($source)
function Read-ZipText([string]$name) {
  $entry = $zip.GetEntry($name)
  if (-not $entry) { return $null }
  $sr = [System.IO.StreamReader]::new($entry.Open())
  try { return $sr.ReadToEnd() } finally { $sr.Dispose() }
}
function Ns($doc) {
  $n = [System.Xml.XmlNamespaceManager]::new($doc.NameTable)
  $n.AddNamespace('p','http://schemas.openxmlformats.org/presentationml/2006/main')
  $n.AddNamespace('a','http://schemas.openxmlformats.org/drawingml/2006/main')
  $n.AddNamespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
  return ,$n
}
function Rel-List([string]$name) {
  $raw = Read-ZipText $name
  if (-not $raw) { return @() }
  [xml]$d=$raw
  return @($d.DocumentElement.ChildNodes | ForEach-Object { @{id=$_.Id;type=$_.Type;target=$_.Target} })
}
[xml]$pres = Read-ZipText 'ppt/presentation.xml'
$pn = Ns $pres
$size = $pres.SelectSingleNode('//p:sldSz',$pn)
$slides=@()
foreach ($entry in @($zip.Entries | Where-Object { $_.FullName -match '^ppt/slides/slide\d+\.xml$' } | Sort-Object { [int]([regex]::Match($_.FullName,'slide(\d+)\.xml').Groups[1].Value) })) {
  $number = [int]([regex]::Match($entry.FullName,'slide(\d+)\.xml').Groups[1].Value)
  [xml]$doc = Read-ZipText $entry.FullName
  $n = Ns $doc
  $rels = Rel-List "ppt/slides/_rels/slide$number.xml.rels"
  $shapes=@()
  foreach($node in $doc.SelectNodes('//p:spTree/*',$n)) {
    $id = $node.SelectSingleNode('.//p:cNvPr',$n)
    if (-not $id) { continue }
    $xfrm = $node.SelectSingleNode('.//a:xfrm',$n)
    if (-not $xfrm) { $xfrm=$node.SelectSingleNode('.//p:xfrm',$n) }
    $box=$null
    if($xfrm){
      $off=$xfrm.SelectSingleNode('./a:off',$n);if(-not $off){$off=$xfrm.SelectSingleNode('./p:off',$n)}
      $ext=$xfrm.SelectSingleNode('./a:ext',$n);if(-not $ext){$ext=$xfrm.SelectSingleNode('./p:ext',$n)}
      if($off -and $ext){$box=@{x=[math]::Round([double]$off.x/914400,3);y=[math]::Round([double]$off.y/914400,3);w=[math]::Round([double]$ext.cx/914400,3);h=[math]::Round([double]$ext.cy/914400,3)}}
    }
    $paras=@($node.SelectNodes('.//a:p',$n) | ForEach-Object { (@($_.SelectNodes('.//a:t',$n)|ForEach-Object{$_.InnerText}) -join '') })
    $fonts=@($node.SelectNodes('.//a:rPr',$n) | ForEach-Object {@{size=$_.sz;bold=$_.b;color=$_.InnerXml}})
    $ph=$node.SelectSingleNode('.//p:ph',$n)
    $blip=$node.SelectSingleNode('.//a:blip',$n)
    $shapes+=@{id=$id.id;name=$id.name;kind=$node.LocalName;box=$box;text=($paras -join "`n");paragraphs=$paras;placeholder=if($ph){$ph.OuterXml}else{$null};imageRel=if($blip){$blip.GetAttribute('embed','http://schemas.openxmlformats.org/officeDocument/2006/relationships')}else{$null};runs=$fonts}
  }
  $notesRel=@($rels|Where-Object{$_.type -match '/notesSlide$'})
  $notes=''
  if($notesRel.Count){
    $notePath='ppt/'+$notesRel[0].target.Replace('../','')
    [xml]$nd=Read-ZipText $notePath
    $nn=Ns $nd
    $notes=(@($nd.SelectNodes('//a:t',$nn)|ForEach-Object{$_.InnerText}) -join "`n")
  }
  $slides+=@{number=$number;shapes=$shapes;notes=$notes;relationships=$rels}
}
$charts=@()
foreach($entry in @($zip.Entries|Where-Object{$_.FullName -match '^ppt/charts/chart\d+\.xml$'})){
  $charts+=@{path=$entry.FullName;xml=(Read-ZipText $entry.FullName)}
}
$out=@{source=$source;width_in=[math]::Round([double]$size.cx/914400,3);height_in=[math]::Round([double]$size.cy/914400,3);slides=$slides;charts=$charts}
$out | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $build 'source_inventory.json') -Encoding UTF8
$assets=Join-Path $build 'source_assets'
New-Item -ItemType Directory -Force -Path $assets | Out-Null
foreach($entry in @($zip.Entries|Where-Object{$_.FullName -like 'ppt/media/*' -or $_.FullName -eq 'docProps/thumbnail.jpeg'})) {
  $target=Join-Path $assets ([System.IO.Path]::GetFileName($entry.FullName))
  [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$target,$true)
}
$zip.Dispose()
$slides|ForEach-Object { "SLIDE $($_.number)"; $_.shapes|ForEach-Object{"ID $($_.id) [$($_.kind)] $($_.name) BOX $($_.box|ConvertTo-Json -Compress) TEXT $($_.text)"}; "NOTES: $($_.notes)" }
