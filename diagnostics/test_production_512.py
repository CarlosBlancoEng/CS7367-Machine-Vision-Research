"""Production notebook regressions. Default: synthetic; `real_masks`: read-only geometry only."""
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch
import nbformat
import numpy as np
from PIL import Image
import torch
import torchvision.models

repoRoot = Path(__file__).resolve().parents[1]
notebookPath = repoRoot / 'notebooks/CS7367_Research_Experiments.ipynb'
notebook = nbformat.read(notebookPath, as_version=4)
nbformat.validate(notebook)
codeCells = [c.source for c in notebook.cells if c.cell_type == 'code']
assert len(notebook.cells) == 12 and len(codeCells) == 6
assert sum(c.cell_type == 'markdown' for c in notebook.cells) == 6
for cell in notebook.cells:
    if cell.cell_type == 'code':
        compile(cell.source, cell.id, 'exec')
        assert not cell.outputs and cell.execution_count is None
allCategories = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
manifestFields = ['category', 'role', 'label', 'group', 'relativeImagePath', 'workloadMode']
fixtureRoot = Path(os.environ.get('CS7367_TEST_ROOT', str(Path(tempfile.gettempdir()) / 'cs7367-production512-tests')))
fixtureRoot.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
originalResnet = torchvision.models.resnet18

def fileHash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def execute(root, selected=None, fakeFeatures=False, interrupt=False, nonfinite=False, stopAfter=None, metadataOnly=False, realRoot=None, expectedDigest=None, emptyCoverage=False, maskMemoryError=False):
    namespace, messages, calls, requests = {}, [], [], []
    archive = root / 'output/experiments/small/20261115T000643_647170Z'
    digest = expectedDigest or fileHash(archive / 'sample_manifest.csv')
    setup = codeCells[0].replace("from google.colab import drive\ndrive.mount('/content/drive')", '')
    setup = setup.replace("Path('/content/drive/MyDrive/ColabData/CS7367_Vision/mvtec_ad_2')", f'Path({str(realRoot or root / "mvtec_ad_2")!r})')
    setup = setup.replace("Path('/content/drive/MyDrive/ColabData/CS7367_Vision/output/experiments')", f'Path({str(root / "output/experiments")!r})')
    # Test-fixture substitution only. Production retains the pinned real archived SHA.
    setup = setup.replace('0588df4ceceabe99b7d4fab5a6798d13df467b02863d9da42517a1edd907c7c0', digest)
    if selected is not None:
        setup = setup.replace('selectedCategories = list(allCategories)', f'selectedCategories = {selected!r}')
    namespace['print'] = lambda *args, **kwargs: messages.append((' '.join(map(str, args)), kwargs.get('flush', False)))
    def offlineResnet(**kwargs):
        requests.append(kwargs['weights'])
        assert kwargs['weights'] == torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        return originalResnet(weights=None)
    def runCell(index, source):
        if index == 4 and (emptyCoverage or maskMemoryError):
            nodes = ast.parse(source).body
            functions = [node for node in nodes if isinstance(node, ast.FunctionDef)]
            body = [node for node in nodes if not isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=functions,type_ignores=[]),'fault_injection_definitions','exec'),namespace)
            if maskMemoryError:
                def failCoverage(*args):
                    raise MemoryError('synthetic mask audit memory failure')
                namespace['coverageMask'] = failCoverage
            else:
                namespace['coverageMask'] = lambda native, target=512: np.zeros((target,target),np.uint8)
            exec(compile(ast.Module(body=body,type_ignores=[]),'fault_injection_body','exec'),namespace)
            return
        exec(compile(source, f'cell-{index}', 'exec'), namespace)
    with patch.object(torchvision.models, 'resnet18', offlineResnet):
        for index, source in enumerate([setup] + codeCells[1:]):
            if stopAfter is not None and index > stopAfter:
                break
            if metadataOnly and index > 1:
                break
            try:
                runCell(index, source)
                if index == 3:
                    originalExtract = namespace['extractFeatures']
                    def tracedExtract(paths, condition):
                        calls.append((condition, tuple(paths), namespace['summary']['maskAuditComplete']))
                        if interrupt and condition == 'gamma':
                            raise KeyboardInterrupt('synthetic mid-run stop')
                        if nonfinite:
                            return np.full((len(paths), 4096, 32), np.nan)
                        if not fakeFeatures:
                            return originalExtract(paths, condition)
                        # Fast failure-path fixture; primary full run uses actual ResNet18 architecture.
                        output = []
                        for path in paths:
                            seed = int(hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:8], 16)
                            output.append(np.random.default_rng(seed).normal(size=(4096, 32)))
                        return np.array(output)
                    namespace['extractFeatures'] = tracedExtract
            except (KeyboardInterrupt, ValueError, FileNotFoundError, MemoryError) as error:
                namespace['caughtError'] = error
                break
    return namespace, messages, calls, requests

def makeFixture(root):
    rng = np.random.default_rng(7367)
    rows = []
    for category in allCategories:
        for role, group, count, partition in [('reference', 'good', 25, 'train'), ('evaluation', 'good', 10, 'test_public'), ('evaluation', 'bad', 10, 'test_public')]:
            # Deliberately non-sorted archived order to detect accidental reselection/reordering.
            for index in list(range(count))[::-1]:
                path = root / 'mvtec_ad_2' / category / partition / group / f'{index:03}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                rgb = rng.integers(20, 90, size=(24, 32, 3), dtype=np.uint8)
                if group == 'bad':
                    rgb[4:12, 8:16] = 230
                Image.fromarray(rgb).save(path)
                if group == 'bad':
                    mask = np.zeros((24, 32), dtype=np.uint8)
                    mask[4:12, 8:16] = 255
                    maskPath = path.parent.parent / 'ground_truth/bad' / f'{index:03}_mask.png'
                    maskPath.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(maskPath)
                rows.append(dict(category=category, role=role, label=str(int(group == 'bad')), group=group,
                                 relativeImagePath=path.relative_to(root / 'mvtec_ad_2').as_posix(), workloadMode='small'))
    archive = root / 'output/experiments/small/20261115T000643_647170Z'
    archive.mkdir(parents=True)
    with (archive / 'sample_manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=manifestFields)
        writer.writeheader()
        writer.writerows(rows)
    return rows

def readEvidence(namespace):
    root = namespace['runRoot']
    summary = json.loads((root / 'experiment_summary.json').read_text())
    with (root / 'metrics.csv').open(newline='') as handle:
        metrics = list(csv.DictReader(handle))
    return summary, metrics

def definitions(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}

def protocolChecks():
    expected = {'imageSize':512,'featureGrid':None,'featureDimension':32,'batchSize':5,
                'randomSeed':7367,'covarianceRegularization':0.01,
                'workloadLimits':{'small':(25,20),'medium':(250,100),'large':(None,None)},
                'conditions':['baseline','gamma','histogram_equalization','clahe'],
                'enhancementParameters':{'gamma':0.7,'claheClipLimit':2.0,'claheTileGrid':[8,8],
                    'channel':'CIELAB L after RGB bilinear resize; baseline RGB unchanged'}}
    values = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(codeCells[0]).body
              if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in expected}
    assert values==expected
    assert 'ResNet18_Weights.IMAGENET1K_V1' in codeCells[3]
    assert 'sigma=4.0' in codeCells[4]
    print('PASS: approved workload caps/conditions/seed/32 features/regularization/batch size, pretrained weights and sigma4 retained; feature grid derived rather than configured',flush=True)

def offlineTests():
    diagnostic = (repoRoot / 'diagnostics/resolution_512.py').read_text()
    sourceFunctions = definitions(diagnostic)
    productionFunctions = {name: node for source in codeCells for name, node in definitions(source).items()}
    for name in ['coverageMask', 'maskStats', 'deriveGrid', 'validateManifest']:
        assert ast.dump(sourceFunctions[name]) == ast.dump(productionFunctions[name])
    baseline = json.loads(subprocess.check_output(['git', 'show', '8c6353d:notebooks/CS7367_Research_Experiments.ipynb'], cwd=repoRoot, encoding='utf-8'))
    for index, name in [(3, 'selectImages'), (5, 'enhanceImage'), (9, 'auPro005'), (1, 'logStage')]:
        assert ast.dump(definitions(baseline['cells'][index]['source'])[name]) == ast.dump(productionFunctions[name])
    print('PASS: notebook schema/compilation/12 cleared cells; tested coverage/geometry helpers copied exactly; selector, enhancement, AU-PRO and UTC helper unchanged', flush=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-', dir=fixtureRoot) as folder:
        root = Path(folder)
        assert root.resolve().is_relative_to(fixtureRoot.resolve())
        originalRows = makeFixture(root)
        originalHashes = {p: fileHash(p) for p in (root / 'mvtec_ad_2').rglob('*.png')}
        namespace, messages, calls, requests = execute(root)
        if 'caughtError' in namespace:
            raise namespace['caughtError']
        summary, metrics = readEvidence(namespace)
        assert summary['status'] == 'complete' and not summary['errors'] and len(metrics) == 64
        assert summary['protocolVersion'] == '512-coverage-layer2-v1'
        assert summary['model']['inputSize'] == 512 and summary['model']['featureGrid'] == 64
        assert summary['model']['layerShapes'] == [[1,64,128,128], [1,128,64,64], [1,256,32,32]]
        assert summary['configuredCaps'] == {'reference': 25, 'evaluation': 20}
        assert summary['selectedCategories'] == allCategories and summary['seed'] == 7367
        assert summary['archivedManifestComparison'] == {'matched': True, 'selectedRows': 360}
        assert summary['fileManifest'] == originalRows
        assert fileHash(namespace['runRoot'] / 'sample_manifest.csv') == fileHash(root / 'output/experiments/small/20261115T000643_647170Z/sample_manifest.csv')
        assert len(summary['maskDiagnostics']) == 80 and all(r['status'] == 'valid' for r in summary['maskDiagnostics'])
        assert summary['maskAuditComplete'] and all(a['validMasks'] == 10 for a in summary['maskAuditOutcomes'])
        assert all(a['nativeComponents'] == a['representedNativeComponents'] for a in summary['maskAuditOutcomes'])
        assert all(r['metricStatus'] == 'available' and 0 <= float(r['metricValue']) <= 1 for r in metrics)
        assert all(audited for _, _, audited in calls)  # Every category audited before the first fit/scoring.
        assert len(requests) == 1 and not namespace['backbone'].training
        assert not any(p.requires_grad for p in namespace['backbone'].parameters())
        for category in allCategories:
            data = namespace['selectedData'][category]
            assert len(data['reference']) == 25 and len(data['evaluation']) == 20
            assert sum(g == 'bad' for g, _ in data['evaluation']) == 10
            for condition in namespace['conditions']:
                for role, records in data.items():
                    observed = [p for cond, paths, _ in calls if cond == condition for p in paths if category in p.parts and (('train' in p.parts) == (role == 'reference'))]
                    assert observed == [p for _, p in records]
                for count in [5, 10, 15, 20]:
                    assert any(f'| {category} | {condition} | inference ' in text and f'{count}/20 images' in text and flushed for text, flushed in messages)
            assert not {p.resolve() for _, p in data['reference']} & {p.resolve() for _, p in data['evaluation']}
        assert all(fileHash(p) == digest for p, digest in originalHashes.items())
        assert all(t['status'] == 'complete' and t['elapsedSeconds'] >= 0 for t in summary['stageTimings'])
        assert any(re.match(r'^\[\d{4}-\d{2}-\d{2}T.*Z\] START', text) for text, _ in messages)
        print('PASS: fresh all-eight 512 synthetic Run All using actual frozen ResNet18 architecture with offline test weights: 32 configurations, full 25/20 caps, 64 supported metrics, 80 mask audits before scoring, exact archived rows/order, immutable inputs, shared identities, timings/progress', flush=True)

        coverage = namespace['coverageMask']
        oraclePath = repoRoot / 'diagnostics/test_resolution_512.py'
        spec = importlib.util.spec_from_file_location('coverage_oracle', oraclePath)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        rng = np.random.default_rng(7367)
        for shape in [(512,512), (1024,2232), (2048,2448), (13,17)]:
            native = np.zeros(shape, np.uint8)
            native[rng.integers(shape[0], size=12), rng.integers(shape[1], size=12)] = 255
            assert np.array_equal(coverage(native), oracle.independentCoverage(native))
            assert np.array_equal(coverage(native), coverage(native))
        pro = namespace['auPro005']
        masks = np.zeros((2,4,6), np.uint8)
        masks[0,0,0] = 1
        masks[0,2,2:5] = 1
        assert abs(pro(masks, masks.astype(float))[0]-1) < 1e-12
        assert abs(pro(masks, np.ones_like(masks, dtype=float))[0]-.025) < 1e-12
        assert pro(np.zeros_like(masks), masks.astype(float))[0] is None
        oldRun = namespace['runRoot']
        repeated, _, _, _ = execute(root, metadataOnly=True)
        assert repeated['runRoot'] != oldRun
        assert (repeated['runRoot']/'sample_manifest.csv').read_bytes() == (oldRun/'sample_manifest.csv').read_bytes()
        subset, _, _, _ = execute(root, selected=['wallplugs','can'], metadataOnly=True)
        assert subset['summary']['fileManifest'] == [r for r in originalRows if r['category'] in ['wallplugs','can']]
        print('PASS: independent footprint oracle, deterministic binary coverage, AU-PRO numeric/degenerate regression, unique folders, repeat/subset manifest order', flush=True)
        # Failure-path runs use synthetic features; the full primary run above used the real architecture.
        interrupted, _, _, _ = execute(root, selected=['can'], fakeFeatures=True, interrupt=True)
        failedSummary, failedRows = readEvidence(interrupted)
        assert failedSummary['status'] == 'incomplete' and len(failedRows) == 2
        assert any(t['status'] == 'incomplete' and t['endUtc'] is None for t in failedSummary['stageTimings'])
        nonfinite, _, _, _ = execute(root, selected=['can'], nonfinite=True)
        badSummary, _ = readEvidence(nonfinite)
        assert badSummary['status'] == 'incomplete' and any('Nonfinite spatial covariance' in e for e in badSummary['errors'])
        missingMask = root / 'mvtec_ad_2/can/test_public/ground_truth/bad/000_mask.png'
        maskBytes = missingMask.read_bytes()
        missingMask.unlink()
        unsupported, _, _, _ = execute(root, selected=['can'], fakeFeatures=True)
        unsupportedSummary, rows = readEvidence(unsupported)
        assert unsupportedSummary['status'] == 'complete'
        assert all(r['metricValue'] == '' and 'unavailable' in r['metricStatus'] for r in rows if r['metricName']=='AU-PRO0.05')
        assert all(r['metricValue'] != '' for r in rows if r['metricName']=='image_AUROC')
        assert unsupportedSummary['actualCounts']['can']['evaluation'] == 20
        missingMask.write_bytes(maskBytes)
        missingImage = root / 'mvtec_ad_2/can/train/good/000.png'
        imageBytes = missingImage.read_bytes()
        missingImage.unlink()
        absent, _, _, _ = execute(root, selected=['can'], metadataOnly=True)
        absentSummary, _ = readEvidence(absent)
        assert absentSummary['status'] == 'incomplete' and 'caughtError' in absent
        missingImage.write_bytes(imageBytes)
        print('PASS: interrupted/nonfinite failures stay incomplete; invalid mask keeps 20 identities/AUROC and blank localization; absent identity stops without shrinking/resampling', flush=True)
    guardTests()

def guardTests():
    with tempfile.TemporaryDirectory(prefix='guards-',dir=fixtureRoot) as folder:
        root = Path(folder)
        assert root.resolve().is_relative_to(fixtureRoot.resolve())
        makeFixture(root)
        path = root/'output/experiments/small/20261115T000643_647170Z/sample_manifest.csv'
        originalBytes, originalDigest = path.read_bytes(), fileHash(path)
        path.write_bytes(originalBytes+b'\n')
        changed, _, _, requests = execute(root,metadataOnly=True,expectedDigest=originalDigest)
        changedSummary, _ = readEvidence(changed)
        assert changedSummary['status']=='incomplete' and not requests
        assert any('refusing to resample identities' in e for e in changedSummary['errors'])
        path.write_bytes(originalBytes)
        unrepresented, _, _, _ = execute(root,selected=['can'],fakeFeatures=True,emptyCoverage=True)
        summary, rows = readEvidence(unrepresented)
        assert summary['status']=='complete' and summary['actualCounts']['can']['evaluation']==20
        assert all(r['metricValue']=='' for r in rows if r['metricName']=='AU-PRO0.05')
        assert all(r['metricValue']!='' for r in rows if r['metricName']=='image_AUROC')
        outcome = summary['maskAuditOutcomes'][0]
        assert outcome['status']=='unsupported' and outcome['invalidMasks']==10
        assert outcome['nativeComponents']==10 and outcome['representedNativeComponents']==0
        assert all('unrepresented' in d['reason'] for d in summary['maskDiagnostics'])
        memoryFailure, _, _, _ = execute(root,selected=['can'],fakeFeatures=True,maskMemoryError=True)
        failedSummary, failedRows = readEvidence(memoryFailure)
        assert failedSummary['status']=='incomplete' and not failedSummary['maskAuditComplete'] and not failedRows
        assert any('MemoryError' in error for error in failedSummary['errors'])
        print('PASS: changed archive hash stops without resampling/model load; unrepresented-component fault keeps identities/AUROC, blank localization and truthful native/target audit counts',flush=True)
        print('PASS: mask-audit MemoryError halts with incomplete evidence and no model metric rows; no silent resource-driven metric suppression',flush=True)

def realMasks():
    driveRoot = Path(os.environ.get('CS7367_DRIVE_ROOT', 'G:/My Drive')) / 'ColabData/CS7367_Vision'
    archive = driveRoot / 'output/experiments/small/20261115T000643_647170Z'
    investigation = driveRoot / 'output/investigations/resolution_512/20261115T005403_964470Z'
    oldAudit = {r['image']: r for r in json.loads((investigation/'mask_audit.json').read_text())}
    archivedHashes = {name: fileHash(archive/name) for name in ['sample_manifest.csv','metrics.csv','experiment_summary.json']}
    assert archivedHashes['sample_manifest.csv'] == '0588df4ceceabe99b7d4fab5a6798d13df467b02863d9da42517a1edd907c7c0'
    with tempfile.TemporaryDirectory(prefix='real-geometry-', dir=fixtureRoot) as folder:
        root = Path(folder)
        assert root.resolve().is_relative_to(fixtureRoot.resolve())
        syntheticArchive = root / 'output/experiments/small/20261115T000643_647170Z'
        syntheticArchive.mkdir(parents=True)
        (syntheticArchive/'sample_manifest.csv').write_bytes((archive/'sample_manifest.csv').read_bytes())
        namespace, _, _, requests = execute(root, metadataOnly=True, realRoot=driveRoot/'mvtec_ad_2')
        assert 'caughtError' not in namespace and not requests
        assert namespace['summary']['archivedManifestComparison'] == {'matched':True,'selectedRows':360}
        functions = list(definitions(codeCells[4]).values())
        exec(compile(ast.Module(body=functions, type_ignores=[]),'mask_definitions','exec'), namespace)
        totals = dict(masks=0, nativeComponents=0, targetComponents=0, representedNativeComponents=0, masksWithMerging=0)
        for category in allCategories:
            masks, reasons = namespace['loadMasks'](namespace['selectedData'][category]['evaluation'],category)
            assert not reasons
            for item in [r for r in namespace['summary']['maskDiagnostics'] if r['category']==category]:
                prior = oldAudit[item['relativeImagePath']]
                assert item['native'] == prior['native']
                for key in ['sizeWH','values','area','components','bboxXYXY']:
                    assert item['target'][key] == prior['coverage512'][key]
                assert item['representedNativeComponents'] == item['native']['components']
                imagePath = driveRoot/'mvtec_ad_2'/item['relativeImagePath']
                maskPath = driveRoot/'mvtec_ad_2'/item['relativeMaskPath']
                assert fileHash(imagePath)==prior['imageSha256'] and fileHash(maskPath)==prior['maskSha256']
            outcome = namespace['summary']['maskAuditOutcomes'][-1]
            totals['masks'] += outcome['validMasks']
            for key in ['nativeComponents','targetComponents','representedNativeComponents','masksWithMerging']:
                totals[key] += outcome[key]
            del masks
            print(f'PASS real geometry only: {category} | {outcome}', flush=True)
        assert totals == dict(masks=80,nativeComponents=172,targetComponents=150,representedNativeComponents=172,masksWithMerging=12)
        assert all(fileHash(archive/name)==digest for name,digest in archivedHashes.items())
        evidence = dict(status='geometry_only_verified; no real model/Colab pilot run', totals=totals,
                        manifestRows=360, archivedHashes=archivedHashes, maskAuditOutcomes=namespace['summary']['maskAuditOutcomes'])
        (fixtureRoot/'Batch4_RealMask_Verification.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence,indent=2),flush=True)

if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    protocolChecks()
    if len(sys.argv)>1 and sys.argv[1]=='real_masks':
        realMasks()
    elif len(sys.argv)>1 and sys.argv[1]=='guards':
        guardTests()
    else:
        offlineTests()
