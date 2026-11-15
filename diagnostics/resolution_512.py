"""Bounded, non-production 512 feasibility investigation. No notebook or input writes.

Run tests/preflight offline first. Then run `python resolution_512.py audit`.
After audit review, `python resolution_512.py model <audit_run_directory>` uses
exactly the archived can/fabric identities and the audit-selected uniform policy.
Requires the production notebook's existing dependencies; no new packages.
"""
import ast
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
import numpy as np
import scipy
from scipy.ndimage import label as connectedComponents, gaussian_filter
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, auc
import cv2
import PIL
from PIL import Image
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F

repoRoot = Path(__file__).resolve().parents[1]
driveRoot = Path(os.environ.get('CS7367_DRIVE_ROOT', 'G:/My Drive'))
datasetRoot = driveRoot / 'ColabData/CS7367_Vision/mvtec_ad_2'
pilotRoot = driveRoot / 'ColabData/CS7367_Vision/output/experiments/small/20261115T000643_647170Z'
investigationRoot = driveRoot / 'ColabData/CS7367_Vision/output/investigations/resolution_512'
notebookPath = repoRoot / 'notebooks/CS7367_Research_Experiments.ipynb'
imageSize = 512
randomSeed = 7367
featureDimension = 32
batchSize = 5
covarianceRegularization = 0.01
workloadSize = 'small-investigation-only'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enhancementParameters = {'gamma': 0.7, 'claheClipLimit': 2.0, 'claheTileGrid': [8, 8]}
runStarted = time.perf_counter()
runTimestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
summary = {'status': 'incomplete', 'stageTimings': []}
stageStarts = {}

def fileHash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def memorySnapshot():
    # Native Windows counters: measured process working set/peak, not a tensor estimate.
    if os.name == 'nt':
        from ctypes import wintypes
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD]
        buffer = ctypes.create_string_buffer(80)
        struct.pack_into('<I', buffer, 0, 80)
        if not query(process(), buffer, 80):
            raise ctypes.WinError()
        peakBytes, residentBytes = struct.unpack_from('<QQ', buffer, 8)
        physical = ctypes.create_string_buffer(64)
        struct.pack_into('<I', physical, 0, 64)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(physical):
            raise ctypes.WinError()
        totalPhysical, availablePhysical = struct.unpack_from('<QQ', physical, 8)
        return dict(residentBytes=residentBytes, processLifetimePeakResidentBytes=peakBytes,
                    totalPhysicalBytes=totalPhysical, availablePhysicalBytes=availablePhysical,
                    method='Windows GetProcessMemoryInfo; peak is process-lifetime high-water mark')
    import resource
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(processLifetimePeakResidentBytes=int(maximum * (1 if sys.platform == 'darwin' else 1024)),
                method='resource.ru_maxrss process-lifetime high-water mark')

def coverageMask(native, target=512):
    """Positive area in the exact edge-aligned target-cell footprint, not dilation.

    Source pixel cells [x,x+1) overlap target cell [j*W/T,(j+1)*W/T).
    floor/ceil integer bounds and a summed-area table test any positive overlap.
    A positive target cell can contain only a fraction of foreground: this is a
    conservative occupancy protocol, not an accurate area/region-shape estimate.
    """
    if native.ndim != 2 or not np.isin(native, [0, 1, 255]).all():
        raise ValueError('Expected a two-dimensional binary mask.')
    height, width = native.shape
    y0 = np.arange(target) * height // target
    y1 = (np.arange(1, target+1) * height + target-1) // target
    x0 = np.arange(target) * width // target
    x1 = (np.arange(1, target+1) * width + target-1) // target
    integral = np.pad(np.cumsum(np.cumsum(native > 0, axis=0, dtype=np.int64), axis=1), ((1, 0), (1, 0)))
    occupied = integral[y1[:, None], x1] - integral[y0[:, None], x1] - integral[y1[:, None], x0] + integral[y0[:, None], x0]
    return (occupied > 0).astype(np.uint8)

def maskStats(mask):
    binary = mask > 0
    regions, count = connectedComponents(binary, structure=np.ones((3, 3)))
    y, x = np.nonzero(binary)
    return dict(sizeWH=[mask.shape[1], mask.shape[0]], values=np.unique(mask).tolist(), area=int(binary.sum()),
                components=int(count), bboxXYXY=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else None), regions

def readPair(relativePath):
    imagePath = datasetRoot / relativePath
    maskPath = imagePath.parent.parent / 'ground_truth/bad' / (imagePath.stem + '_mask.png')
    imageDigest, maskDigest = fileHash(imagePath), fileHash(maskPath)
    with Image.open(imagePath) as image:
        size = image.size
    with Image.open(maskPath) as image:
        image.load()
        native = np.asarray(image).copy()
        if image.size != size or native.ndim != 2 or not np.isin(native, [0, 255]).all() or not native.any():
            raise ValueError(f'Invalid or misaligned public annotation: {relativePath}')
    return imagePath, maskPath, native, imageDigest, maskDigest

def validateManifest(manifest):
    if len({r['relativeImagePath'] for r in manifest}) != len(manifest):
        raise ValueError('Duplicate/leaking manifest identities.')
    for row in manifest:
        relative = Path(row['relativeImagePath'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest path must be relative to the dataset.')
        if row['role'] == 'reference' and row['group'] == 'good' and row['label'] == '0':
            expected = Path(row['category']) / 'train/good'
        elif row['role'] == 'evaluation' and row['group'] in ['good', 'bad'] and row['label'] == str(int(row['group'] == 'bad')):
            expected = Path(row['category']) / 'test_public' / row['group']
        else:
            raise ValueError('Manifest role/group/label mismatch.')
        if relative.parent != expected:
            raise ValueError('Manifest split disagrees with its role; refuse evaluation-to-reference leakage.')

def loadBaselineFunctions():
    # Read definitions only: never execute notebook setup, sampling, or experiment cells.
    notebook = json.loads(notebookPath.read_text(encoding='utf-8'))
    names = {'logStage', 'enhanceImage', 'extractFeatures', 'fitPadim', 'auPro005'}
    definitions = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                   for node in ast.parse(cell['source']).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError('Required baseline functions not found.')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebookPath), 'exec'), globals())

def deriveGrid(model):
    with torch.inference_mode():
        x = torch.zeros((1, 3, 512, 512), device=device)
        x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        first = model.layer1(x)
        second = model.layer2(first)
        third = model.layer3(second)
    shapes = [list(layer.shape) for layer in [first, second, third]]
    if second.shape[-2] != second.shape[-1]:
        raise ValueError('Non-square feature grid for the fixed square input.')
    return int(second.shape[-1]), shapes

def preflight():
    global featureGrid, summary, extractFeatures
    loadBaselineFunctions()
    torch.manual_seed(randomSeed)
    torch.set_num_threads(2)  # CPU scheduling control only; no scientific parameter change.
    featureGrid, shapes = deriveGrid(resnet18(weights=None).eval().to(device))
    actualExtract = extractFeatures
    rng = np.random.default_rng(randomSeed)
    witnesses = []
    def syntheticFeatures(paths, condition):
        batch = rng.normal(size=(len(paths), featureGrid ** 2, featureDimension))
        witnesses.extend(batch[:, 0].copy())
        return batch
    extractFeatures = syntheticFeatures
    started = time.perf_counter()
    mean, precision = fitPadim([('good', Path(str(i))) for i in range(25)], 'synthetic', 'baseline')
    witness = np.asarray(witnesses)
    expected = np.cov(witness, rowvar=False) + covarianceRegularization * np.eye(featureDimension)
    np.testing.assert_allclose(mean[0], witness.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(precision[0], np.linalg.inv(expected), atol=1e-9)
    assert np.isfinite(precision).all()
    np.linalg.cholesky(precision)
    result = dict(status='pass', inputSize=512, featureGrid=featureGrid, layerShapes=shapes,
                  covarianceShape=list(precision.shape), bytesPerSpatialCovarianceArray=precision.nbytes,
                  elapsedSeconds=time.perf_counter()-started, memory=memorySnapshot())
    extractFeatures = actualExtract
    del mean, precision
    return result

def writeJson(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def audit():
    global summary, runTimestamp, runStarted, stageStarts
    runStarted = time.perf_counter()
    runTimestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    runRoot = investigationRoot / runTimestamp
    runRoot.mkdir(parents=True, exist_ok=False)
    summary = dict(status='incomplete', purpose='NOT PRODUCTION RESULTS: bounded resolution/protocol feasibility',
                   runUtc=runTimestamp, outputPath=str(runRoot), targetSize=512, baselinePath=str(pilotRoot),
                   stageTimings=[], errors=[], sourceNotebookSha256=fileHash(notebookPath))
    stageStarts = {}
    writeJson(runRoot / 'investigation_summary.json', summary)
    loadBaselineFunctions()
    logStage('START', 'all-eight selected anomaly mask audit')
    try:
        baseline = json.loads((pilotRoot / 'experiment_summary.json').read_text())
        if baseline['status'] != 'complete' or baseline['seed'] != 7367:
            raise ValueError('Archived pilot is incomplete or has a different seed.')
        with (pilotRoot / 'sample_manifest.csv').open(newline='') as handle:
            manifest = list(csv.DictReader(handle))
        validateManifest(manifest)
        for category, counts in baseline['actualCounts'].items():
            expected = {'reference': 25, 'evaluation': 20, 'normalEvaluation': 10, 'anomalyEvaluation': 10}
            if counts != expected:
                raise ValueError(f'Unexpected fixed pilot workload: {category} {counts}')
        anomalies = [r for r in manifest if r['role'] == 'evaluation' and r['group'] == 'bad' and r['label'] == '1']
        if len(anomalies) != 80 or len({r['category'] for r in anomalies}) != 8:
            raise ValueError('Expected all 80 selected public anomalies across eight categories.')
        required = {'can/test_public/bad/002_shift_1.png', 'fabric/test_public/bad/007_overexposed.png'}
        if not required.issubset({r['relativeImagePath'] for r in anomalies}):
            raise ValueError('Known tiny masks are absent from the archived manifest.')
        summary['archiveHashes'] = {name: fileHash(pilotRoot / name) for name in ['experiment_summary.json', 'metrics.csv', 'sample_manifest.csv']}
        (runRoot / 'sample_manifest.csv').write_bytes((pilotRoot / 'sample_manifest.csv').read_bytes())
        auditRows = []
        for processed, row in enumerate(anomalies, 1):
            imagePath, maskPath, native, imageDigest, maskDigest = readPair(row['relativeImagePath'])
            nativeStats, nativeRegions = maskStats(native)
            record = dict(category=row['category'], image=row['relativeImagePath'], mask=maskPath.relative_to(datasetRoot).as_posix(),
                          imageSha256=imageDigest, maskSha256=maskDigest, native=nativeStats,
                          aspectRatio=native.shape[1]/native.shape[0],
                          scaleX=512/native.shape[1], scaleY=512/native.shape[0],
                          axisStretchRatio=native.shape[1]/native.shape[0])
            for name, target in [('archived_nearest256', 256), ('nearest512', 512), ('coverage512', 512)]:
                result = coverageMask(native) if name == 'coverage512' else np.asarray(Image.fromarray(native).resize((target, target), Image.Resampling.NEAREST))
                stats, _ = maskStats(result)
                sampledRegions = np.asarray(Image.fromarray(nativeRegions.astype(np.int32)).resize((target, target), Image.Resampling.NEAREST))
                stats['nativeComponentsRepresented'] = nativeStats['components'] if name == 'coverage512' else int(len(np.unique(sampledRegions[sampledRegions > 0])))
                stats['mappedNativeBBoxXYXY'] = [nativeStats['bboxXYXY'][i] * target / (native.shape[1] if i % 2 == 0 else native.shape[0]) for i in range(4)]
                stats['nativeEquivalentArea'] = stats['area'] * native.size / target ** 2
                record[name] = stats
            if fileHash(imagePath) != imageDigest or fileHash(maskPath) != maskDigest:
                raise ValueError(f'Source immutability failure: {imagePath}')
            auditRows.append(record)
            if processed % 5 == 0:
                logStage('PROGRESS', 'all-eight selected anomaly mask audit', processed=processed, total=len(anomalies))
        logStage('COMPLETE', 'all-eight selected anomaly mask audit', processed=80, total=80)
        writeJson(runRoot / 'mask_audit.json', auditRows)
        summary['auditCounts'] = {policy: dict(audited=80, disappeared=sum(r[policy]['area'] == 0 for r in auditRows),
                                             componentCountChanged=sum(r[policy]['components'] != r['native']['components'] for r in auditRows),
                                             nativeComponentsLost=sum(r['native']['components']-r[policy]['nativeComponentsRepresented'] for r in auditRows))
                                  for policy in ['archived_nearest256', 'nearest512', 'coverage512']}
        # Select on annotation retention only, before looking at any new model scores.
        nearest = summary['auditCounts']['nearest512']
        summary['proposedMaskPolicy'] = 'nearest512' if nearest['disappeared'] == 0 and nearest['nativeComponentsLost'] == 0 else 'coverage512'
        summary['modelGate'] = 'pass' if summary['auditCounts'][summary['proposedMaskPolicy']]['disappeared'] == 0 else 'fail'
        summary['status'] = 'audit_complete'
        summary['totalRuntimeSeconds'] = time.perf_counter() - runStarted
        summary['memory'] = memorySnapshot()
        summary['allAuditSourcesUnchanged'] = True
        writeJson(runRoot / 'investigation_summary.json', summary)
        print('AUDIT EVIDENCE:', runRoot, flush=True)
        print(json.dumps(summary['auditCounts'], indent=2), flush=True)
        return runRoot
    except BaseException as error:
        summary['errors'].append(f'{type(error).__name__}: {error}')
        writeJson(runRoot / 'investigation_summary.json', summary)
        raise

def modelRun(runRoot):
    global summary, stageStarts, runStarted, runTimestamp, featureGrid, backbone, featureIndices, normalizationMean, normalizationStd
    runRoot = Path(runRoot)
    if not runRoot.resolve().is_relative_to(investigationRoot.resolve()):
        raise ValueError('Model evidence must stay within the distinct investigation output root.')
    summary = json.loads((runRoot / 'investigation_summary.json').read_text())
    if summary['status'] != 'audit_complete' or summary['modelGate'] != 'pass':
        raise ValueError('A completed passing audit is required; cannot rerun/overwrite a model investigation.')
    if fileHash(notebookPath) != summary['sourceNotebookSha256']:
        raise ValueError('Production notebook changed after audit.')
    for name, digest in summary['archiveHashes'].items():
        if fileHash(pilotRoot / name) != digest:
            raise ValueError('Archived pilot changed after audit; stop before model execution.')
    if fileHash(runRoot / 'sample_manifest.csv') != summary['archiveHashes']['sample_manifest.csv']:
        raise ValueError('Investigation manifest differs from the archived identities.')
    summary['status'] = 'model_incomplete'
    summary['auditRuntimeSeconds'] = summary['totalRuntimeSeconds']
    summary.update(seed=7367, modelCategories=['can', 'fabric'],
                   conditions=['baseline', 'gamma', 'histogram_equalization', 'clahe'],
                   splitPolicy='train/good only for fitting; test_public/good and test_public/bad only for evaluation',
                   maskPolicySemantics='Exact positive-source-cell footprint occupancy; conservative binary quantization, not precise region-shape reconstruction')
    runTimestamp, runStarted, stageStarts = summary['runUtc'], time.perf_counter(), {}
    loadBaselineFunctions()
    baseline = json.loads((pilotRoot / 'experiment_summary.json').read_text())
    with (runRoot / 'sample_manifest.csv').open(newline='') as handle:
        manifest = list(csv.DictReader(handle))
    validateManifest(manifest)
    modelManifest = [r for r in manifest if r['category'] in ['can', 'fabric']]
    with (runRoot / 'model_sample_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(modelManifest)
    with (pilotRoot / 'metrics.csv').open(newline='') as handle:
        oldMetrics = list(csv.DictReader(handle))
    auditRows = json.loads((runRoot / 'mask_audit.json').read_text())
    sourceHashes = {r['mask']: r['maskSha256'] for r in auditRows}
    sourceHashes.update({r['image']: r['imageSha256'] for r in auditRows})
    torch.manual_seed(7367)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(Path(os.environ.get('CS7367_TORCH_CACHE', str(Path.home() / '.cache/torch/hub')))))
    metrics, comparisons, policyMetrics = [], [], []
    writeJson(runRoot / 'investigation_summary.json', summary)
    try:
        logStage('START', 'pretrained backbone loading')
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().requires_grad_(False).to(device)
        featureGrid, layerShapes = deriveGrid(backbone)
        featureIndices = torch.tensor(baseline['model']['featureIndices'], dtype=torch.long, device=device)
        generated = torch.randperm(448, generator=torch.Generator().manual_seed(7367))[:32].tolist()
        if generated != featureIndices.cpu().tolist() or featureGrid != 64:
            raise ValueError('Unexpected feature identities or resolution-derived layer2 grid.')
        normalizationMean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        normalizationStd = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        logStage('COMPLETE', 'pretrained backbone loading')
        summary.update(model={'backbone': 'ResNet18 IMAGENET1K_V1', 'inputSize': 512, 'layerShapes': layerShapes,
                              'featureGrid': featureGrid, 'featureIndices': featureIndices.cpu().tolist(),
                              'featureDimension': 32, 'covarianceRegularization': 0.01, 'batchSize': 5, 'smoothingSigma': 4},
                       environment={'device': str(device), 'python': platform.python_version(), 'torch': torch.__version__,
                                    'torchvision': torchvision.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                                    'sklearn': sklearn.__version__, 'pillow': PIL.__version__, 'opencv': cv2.__version__, 'cpuThreads': torch.get_num_threads()},
                       archivedEnvironment=baseline['environment'], enhancementParameters=baseline['enhancementParameters'],
                       comparisonCaveat='Descriptive resolution/protocol comparison only: grid, mask protocol (if changed), libraries and hardware differ; no causal attribution to resolution alone.')
        for category in ['can', 'fabric']:
            selected = [r for r in manifest if r['category'] == category]
            references = [(r['group'], datasetRoot / r['relativeImagePath']) for r in selected if r['role'] == 'reference']
            evaluations = [(r['group'], datasetRoot / r['relativeImagePath']) for r in selected if r['role'] == 'evaluation']
            if len(references) != 25 or len(evaluations) != 20 or sum(g == 'bad' for g, _ in evaluations) != 10:
                raise ValueError('Fixed workload changed; no silent reduction permitted.')
            if {p.resolve() for _, p in references} & {p.resolve() for _, p in evaluations}:
                raise ValueError('Reference/evaluation leakage.')
            for _, path in references + evaluations:
                sourceHashes.setdefault(path.relative_to(datasetRoot).as_posix(), fileHash(path))
            masks = np.zeros((20, 512, 512), dtype=np.uint8)
            nearestMasks = np.zeros_like(masks)
            for index, (group, path) in enumerate(evaluations):
                if group == 'bad':
                    _, currentMaskPath, native, _, currentMaskDigest = readPair(path.relative_to(datasetRoot).as_posix())
                    if currentMaskDigest != sourceHashes[currentMaskPath.relative_to(datasetRoot).as_posix()]:
                        raise ValueError('Audited annotation changed before model evaluation.')
                    masks[index] = coverageMask(native) if summary['proposedMaskPolicy'] == 'coverage512' else np.asarray(Image.fromarray(native).resize((512, 512), Image.Resampling.NEAREST)) > 0
                    nearestMasks[index] = np.asarray(Image.fromarray(native).resize((512, 512), Image.Resampling.NEAREST)) > 0
                    if not masks[index].any():
                        raise ValueError('Selected anomaly lost under the audited policy.')
            labels = np.array([int(group == 'bad') for group, _ in evaluations])
            for condition in ['baseline', 'gamma', 'histogram_equalization', 'clahe']:
                started = time.perf_counter()
                logStage('START', 'condition', category, condition)
                mean, precision = fitPadim(references, category, condition)
                if not np.isfinite(mean).all() or not np.isfinite(precision).all():
                    raise ValueError('Nonfinite Gaussian parameters.')
                scores = np.empty((20, 512, 512), dtype=np.float32)
                logStage('START', 'inference', category, condition, 0, 20)
                for start in range(0, 20, 5):
                    difference = extractFeatures([p for _, p in evaluations[start:start+5]], condition) - mean
                    distances = np.sqrt(np.maximum(0, np.einsum('npd,pde,npe->np', difference, precision, difference, optimize=True)))
                    maps = F.interpolate(torch.from_numpy(distances.reshape(-1, 1, featureGrid, featureGrid)), size=(512, 512), mode='bilinear', align_corners=False).numpy()[:, 0]
                    for offset, scoreMap in enumerate(maps):
                        scores[start+offset] = gaussian_filter(scoreMap, sigma=4)
                    logStage('PROGRESS', 'inference', category, condition, start+5, 20)
                logStage('COMPLETE', 'inference', category, condition, 20, 20)
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite anomaly scores.')
                logStage('START', 'metric evaluation', category, condition)
                proValue, proStatus = auPro005(masks, scores)
                # Same fixed scores: quantify annotation-policy sensitivity without another model fit.
                if all(nearestMasks[i].any() for i, label in enumerate(labels) if label == 1):
                    nearestValue, nearestStatus = auPro005(nearestMasks, scores)
                else:
                    nearestValue, nearestStatus = None, 'unavailable: selected anomaly lost under NEAREST512'
                policyMetrics.append(dict(category=category, condition=condition, fixedInputSize=512, fixedFeatureGrid=featureGrid,
                                          selectedPolicy=summary['proposedMaskPolicy'], selectedPolicyAuPro=proValue,
                                          nearest512AuPro=nearestValue, nearest512Status=nearestStatus,
                                          caveat='Annotation-policy-only contrast on identical 512 scores; not a second model run.'))
                imageValue = float(roc_auc_score(labels, scores.max(axis=(1, 2))))
                logStage('COMPLETE', 'metric evaluation', category, condition)
                logStage('COMPLETE', 'condition', category, condition)
                elapsed = time.perf_counter()-started
                for name, value, status in [('image_AUROC', imageValue, 'available'), ('AU-PRO0.05', proValue, proStatus)]:
                    metrics.append(dict(dataset='MVTec AD 2', category=category, condition=condition, workloadMode='small', referenceCount=25,
                                        evaluationCount=20, normalEvaluationCount=10, anomalyEvaluationCount=10, metricName=name,
                                        metricValue=value, metricStatus=status, runtimeSeconds=elapsed))
                    old = next(r for r in oldMetrics if r['category'] == category and r['condition'] == condition and r['metricName'] == name)
                    comparisons.append(dict(category=category, condition=condition, metric=name, archived256Value=old['metricValue'],
                                            archived256Status=old['metricStatus'], investigation512Value=value, investigation512Status=status,
                                            descriptiveDifference=None if not old['metricValue'] or value is None else value-float(old['metricValue'])))
                summary.setdefault('conditionMemory', []).append(dict(category=category, condition=condition, **memorySnapshot()))
                summary['modelRuntimeSeconds'] = time.perf_counter()-runStarted
                summary['totalRuntimeSeconds'] = summary['auditRuntimeSeconds'] + summary['modelRuntimeSeconds']
                with (runRoot / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
                    writer.writeheader()
                    writer.writerows(metrics)
                writeJson(runRoot / 'comparison_summary.json', comparisons)
                writeJson(runRoot / 'mask_policy_metrics.json', policyMetrics)
                writeJson(runRoot / 'investigation_summary.json', summary)
                del mean, precision, scores
        for path, digest in sourceHashes.items():
            if fileHash(datasetRoot / path) != digest:
                raise ValueError(f'Original input changed: {path}')
        for name, digest in summary['archiveHashes'].items():
            if fileHash(pilotRoot / name) != digest:
                raise ValueError('Archived evidence changed.')
        if fileHash(notebookPath) != summary['sourceNotebookSha256']:
            raise ValueError('Production notebook changed.')
        summary.update(status='complete', modelRuntimeSeconds=time.perf_counter()-runStarted, allModelSourcesUnchanged=True,
                       allArchivedEvidenceUnchanged=True, productionNotebookUnchanged=True, finalMemory=memorySnapshot())
        summary['totalRuntimeSeconds'] = summary['auditRuntimeSeconds'] + summary['modelRuntimeSeconds']
        writeJson(runRoot / 'investigation_summary.json', summary)
        print('INVESTIGATION COMPLETE:', runRoot, flush=True)
    except BaseException as error:
        summary['errors'].append(f'{type(error).__name__}: {error}')
        writeJson(runRoot / 'investigation_summary.json', summary)
        raise

if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    phase = sys.argv[1] if len(sys.argv) > 1 else 'preflight'
    if phase == 'preflight':
        print(json.dumps(preflight(), indent=2), flush=True)
    elif phase == 'audit':
        audit()
    elif phase == 'model' and len(sys.argv) == 3:
        modelRun(sys.argv[2])
    else:
        raise ValueError('Use preflight, audit, or model <completed_audit_run_directory>.')
