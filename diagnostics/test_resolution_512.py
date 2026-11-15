"""Offline 512-only shape, geometry, metric and immutability checks."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

modulePath = Path(__file__).with_name('resolution_512.py')
spec = importlib.util.spec_from_file_location('resolution512', modulePath)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)

def independentCoverage(mask):
    # Source-centric exact footprint oracle, independent of summed-area implementation.
    height, width = mask.shape
    result = np.zeros((512, 512), dtype=np.uint8)
    for y, x in zip(*np.nonzero(mask)):
        y0, y1 = y * 512 // height, ((y+1) * 512 + height-1) // height
        x0, x1 = x * 512 // width, ((x+1) * 512 + width-1) // width
        result[y0:y1, x0:x1] = 1
    return result

def tests():
    validRow = dict(category='can', role='reference', group='good', label='0', relativeImagePath='can/train/good/000.png')
    diagnostic.validateManifest([validRow])
    for changed in [dict(role='evaluation'), dict(group='bad'), dict(relativeImagePath='can/test_public/good/000.png'), dict(relativeImagePath='../other/000.png')]:
        try:
            diagnostic.validateManifest([{**validRow, **changed}])
            raise AssertionError('Invalid manifest accepted')
        except ValueError:
            pass
    try:
        diagnostic.validateManifest([validRow, validRow])
        raise AssertionError('Duplicate identity accepted')
    except ValueError:
        pass
    try:
        diagnostic.modelRun(Path(__file__).parent)
        raise AssertionError('Non-investigation evidence root accepted')
    except ValueError:
        pass
    print('PASS: manifest role/split/label, traversal, duplicate/leakage guards and bounded evidence-root guard', flush=True)
    rng = np.random.default_rng(7367)
    cases = []
    for shape in [(512, 512), (1024, 2232), (2048, 2448), (13, 17)]:
        mask = np.zeros(shape, dtype=np.uint8)
        for _ in range(12):
            mask[rng.integers(shape[0]), rng.integers(shape[1])] = 255
        cases.append(mask)
    solid = np.zeros((1024, 1024), dtype=np.uint8)
    solid[200:400, 300:600] = 255
    cases.append(solid)
    for native in cases:
        before = native.copy()
        resized = diagnostic.coverageMask(native)
        assert np.array_equal(resized, independentCoverage(native))
        assert np.array_equal(resized, diagnostic.coverageMask(native))
        assert np.array_equal(native, before)
        assert set(np.unique(resized)).issubset({0, 1}) and resized.shape == (512, 512)
    assert np.array_equal(diagnostic.coverageMask(cases[0]), cases[0] > 0)
    assert np.all(diagnostic.coverageMask(solid)[100:200, 150:300] == 1)
    assert diagnostic.coverageMask(solid).sum() == 100 * 150
    tiny = np.zeros((1024, 2232), dtype=np.uint8)
    tiny[0, 0] = 255
    assert not np.asarray(Image.fromarray(tiny).resize((512, 512), Image.Resampling.NEAREST)).any()
    assert diagnostic.coverageMask(tiny)[0, 0] == 1
    assert diagnostic.coverageMask(tiny).sum() == 1
    assert not diagnostic.coverageMask(np.zeros((1024, 2232), dtype=np.uint8)).any()
    try:
        diagnostic.coverageMask(np.full((10, 10), 2, dtype=np.uint8))
        raise AssertionError('Nonbinary mask accepted')
    except ValueError:
        pass
    print('PASS: five footprint-oracle cases, deterministic binary masks, exact alignment/rectangle shape, no invented positives, vanished tiny component retained', flush=True)
    fixtureRoot = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='resolution512-test-', dir=fixtureRoot) as folder:
        root = Path(folder)
        assert root.resolve().is_relative_to(fixtureRoot)
        diagnostic.datasetRoot = root
        imagePath = root / 'can/test_public/bad/tiny.png'
        maskPath = root / 'can/test_public/ground_truth/bad/tiny_mask.png'
        imagePath.parent.mkdir(parents=True)
        maskPath.parent.mkdir(parents=True)
        Image.new('RGB', (2232, 1024)).save(imagePath)
        Image.fromarray(tiny).save(maskPath)
        before = (diagnostic.fileHash(imagePath), diagnostic.fileHash(maskPath))
        _, _, native, _, _ = diagnostic.readPair('can/test_public/bad/tiny.png')
        diagnostic.coverageMask(native)
        assert before == (diagnostic.fileHash(imagePath), diagnostic.fileHash(maskPath))
        Image.fromarray(tiny[:100]).save(maskPath)
        try:
            diagnostic.readPair('can/test_public/bad/tiny.png')
            raise AssertionError('Dimension mismatch accepted')
        except ValueError:
            pass
        maskPath.unlink()
        try:
            diagnostic.readPair('can/test_public/bad/tiny.png')
            raise AssertionError('Missing mask accepted')
        except FileNotFoundError:
            pass
    diagnostic.loadBaselineFunctions()
    masks = np.zeros((2, 512, 512), dtype=np.uint8)
    masks[0, 20:30, 20:30] = 1
    assert diagnostic.auPro005(masks, masks.astype(float))[0] == 1
    assert diagnostic.auPro005(np.zeros_like(masks), masks.astype(float))[0] is None
    assert diagnostic.auPro005(masks, masks[:1].astype(float))[0] is None
    print('PASS: original file hashes unchanged; missing/misaligned masks rejected; exact AU-PRO and unsupported degenerate masks', flush=True)
    result = diagnostic.preflight()
    assert result['featureGrid'] == 64
    assert result['layerShapes'] == [[1, 64, 128, 128], [1, 128, 64, 64], [1, 256, 32, 32]]
    assert result['covarianceShape'] == [4096, 32, 32]
    assert result['bytesPerSpatialCovarianceArray'] == 33554432
    print('PASS: offline actual ResNet18 512 layer shapes, 64x64 grid, 25-reference covariance/inverse accuracy, finite positive-definite precision, native peak-memory measurement', flush=True)
    return result

if __name__ == '__main__':
    import json
    print(json.dumps(tests(), indent=2), flush=True)
