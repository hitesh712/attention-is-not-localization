"""VinDr-Mammo (PhysioNet release 1.0.0, original DICOMs): selection, box annotations and image readers shared by the
revision scripts. The images are the ones fetched by e2e/download_vindr_zip.py (D:/vindr_dicom/selection.csv):
every image with a mass or suspicious-calcification box (both official splits; nothing is trained on VinDr-Mammo)
and 300 BI-RADS 1 images of the official test split as normal controls.

Boxes are in the pixel grid of the DICOM image (verified: the annotation height/width equal Rows/Columns).
Box convention here: (y0, y1, x0, x1) with exclusive ends, clipped to the image.
"""
import os, csv, ast
import numpy as np

ROOT = 'D:/vindr_dicom'
LESION_SETS = ('mass_test', 'calc_test', 'mass_training', 'calc_training'); NORMAL_SET = 'normal_test_birads1'
TARGETS = ('Mass', 'Suspicious Calcification')             # the two lesion types the CBIS-DDSM models were trained on


def path(study, image):
    return f'{ROOT}/images/{study}/{image}.dicom'


def selection(sets=None, existing_only=True):
    rows = [r for r in csv.DictReader(open(ROOT + '/selection.csv')) if sets is None or r['set'] in sets]
    return [r for r in rows if not existing_only or os.path.exists(path(r['study_id'], r['image_id']))]


def annotations():
    """image_id -> dict(study, H, W, view, laterality, split, boxes=[dict(cats, mass, calc, target, box, birads)])."""
    A = {}
    for r in csv.DictReader(open(ROOT + '/finding_annotations.csv', encoding='utf-8')):
        a = A.setdefault(r['image_id'], dict(study=r['study_id'], H=int(r['height']), W=int(r['width']), view=r['view_position'],
                                             laterality=r['laterality'], split=r['split'], boxes=[]))
        if r['xmin'] in ('', None): continue
        cats = ast.literal_eval(r['finding_categories']); H, W = a['H'], a['W']
        x0, y0 = max(0, int(np.floor(float(r['xmin'])))), max(0, int(np.floor(float(r['ymin']))))
        x1, y1 = min(W, int(np.ceil(float(r['xmax'])))), min(H, int(np.ceil(float(r['ymax']))))
        if x1 - x0 < 1 or y1 - y0 < 1: continue
        a['boxes'].append(dict(cats=cats, mass='Mass' in cats, calc='Suspicious Calcification' in cats,
                               target=any(c in cats for c in TARGETS), box=(y0, y1, x0, x1), birads=r['finding_birads']))
    return A


def inside(box, r, c):
    return box[0] <= r < box[1] and box[2] <= c < box[3]


def box_mask(shape, boxes):
    g = np.zeros(shape, bool)
    for y0, y1, x0, x1 in boxes: g[y0:y1, x0:x1] = True
    return g


VOI = True      # apply the display transform stored in each DICOM (False: plain min-max of the stored values, as for INbreast)


def display_pixels(d, voi=None):
    """Stored pixel values -> display values, following the DICOM grayscale pipeline: the file's own VOI LUT (Siemens:
    VOI LUT Sequence; Planmed, IMS Giotto: window centre / width; the first one when several are offered), then inversion
    of MONOCHROME1 images. Without the VOI LUT the breast occupies a narrow band of the stored range (for IMS Giotto
    about a tenth of it), so the images are much flatter than the displayed mammogram."""
    try: from pydicom.pixels import apply_voi_lut
    except ImportError: from pydicom.pixel_data_handlers.util import apply_voi_lut
    a = d.pixel_array
    if VOI if voi is None else voi: a = apply_voi_lut(a, d)
    a = np.asarray(a, np.float32)
    if str(getattr(d, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1': a = a.max() - a   # stored inverted
    return a


def _pixels(study, image):
    import pydicom
    d = pydicom.dcmread(path(study, image))
    return display_pixels(d), d


def read_gray(study, image):
    """8-bit image: display values (see display_pixels), min-max scaled as for INbreast."""
    a, _ = _pixels(study, image)
    return ((a - a.min()) / (a.max() - a.min() + 1e-6) * 255).astype(np.uint8)


def read_float(study, image):
    """[0, 1] float image (same transform, no quantisation), for the models with their own preprocessing."""
    a, _ = _pixels(study, image)
    return (a - a.min()) / (a.max() - a.min() + 1e-6)
