"""Optional native-grid CT isosurfaces with the published scene display presets."""
from functools import lru_cache
import json
from pathlib import Path
import struct

import numpy as np


RGBA = [.72, .74, .77, .5]
SCENE_YAWS = dict(chest=-90, abdomen=180, foot=90, head=180,
                  pancreas=0, pine=-35, seashell=35, walnut=0)


@lru_cache(maxsize=1)
def mesh_dependencies():
    try:
        from scipy import ndimage
        from skimage import measure
        from skimage.filters import threshold_multiotsu
        from skimage.morphology import convex_hull_image
        import trimesh
        import fast_simplification
    except ImportError as error:
        raise ImportError('Mesh export requires the packages in environment.yml: '
                          'scikit-image, trimesh and fast-simplification.') from error
    return ndimage, measure, threshold_multiotsu, convex_hull_image, trimesh, fast_simplification


def infer_scene(model_path, config):
    """Known model folders and saved source paths select their display recipe."""
    for value in [config.get('source_path'), config.get('model_path'), model_path]:
        if not value:
            continue
        path = Path(value)
        for token in [path.stem, *reversed(path.parts)]:
            token = token.lower()
            if token.endswith('_50'):
                token = token[:-3]
            if token in SCENE_YAWS:
                return token
    return None


def mask_resolution(resolution):
    """A small auxiliary field locates support arcs; the exported surface stays native."""
    factor = min(1., 256. / max(resolution))
    return [max(2, int(np.ceil(n * factor))) for n in resolution]


def mesh_rotation(scene):
    if scene == 'head':
        return np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    if scene == 'foot':
        return np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
    return np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def _disk(radius, shape):
    if radius == 0:
        return np.ones((1, 1, 1), dtype=bool)
    rx, ry = [max(1, int(round(radius * n / 256.))) for n in shape[:2]]
    x, y = np.ogrid[-rx:rx+1, -ry:ry+1]
    return ((x / rx)**2 + (y / ry)**2 <= 1.)[..., None]


def _body_mask(volume, scale, scene, center, size, reference_geometry):
    ndimage, _, _, convex_hull_image, _, _ = mesh_dependencies()
    # Match the published preview quantization when selecting the body envelope.
    density = np.rint(np.clip(volume / scale, 0., 1.) * 4095) * (scale / 4095.)
    core = density >= (.06 if scene == 'abdomen' else .03)
    opening, margin = (2, 2) if scene == 'abdomen' else (0, 3)
    opened = ndimage.binary_opening(core, structure=_disk(opening, core.shape))
    labels, _ = ndimage.label(opened)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if not counts.any():
        return np.ones(core.shape, dtype=bool)
    body = labels == counts.argmax()
    keep = np.zeros(core.shape, dtype=bool)
    for z in range(core.shape[2]):
        section = body[:, :, z]
        xy = np.argwhere(section)
        if len(xy) >= 3 and (np.ptp(xy, axis=0) > 0).all():
            keep[:, :, z] = convex_hull_image(section)
        else:
            keep[:, :, z] = section
    keep = ndimage.binary_dilation(keep, structure=_disk(margin, keep.shape))
    # Restrict cleanup to the identified posterior support band. Other tissue
    # and small components remain untouched, including with custom CT extents.
    reference_geometry = reference_geometry or {}
    reference_center = np.asarray(reference_geometry.get('offOrigin', center))
    reference_size = np.asarray(reference_geometry.get('sVoxel', size))
    fraction = 211. / 256. if scene == 'abdomen' else 163. / 256.
    guard = reference_center[1] + (fraction - .5) * reference_size[1]
    y = center[1] + ((np.arange(keep.shape[1]) + .5) / keep.shape[1] - .5) * size[1]
    keep[:, y < guard, :] = True
    return keep


def _level(volume, scene):
    if scene == 'chest':
        return .55
    _, _, threshold_multiotsu, _, _, _ = mesh_dependencies()
    hist, _ = np.histogram(volume, bins=256)
    if np.count_nonzero(hist) < 4:
        return (float(volume.min()) + float(volume.max())) / 2. if volume.min() < volume.max() else float(volume.max()) / 2.
    cuts = threshold_multiotsu(volume, classes=4)
    return float(cuts[0 if scene == 'pine' else 1])


def _glb(mesh, trimesh):
    material = trimesh.visual.material.PBRMaterial(
        name='Light gray, 50% opacity', baseColorFactor=RGBA, metallicFactor=0.,
        roughnessFactor=.72, alphaMode='BLEND', doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    data = trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh), include_normals=True)
    length, kind = struct.unpack_from('<II', data, 12)
    assert kind == 0x4e4f534a
    doc = json.loads(data[20:20+length])
    # Preserve exact float opacity/color instead of an 8-bit material rounding.
    for mat in doc['materials']:
        mat.update(alphaMode='BLEND', doubleSided=True)
        mat['pbrMetallicRoughness'].update(baseColorFactor=RGBA, metallicFactor=0., roughnessFactor=.72)
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    tail = data[20+length:]
    return (struct.pack('<III', 0x46546c67, 2, 20+len(encoded)+len(tail))
            + struct.pack('<II', len(encoded), 0x4e4f534a) + encoded + tail)


def export_mesh(volume, output, center, size, scene=None, roi_volume=None,
                reference_geometry=None, level=None):
    """Export a GLB without changing the reconstructed CT array or point attributes."""
    ndimage, measure, _, _, trimesh, simplification = mesh_dependencies()
    output = Path(output)
    if output.suffix.lower() != '.glb':
        raise ValueError('The mesh output must be a .glb filename')
    volume = np.asarray(volume)
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError('Mesh extraction needs at least two voxels per axis')
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center, size]).all() or (size <= 0).any():
        raise ValueError('Mesh geometry needs a finite center and positive XYZ size')
    if not np.isfinite(volume).all() or float(volume.max()) <= 0:
        raise ValueError('Cannot extract a mesh from an empty or nonfinite CT volume')
    v = np.maximum(volume, 0.).astype(np.float32, copy=True)
    scale = float(v.max())
    # The approved Chest display shows high-density structures. The original
    # full CT remains in volume.npy; its soft tissue is not discarded there.
    if scene == 'chest' and level is None:
        v = np.rint(v / scale * 4095) * (scale / 4095.)
    isovalue = _level(v, scene) if level is None else float(level)
    if not np.isfinite(isovalue) or not 0 < isovalue < float(v.max()):
        raise ValueError(f'No CT isosurface at level {isovalue:g}')
    if scene != 'chest':
        labels, _ = ndimage.label(v >= isovalue)
        counts = np.bincount(labels.ravel())
        largest = int(counts[1:].max())
        minimum = min(largest, max(32, int(largest * 1e-5)))
        keep = counts >= minimum
        keep[0] = False
        v[(v >= isovalue) & ~keep[labels]] = 0
        del labels, counts, keep
    spacing = size / np.asarray(v.shape)
    vertices, faces, _, _ = measure.marching_cubes(
        np.pad(v, 1), level=isovalue, spacing=spacing, allow_degenerate=False)
    vertices += center - size / 2. - spacing / 2.
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    limit = 250000 if scene == 'chest' else 160000
    if len(mesh.faces) > limit:
        vertices, faces = simplification.simplify(mesh.vertices, mesh.faces, target_count=limit)
        mesh = trimesh.Trimesh(vertices, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=.5, nu=.53, iterations=4)
    mesh.fix_normals(multibody=True)
    if scene in ['abdomen', 'pancreas']:
        roi = volume if roi_volume is None else np.asarray(roi_volume)
        if roi.ndim != 3 or not np.isfinite(roi).all():
            raise ValueError('The auxiliary mesh ROI field must be finite and XYZ')
        keep = _body_mask(roi, scale, scene, center, size, reference_geometry)
        uvw = (mesh.vertices - center) / size + .5
        indices = np.clip(np.floor(uvw * keep.shape).astype(int), 0, np.array(keep.shape)-1)
        selected = keep[indices[:, 0], indices[:, 1], indices[:, 2]]
        mesh.update_faces(np.all(selected[mesh.faces], axis=1))
        mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('No mesh remains inside the scene display envelope')
    rotation = mesh_rotation(scene)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    mesh.apply_transform(transform)
    mesh.metadata.update(scene=scene or 'generic', isovalue=isovalue, visualization_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_glb(mesh, trimesh))
    return dict(file=output.name, scene=scene or 'generic', isovalue=isovalue,
                vertices=len(mesh.vertices), faces=len(mesh.faces), rgba=RGBA,
                native_to_mesh=rotation.tolist(),
                camera_orbit=f'{SCENE_YAWS.get(scene, 0)}deg 75deg 135%',
                visualization_only=True)
