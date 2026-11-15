"""Analytic geometry and input-conservation checks for optional CT mesh export."""
from pathlib import Path
import json
import math
import struct
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xct.mesh import export_mesh, infer_scene, mesh_rotation, RGBA


def main():
    center = np.array([.7, -.4, 1.2])
    size = np.array([2., 3., 4.])
    sigma = np.array([.25, .4, .6])
    shape = (32, 40, 48)
    axes = [(np.arange(n)+.5)*s/n-s/2 for n, s in zip(shape, size)]
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    volume = np.exp(-.5*((xyz/sigma)**2).sum(-1)).astype(np.float32)
    original = volume.copy()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for scene in [None, 'foot', 'head']:
            path = root/f'{scene or "generic"}.glb'
            info = export_mesh(volume, path, center, size, scene=scene, level=.5)
            mesh = trimesh.load(path, force='mesh', process=False)
            rotation = mesh_rotation(scene)
            expected_center = rotation @ center
            extent = abs(rotation) @ (math.sqrt(2*math.log(2))*sigma)
            expected = np.stack([expected_center-extent, expected_center+extent])
            assert np.allclose(mesh.bounds, expected, atol=1.5*max(size/shape)), (scene, mesh.bounds, expected)
            assert np.array_equal(volume, original)
            assert info['vertices'] == len(mesh.vertices) and info['faces'] == len(mesh.faces)
            data = path.read_bytes()
            assert data[:4] == b'glTF'
            length = struct.unpack_from('<I', data, 12)[0]
            doc = json.loads(data[20:20+length])
            assert all(m['alphaMode'] == 'BLEND' and m['doubleSided']
                       and m['pbrMetallicRoughness']['baseColorFactor'] == RGBA for m in doc['materials'])
        print('PASS voxel-center geometry, anisotropic spacing, orientation, material and unchanged CT input')
        # Independent synthetic patient + a detached posterior support sheet.
        axes = [(np.arange(n)+.5)*2/n-1 for n in (40, 40, 24)]
        xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        phantom = np.zeros((40, 40, 24), np.float32)
        phantom[(xyz**2).sum(-1) < .45**2] = 1.
        phantom[:, 36:37, :] = .7
        snapshot = phantom.copy()
        path = root/'without_support.glb'
        export_mesh(phantom, path, [0.]*3, [2.]*3, scene='pancreas', level=.5)
        mesh = trimesh.load(path, force='mesh', process=False)
        # World Z=-native Y: the sheet at nativeY~.825 must disappear.
        assert mesh.bounds[0, 2] > -.6, mesh.bounds
        assert mesh.bounds[1, 2] > .3
        assert np.array_equal(phantom, snapshot)
        print('PASS posterior support removed while patient and original CT array are retained')
        for bad in [np.zeros((4, 4, 4)), np.full((4, 4, 4), np.nan), np.ones((1, 4, 4))]:
            try:
                export_mesh(bad, root/'bad.glb', [0.]*3, [2.]*3)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid or empty volume accepted')
        assert infer_scene('/tmp/copied.ply', {'source_path': 'data/synthetic/head_50.pickle'}) == 'head'
        assert infer_scene('/tmp/pretrained/real/walnut/point_cloud/iteration_30000/point_cloud.ply', {}) == 'walnut'
        assert infer_scene('/tmp/unknown.ply', {}) is None
        print('PASS empty/nonfinite grids rejected and directory/PLY scene selection')


if __name__ == '__main__':
    main()
