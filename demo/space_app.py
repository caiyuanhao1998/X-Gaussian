"""The same interactive UI, using short ZeroGPU calls and cached native CT slices."""
import json
import os
import sys
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Install ZeroGPU's CUDA emulation before importing PyTorch or the inference engine.
print('Initializing X-Gaussian ZeroGPU runtime', flush=True)
import spaces
import gradio as gr
import uvicorn
from assets import CACHE
print('Loading example checkpoint metadata', flush=True)
from worker import Engine, CATALOG, load_model
from app import build_app

engine = Engine(0, shared_gpu=True)
# Keep the eight small Gaussian models in the parent CUDA emulation state.
# ZeroGPU makes them available in each request without training or reloading data.
for scene, item in CATALOG.items():
    engine.models[scene] = load_model(item['checkpoint'])
    print(f'Loaded {scene} Gaussian model', flush=True)


def gpu_duration(request):
    return 60 if request['operation'] == 'mesh' else 30


@spaces.GPU(duration=gpu_duration)
def gpu_request(request):
    result = engine.handle(request)
    if request['operation'] == 'render':
        # One short allocation prepares a scene; moving its slice slider is CPU only.
        engine.volume_file(request['scene'])
        engine.handle(dict(operation='points', scene=request['scene']))
    return result


class SpaceWorker:
    info = {'device': 'ZeroGPU'}

    def available(self):
        return True

    def call(self, operation, scene=None, **params):
        request = dict(operation=operation, scene=scene, **params)
        try:
            if operation in ('slice', 'points'):
                item = engine.catalog[scene]
                source = Path(item['checkpoint']).stat()
                key = f'{scene}-{source.st_size}-{source.st_mtime_ns}'
                if operation == 'slice':
                    volume = CACHE / 'volumes' / (key + '.npy')
                    if not volume.is_file():
                        gpu_request(dict(operation='volume', scene=scene))
                    return engine.slice_cached(scene, request, volume)
                point = CACHE / 'points' / (key + '.bin')
                if point.is_file() and point.with_suffix('.json').is_file():
                    result = json.loads(point.with_suffix('.json').read_text())
                    result['file'] = str(point)
                    return result
            result = gpu_request(request)
            if operation == 'register':
                # ZeroGPU requests run in child processes; retain only public metadata.
                engine.catalog[result['scene']] = result['item']
            return result
        except Exception as error:
            raise gr.Error(str(error)) from error


if __name__ == '__main__':
    uvicorn.run(build_app(SpaceWorker()), host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
