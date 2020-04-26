import argparse, importlib

import torch
import runway

from vaesampler import BPEmbVaeSampler

model_path = "wake_aggressive1_kls0.10_warm10_0_0_783435.pt"

@runway.setup
def setup():
    use_gpu = torch.cuda.is_available()
    config_file = "config.config_wake"
    params = argparse.Namespace(**importlib.import_module(config_file).params)
    model = BPEmbVaeSampler(lang=params.bpemb['lang'],
            vs=params.bpemb['vs'], dim=params.bpemb['dim'],
            decode_from=model_path, params=params, cuda=use_gpu)
    return model

@runway.command('generate',
        inputs={
            'z': runway.vector(length=64),
            'temperature': runway.number(default=0.5, min=0.05, max=2.0,
                step=0.05)
        },
        outputs={'out': runway.text})
def generate(model, inputs):
    z = torch.from_numpy(inputs['z']).float().unsqueeze(0).to(model.device)
    temperature = inputs['temperature']
    with torch.no_grad():
        return model.sample(z, temperature)[0]

@runway.command('reconstruct',
        inputs={
            'in': runway.text,
            'temperature': runway.number(default=0.5, min=0.05, max=2.0,
                step=0.05)
        },
        outputs={'out': runway.text})
def reconstruct(model, inputs):
    with torch.no_grad():
        return model.sample(
                model.z([inputs['in']]), inputs['temperature'])[0]

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)

