"""CPU Gym export tensors with CUDA policy mirrors; GPU PhysX is unchanged.

Copies occur only at existing native acquire/refresh/set calls. No inference,
reward, action, physics timestep, body geometry or observation transformation.
Both descriptors and temporary setters have strong lifetime ownership.
"""
from isaacgym import gymtorch
import torch


class CpuExportGymProxy:
    def __init__(self, native, device):
        self.native = native
        self.device = device
        self.buffers = {}
        self.setter_holders = {}
        self.counters = {}

    def __getattr__(self, name):
        native = getattr(self.native, name)
        if name.startswith('acquire_') and name.endswith('_tensor'):
            def acquire(*args, **kwargs):
                field = name[len('acquire_'):-len('_tensor')]
                if field not in self.buffers:
                    descriptor = native(*args, **kwargs)
                    cpu = gymtorch.wrap_tensor(descriptor)
                    assert cpu.device.type == 'cpu', 'Expected CPU tensor export'
                    mirror = cpu.to(self.device).clone()
                    self.buffers[field] = (descriptor, cpu, mirror)
                return gymtorch.unwrap_tensor(self.buffers[field][2])
            return acquire
        if name.startswith('refresh_') and name.endswith('_tensor'):
            def refresh(*args, **kwargs):
                result = native(*args, **kwargs)
                field = name[len('refresh_'):-len('_tensor')]
                if field in self.buffers:
                    _, cpu, mirror = self.buffers[field]
                    mirror.copy_(cpu, non_blocking=False)
                    torch.cuda.synchronize()
                self.counters[name] = self.counters.get(name, 0) + 1
                return result
            return refresh
        if name.startswith('set_') and '_tensor' in name:
            def setter(*args, **kwargs):
                assert not kwargs, 'Re-review keyword tensor setters'
                converted = [args[0]]
                holders = []
                for value in args[1:]:
                    if isinstance(value, (int, float, bool)):
                        converted.append(value)
                    else:
                        tensor = gymtorch.wrap_tensor(value)
                        cpu = tensor.detach().cpu().contiguous().clone()
                        holders.append(cpu)
                        converted.append(gymtorch.unwrap_tensor(cpu))
                self.setter_holders[name] = holders
                self.counters[name] = self.counters.get(name, 0) + 1
                return native(*converted)
            return setter
        return native

    def diagnostic(self):
        return dict(bridge_buffers={name: dict(native_device=str(cpu.device),
                    policy_device=str(mirror.device),shape=list(cpu.shape))
                    for name,(_,cpu,mirror) in self.buffers.items()},
                    bridge_calls=dict(self.counters))
