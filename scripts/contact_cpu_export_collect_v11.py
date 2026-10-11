"""Paired diagnostic of CPU tensor export with frozen CUDA models/observations."""
from pathlib import Path

previous=Path(__file__).with_name('contact_sensor_collect_v10.py')
adapter=previous.read_text()
ending="exec(compile(source,str(Path(__file__)),'exec'),globals())"
assert adapter.rstrip().endswith(ending)
exec(compile(adapter.rsplit(ending,1)[0],str(previous),'exec'),globals())

patches=[
    ("from closd.env.tasks.humanoid import Humanoid\n",
     "from closd.env.tasks.humanoid import Humanoid\n"
     "from closd_cpu_export_bridge import CpuExportGymProxy\n"
     "original_create_sim=Humanoid.create_sim\n"
     "def cpu_export_create_sim(self):\n"
     "    assert self.device.startswith('cuda')\n"
     "    assert self.sim_params.physx.use_gpu\n"
     "    self.sim_params.use_gpu_pipeline=False\n"
     "    self.export_bridge=CpuExportGymProxy(self.gym,self.device)\n"
     "    self.gym=self.export_bridge\n"
     "    return original_create_sim(self)\n"
     "Humanoid.create_sim=cpu_export_create_sim\n"),
    ("    metadata['sim_params']=dict(dt=float(sim_params.dt),substeps=int(sim_params.substeps),use_gpu_pipeline=bool(sim_params.use_gpu_pipeline),contact_collection=int(sim_params.physx.contact_collection),physx_use_gpu=bool(sim_params.physx.use_gpu))\n",
     "    metadata['sim_params']=dict(dt=float(sim_params.dt),substeps=int(sim_params.substeps),use_gpu_pipeline=bool(sim_params.use_gpu_pipeline),contact_collection=int(sim_params.physx.contact_collection),physx_use_gpu=bool(sim_params.physx.use_gpu))\n"
     "    assert not sim_params.use_gpu_pipeline and sim_params.physx.use_gpu\n"
     "    metadata['tensor_export_route']='CPU native exports, explicit unchanged CUDA mirrors; GPU PhysX and frozen CUDA models'\n"),
    ("    metadata=store['metadata']; metadata['elapsed_seconds']=time.time()-started\n",
     "    metadata=store['metadata']; metadata['elapsed_seconds']=time.time()-started\n"
     "    metadata.update(task.export_bridge.diagnostic())\n"),
]
for old,new in patches:
    assert source.count(old)==1,'Parent adapter changed; review bridge'
    source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'),globals())
