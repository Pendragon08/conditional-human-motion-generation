"""Independent inference-only runtime for genuine accepted100k weights.

Importing this module does not import Torch or load data/models. Construction
is reserved for a separately authorized, common-lock-supervised GPU job.
It never constructs a manager/optimizer or uses future-posterior VAE encoding.
"""
import hashlib
import pathlib
from ldm100k_generation_protocol import pair_seed, MODES


class Runtime:
    def __init__(self, accepted, source_index):
        import sys
        import numpy as np
        import torch
        from ldm100k_verified_input import ROOT, read, sha, owned
        from generator_training_state import module_sha
        self.torch, self.np = torch, np
        self.sha, self.owned = sha, owned
        self.source = accepted['export']['stopped']['source_spec']
        self.index = {x['canonical_source']: x for x in source_index}
        assert len(self.index) == len(source_index)
        base = read(ROOT / 'configs/paper69-env-spec.json')
        sys.path.insert(0, str(ROOT / 'upstream/TextOpRobotMDAR'))
        # Model modules use the pinned architecture without installing or
        # mutating any dataset/training-manager adapter in this process.
        from robotmdar.model.mld_vae import AutoMldVae
        from robotmdar.model.mld_denoiser import DenoiserTransformer
        from robotmdar.diffusion.gaussian_diffusion import (
            GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule)
        import robotmdar.diffusion.gaussian_diffusion as diffusion_module
        extract = diffusion_module._extract_into_tensor

        def extract_int64(array, timesteps, shape):
            return extract(array, timesteps.long(), shape)

        # Same process-local int64 compatibility rule as the frozen training
        # adapter; no released source or running process is changed.
        diffusion_module._extract_into_tensor = extract_int64
        import paper69_runtime as codec
        self.codec = codec
        self.device = torch.device('cuda:0')
        assert torch.cuda.device_count() == 1
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.65, device=0)
        assert torch.cuda.mem_get_info()[0] >= 20 * 1024**3
        self.vae = AutoMldVae(nfeats=69, latent_dim=[1, 128], h_dim=512,
                              ff_size=1024, num_layers=9, num_heads=4).cpu().eval()
        self.model = DenoiserTransformer(h_dim=512, ff_size=1024, num_layers=8,
            num_heads=4, clip_dim=512, history_shape=(2, 69), noise_shape=(1, 128),
            cond_mask_prob=.1).cpu().eval()
        ld = accepted['export']['inference']
        vae = self.source['vae_inference']
        assert sha(ld['path']) == ld['sha256'] and sha(vae['path']) == vae['sha256']
        lp = torch.load(ld['path'], map_location='cpu')
        vp = torch.load(vae['path'], map_location='cpu')
        assert lp['schema'] == 'textop.ldm100k_prefix_inference.v1'
        assert vp['schema'] == 'textop.filtered15000_vae_inference.v1'
        assert lp['completed_primitive_attempts'] == vp['completed_primitive_attempts'] == 100000
        assert lp['source_full_state_sha256'] == accepted['export']['stopped']['checkpoint']['sha256']
        assert vp['source_full_state_sha256'] == accepted['export']['stopped']['policy']['vae_checkpoint_sha256']
        assert lp['architecture'] == self.source['ldm']
        assert lp['source_numeric_binding_sha256'] == accepted['export']['stopped']['policy']['original_runtime_binding_sha256']
        assert lp['vae_inference_sha256'] == vae['sha256']
        assert lp['corpus_receipt_sha256'] == vp['corpus_receipt_sha256'] == self.source['corpus_receipt_sha256']
        assert vp['data_acceptance_sha256'] == self.source['data_acceptance_sha256']
        self.model.load_state_dict(lp['denoiser'], strict=True)
        self.vae.load_state_dict(vp['vae'], strict=True)
        assert module_sha(self.model) == lp['trained_denoiser_sha256']
        assert module_sha(self.vae) == vp['trained_model_sha256']
        for model in [self.model, self.vae]:
            for value in model.state_dict().values():
                assert torch.isfinite(value).all()
            for param in model.parameters():
                param.requires_grad_(False)
            model.to(self.device).eval()
        self.model_hash, self.vae_hash = module_sha(self.model), module_sha(self.vae)
        self.module_sha = module_sha
        self.diffusion = GaussianDiffusion(betas=get_named_beta_schedule('cosine', 5),
            model_mean_type=ModelMeanType.START_X, model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE, rescale_timesteps=False)
        view = pathlib.Path(self.source['corpus_dir'])
        self.view = view
        self.view_root = pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
        owned(view, self.view_root)
        assert sha(view / 'view_receipt.json') == self.source['corpus_receipt_sha256']
        meta = read(view / 'view_receipt.json')
        self.metadata = meta
        for name in ['meanstd.pkl', 'val_text_embed.pkl']:
            path = owned(view / name, self.view_root)
            assert path.stat().st_size == meta['artifacts'][name]['bytes']
            assert sha(path) == meta['artifacts'][name]['sha256']
        self.mean, self.std = torch.load(view / 'meanstd.pkl', map_location='cpu')
        assert self.mean.shape == self.std.shape == (69,)
        assert self.mean.dtype == self.std.dtype == torch.float32
        assert torch.isfinite(self.mean).all() and torch.isfinite(self.std).all() and (self.std > 0).all()
        self.embeddings = torch.load(view / 'val_text_embed.pkl', map_location='cpu')
        self.cached_source = None
        self.reference = self.qpos = None
        self.inference_paths = [ld, vae]
        self.verified_source_hashes = {}

    def _source(self, key):
        t, np = self.torch, self.np
        if self.cached_source == key:
            return
        row = self.index[key]
        assert row['split'] == 'val' and row['mirrored'] is False
        path = pathlib.Path(row['output_file'])
        roots = [p for p in [pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024'),
                             self.view_root] if path == p or p in path.parents]
        assert len(roots) == 1
        self.owned(path, roots[0])
        assert self.sha(path) == row['output_sha256']
        with np.load(path, allow_pickle=False) as motion:
            q = np.array(motion['qpos_29'], dtype=np.float32)
            c = np.array(motion['contact'], dtype=np.float32)
        assert q.shape == (row['frames50'], 36) and c.shape == (len(q), 2)
        assert np.isfinite(q).all() and np.isfinite(c).all()
        assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-3)
        features, _ = self.codec.encode69(dict(root_trans_offset=t.from_numpy(q[:, :3]),
            root_rot=t.from_numpy(q[:, [4, 5, 6, 3]]), dof=t.from_numpy(q[:, 7:]),
            contact_mask=t.from_numpy(c)))
        assert features.shape == (len(q) - 1, 69) and t.isfinite(features).all()
        normalized = (features - self.mean) / self.std
        assert t.isfinite(normalized).all()
        self.reference, self.qpos = normalized, q
        self.cached_source = key
        self.verified_source_hashes[str(path)] = row['output_sha256']

    def paired(self, segment, seed, on_mode=None):
        t, np = self.torch, self.np
        self._source(segment['canonical_source'])
        start, end = segment['start_feature'], segment['end_feature_exclusive']
        assert self.index[segment['canonical_source']]['output_sha256'] == segment['reference_output_sha256']
        initial = self.reference[start - 2:start].unsqueeze(0).to(self.device)
        assert initial.shape == (1, 2, 69)
        target = self.reference[start:end].to(self.device)
        assert segment['text'] in self.embeddings, 'Never substitute missing text with zero embeddings'
        text = self.embeddings[segment['text']].float().reshape(1, 512).to(self.device)
        assert t.isfinite(text).all()
        pairseed = pair_seed(segment['segment_id'], seed)
        output, traces = {}, {}
        for mode in MODES:
            t.manual_seed(pairseed)
            t.cuda.manual_seed_all(pairseed)
            history = initial.clone()
            generated, trace = [], []
            with t.no_grad():
                for primitive in range(segment['primitive_calls']):
                    before = hashlib.sha256(t.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
                    initial_noise = t.randn((1, 1, 128), device=self.device)
                    noise_sha = hashlib.sha256(initial_noise.cpu().numpy().tobytes()).hexdigest()
                    forwards = [0]

                    def guided(x, time, y):
                        if mode == 'unconditional':
                            forwards[0] += 1
                            return self.model(x, time.long(), y=dict(y, uncond=True))
                        forwards[0] += 2
                        un = self.model(x, time.long(), y=dict(y, uncond=True))
                        yes = self.model(x, time.long(), y=dict(y, uncond=False))
                        return un + 5 * (yes - un)

                    sample = self.diffusion.p_sample_loop(guided, (1, 1, 128), noise=initial_noise,
                        device=self.device, clip_denoised=False, progress=False,
                        model_kwargs={'y': dict(text_embedding=text, history_motion_normalized=history)})
                    future = self.vae.decode(sample.permute(1, 0, 2), history, nfuture=8)
                    assert future.shape == (1, 8, 69) and t.isfinite(future).all()
                    assert forwards[0] == (5 if mode == 'unconditional' else 10)
                    generated.append(future)
                    history = future[:, -2:]
                    after = hashlib.sha256(t.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
                    trace.append(dict(primitive=primitive, rng_before_sha256=before,
                        initial_noise_sha256=noise_sha, rng_after_sha256=after))
                full = t.cat(generated, dim=1)
                future = full[0, :end - start]
                assert future.shape == target.shape
                features = t.cat([initial, full], dim=1) * self.std.to(self.device) + self.mean.to(self.device)
                pose = self.qpos[start - 2]
                absolute = dict(root_trans_offset=t.as_tensor(pose[:3], device=self.device),
                                root_rot=t.as_tensor(pose[[4, 5, 6, 3]], device=self.device))
                motion = self.codec.decode69(features, absolute)
                q, contact = self.codec.to_qpos69(motion)
                q, contact = q[0, 2:2 + len(target)], contact[0, 2:2 + len(target)]
                assert q.shape == (len(target), 36) and t.isfinite(q).all() and t.isfinite(contact).all()
                output[mode] = dict(generated_normalized_future=full[0].cpu().numpy().astype(np.float32),
                    reference_normalized_future=target.cpu().numpy().astype(np.float32),
                    generated_qpos=q.cpu().numpy().astype(np.float32),
                    generated_contact=contact.cpu().numpy().astype(np.float32),
                    initial_reference_history=initial[0].cpu().numpy().astype(np.float32),
                    normalized_huber=float(t.nn.functional.huber_loss(future, target)))
            traces[mode] = trace
            if on_mode is not None:
                on_mode(mode, output[mode], trace)
        assert traces[MODES[0]] == traces[MODES[1]], 'Paired native DDPM noise sequence changed'
        return output, dict(pair_seed=pairseed, same_noise_trace_exact=True, noise_trace=traces[MODES[0]],
            feature_history_after_initial_generated_only=True, future_posterior_encoded=False,
            contact_or_dof_clipping_applied=False, diffusion_steps=5, cfg=5,
            variance_type='FIXED_LARGE_native_training', semantic_metrics_computed=False,
            physical_quality_accepted=False)

    def verify_unchanged(self):
        assert self.module_sha(self.model) == self.model_hash and self.module_sha(self.vae) == self.vae_hash
        for item in self.inference_paths:
            assert self.sha(item['path']) == item['sha256']
        for name in ['meanstd.pkl', 'val_text_embed.pkl']:
            assert self.sha(self.view / name) == self.metadata['artifacts'][name]['sha256']
        for path, digest in self.verified_source_hashes.items():
            assert self.sha(path) == digest
