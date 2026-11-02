"""Manual-loop adapter for pinned original TMR model/loss method bodies.

No model, tensors, robot data or CUDA is constructed on import. Lightning is
replaced only as a lifecycle base; original TMR/TEMOS method AST is unchanged.
The original configure_optimizers creates AdamW, not Adam.
"""
import ast
import hashlib
import importlib.util
import json
import pathlib
import sys
import types


def _sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _classes(tree):
    return {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
            for n in tree.body if isinstance(n, ast.ClassDef)}


def original_tmr_classes(root, bindings):
    import torch
    root = pathlib.Path(root)
    assert root.resolve() == root
    names = ['actor', 'losses', 'metrics', 'temos', 'tmr']
    assert set(bindings) == set(names)
    namespace = '_textop_owned_tmr_' + hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()[:16]
    assert namespace not in sys.modules, 'Construct once per process; do not hide duplicate model initialization'
    package = types.ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    audits = {}
    for name in names:
        path = root / (name + '.py')
        assert path.resolve() == path and _sha(path) == bindings[name]
        tree = ast.parse(path.read_bytes(), filename=str(path))
        original_classes = _classes(tree)
        changed = []
        if name == 'temos':
            newbody = []
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == 'pytorch_lightning':
                    assert node.level == 0 and [(x.name, x.asname) for x in node.names] == [('LightningModule', None)]
                    changed.append('Lightning lifecycle import replaced by local torch.nn.Module base')
                    continue
                if isinstance(node, ast.ImportFrom) and node.module == 'src.model.losses':
                    node.module = 'losses'
                    node.level = 1
                    changed.append('Loss import resolved within signed private package')
                newbody.append(node)
            tree.body = newbody
            assert len(changed) == 2
        assert _classes(tree) == original_classes, 'Never rewrite original class or numerical method bodies'
        ast.fix_missing_locations(tree)
        module = types.ModuleType(namespace + '.' + name)
        module.__file__ = str(path)
        module.__package__ = namespace
        if name == 'temos':
            module.LightningModule = torch.nn.Module
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        setattr(package, name, module)
        audits[name] = dict(original_source_sha256=bindings[name], original_class_AST_sha256=original_classes,
                            original_class_AST_unchanged=True, lifecycle_import_adaptations=changed)
    return package.actor, package.tmr.TMR, audits


class Sampler:
    """Complete deterministic sample permutation and cursor, including final batch."""
    def __init__(self, count, seed):
        import torch
        assert type(count) is int and count > 0
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.count = count
        self.epoch = self.cursor = self.steps = 0
        self.order = torch.randperm(count, generator=self.generator)

    def take(self, batch_size):
        import torch
        assert type(batch_size) is int and batch_size > 1
        if self.cursor == self.count:
            self.epoch += 1
            self.cursor = 0
            self.order = torch.randperm(self.count, generator=self.generator)
        end = min(self.cursor + batch_size, self.count)
        result = self.order[self.cursor:end].tolist()
        assert len(result) > 1, 'Explicit handling required if corpus leaves a singleton contrastive batch'
        self.cursor = end
        self.steps += 1
        return result

    def state(self):
        return dict(count=self.count, epoch=self.epoch, cursor=self.cursor, steps=self.steps,
            order=self.order.clone(), generator=self.generator.get_state().clone())

    def restore(self, state):
        import torch
        assert state['count'] == self.count and 0 <= state['cursor'] <= self.count
        assert sorted(state['order'].tolist()) == list(range(self.count))
        assert state['order'].dtype == torch.int64
        self.epoch, self.cursor, self.steps = state['epoch'], state['cursor'], state['steps']
        self.order = state['order'].clone()
        self.generator.set_state(state['generator'].clone())


def make_model(root, bindings, device):
    import torch
    actor, TMR, audit = original_tmr_classes(root, bindings)
    kwargs = dict(latent_dim=256, ff_size=1024, num_layers=6, num_heads=4, dropout=.1, activation='gelu')
    model = TMR(motion_encoder=actor.ACTORStyleEncoder(nfeats=69, vae=True, **kwargs),
        text_encoder=actor.ACTORStyleEncoder(nfeats=768, vae=True, **kwargs),
        motion_decoder=actor.ACTORStyleDecoder(nfeats=69, **kwargs), vae=True, fact=1., sample_mean=False,
        lmd=dict(recons=1., latent=1e-5, kl=1e-5, contrastive=.1), lr=1e-4,
        temperature=.1, threshold_selfsim=.8, threshold_selfsim_metrics=.95).to(device)
    optimizer = model.configure_optimizers()['optimizer']
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults['lr'] == 1e-4 and optimizer.defaults['weight_decay'] == .01
    assert sum(p.numel() for p in model.parameters()) == 16031301
    return model, optimizer, audit


def rng_state():
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def restore_rng(state):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        assert torch.cuda.is_initialized() and len(state['cuda']) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def step(model, optimizer, batch, training=True):
    import torch
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        losses = model.compute_loss(batch)
        assert set(losses) == {'recons', 'latent', 'kl', 'contrastive', 'loss'}
        assert all(x.numel() == 1 and torch.isfinite(x).all() for x in losses.values())
        if training:
            losses['loss'].backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(bool((g != 0).any()) for g in grads)
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
    return {name: float(value.detach()) for name, value in losses.items()}
