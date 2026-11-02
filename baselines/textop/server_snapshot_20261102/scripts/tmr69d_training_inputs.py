"""Real accepted TRAIN69D and frozen public text features, retained only in RAM.

Construction is reserved for an authorized common-lock GPU job. Import is
stdlib-only and never touches datasets, pretrained weights or CUDA.
"""
import hashlib
import json
import pathlib
import sys


class Inputs:
    def __init__(self, spec, device, progress=None):
        import numpy as np
        import torch
        from ldm100k_verified_input import ROOT, owned, read, sha
        self.torch, self.np, self.device = torch, np, torch.device(device)
        self.owned, self.sha = owned, sha
        self.spec = spec
        self.progress = progress or (lambda **kw: None)
        assert sha(ROOT / 'scripts/paper69_runtime.py') == spec['codec_sha256']
        import paper69_runtime as codec
        self.codec = codec
        records = {}
        for key in ['scope', 'numeric', 'text']:
            item = spec[key + '_receipt']
            path = owned(item['path'])
            assert sha(path) == item['sha256']
            records[key] = read(path)
            assert records[key]['output_writers_closed']
        scope, numeric, text = [records[x] for x in ['scope', 'numeric', 'text']]
        assert scope['status'] == 'TMR69D_TRAINING_METADATA_SCOPE_BUILT'
        assert numeric['status'] == 'TMR69D_NUMERIC_TRAIN_INPUTS_ACCEPTED'
        assert text['status'] == 'IMPORTED_TMR_TEXT_MODELS_CPU_VERIFIED'
        assert numeric['scope_receipt_sha256'] == spec['scope_receipt']['sha256']
        for item in [scope['private_artifact'], numeric['private_artifact']]:
            path = owned(item['path'])
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        data = read(scope['private_artifact']['path'])
        nr = read(numeric['private_artifact']['path'])
        expected = {x['canonical_source']: x for x in nr}
        assert len(expected) == len(data['sources']) == 5361
        self.rows = {x['canonical_source']: x for x in data['sources']}
        assert set(self.rows) == set(expected)
        self.data = data
        self.gradient = [x for x in data['samples'] if x['split'] == 'gradient_train']
        self.internal = [x for x in data['samples'] if x['split'] == 'internal_validation']
        assert len(self.gradient) == 32462 and len(self.internal) == 1606
        norm = self.own_data(data['normalization']['path'])
        assert sha(norm) == data['normalization']['sha256'] == numeric['normalization_sha256']
        mean, std = torch.load(norm, map_location='cpu')
        assert mean.shape == std.shape == (69,) and mean.dtype == std.dtype == torch.float32
        assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
        self.features, self.source_fingerprints = {}, {}
        total_bytes = 0
        for index, row in enumerate(data['sources']):
            key = row['canonical_source']
            assert row['original_split'] == 'train' and row['mirrored'] is False
            path = self.own_data(row['output_file'])
            assert sha(path) == row['output_sha256'] == expected[key]['source_sha256']
            before = path.stat()
            with np.load(path, allow_pickle=False) as m:
                q = np.array(m['qpos_29'], dtype=np.float32)
                c = np.array(m['contact'], dtype=np.float32)
            assert q.shape == (row['frames50'], 36) and c.shape == (row['frames50'], 2)
            assert np.isfinite(q).all() and np.isfinite(c).all()
            assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-3)
            assert ((c >= 0) & (c <= 1)).all()
            with torch.inference_mode():
                features, _ = codec.encode69(dict(root_trans_offset=torch.from_numpy(q[:, :3]),
                    root_rot=torch.from_numpy(q[:, [4, 5, 6, 3]]), dof=torch.from_numpy(q[:, 7:]), contact_mask=torch.from_numpy(c)))
                normalized = ((features - mean) / std).contiguous()
            assert normalized.shape == (row['frames50'] - 1, 69) and torch.isfinite(normalized).all()
            assert hashlib.sha256(normalized.numpy().tobytes()).hexdigest() == expected[key]['normalized_feature_sha256']
            self.features[key] = normalized
            total_bytes += normalized.numel() * normalized.element_size()
            after = path.stat()
            assert self.fingerprint(before) == self.fingerprint(after)
            self.source_fingerprints[str(path)] = self.fingerprint(after)
            if (index + 1) % 100 == 0:
                self.progress(phase='building_real_motion_RAM', sources=index + 1, motion_RAM_bytes=total_bytes)
        assert total_bytes == numeric['full_normalized_feature_RAM_bytes'] == 782643540
        self.motion_RAM_bytes = total_bytes
        dependencies = owned(text['runtime_dependency_dir'])
        sys.path.insert(0, str(dependencies))
        from transformers import AutoModel, AutoTokenizer
        imported_path = owned(spec['import_receipt']['path'])
        assert sha(imported_path) == spec['import_receipt']['sha256'] == text['import_receipt_sha256']
        imported = read(imported_path)
        for item in imported['files']:
            assert sha(owned(item['path'])) == item['sha256']
        self.public_files = imported['files']
        text_items = sorted(data['texts'].items())
        assert len(text_items) == 4043
        self.tokens, self.sentences = {}, {}
        token_bytes = sentence_bytes = 0
        token_h, sent_h = hashlib.sha256(), hashlib.sha256()
        max_tokens = 0
        for modality, directory in enumerate(text['model_directories']):
            path = owned(directory)
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
            model = AutoModel.from_pretrained(str(path), local_files_only=True, use_safetensors=True).eval().requires_grad_(False).to(self.device)
            assert all(not p.requires_grad for p in model.parameters())
            for start in range(0, len(text_items), 32):
                entries = text_items[start:start + 32]
                sentences = [value for _, value in entries]
                encoded = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')
                assert encoded['input_ids'].shape[1] <= min(tokenizer.model_max_length, model.config.max_position_embeddings)
                lengths = encoded['attention_mask'].sum(1).tolist()
                max_tokens = max(max_tokens, max(lengths))
                with torch.inference_mode():
                    result = model(**encoded.to(self.device)).last_hidden_state
                    assert result.shape[-1] == 768 and torch.isfinite(result).all()
                    if modality == 0:
                        for i, (key, _) in enumerate(entries):
                            value = result[i, :lengths[i]].float().cpu().contiguous()
                            self.tokens[key] = value
                            token_bytes += value.numel() * 4
                            token_h.update(key.encode())
                            token_h.update(value.numpy().tobytes())
                    else:
                        mask = encoded['attention_mask'].unsqueeze(-1).expand_as(result).float()
                        pooled = torch.sum(result * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).float().cpu().contiguous()
                        assert torch.allclose(torch.linalg.vector_norm(pooled, dim=-1), torch.ones(len(entries)), atol=1e-6)
                        for i, (key, _) in enumerate(entries):
                            value = pooled[i].clone()
                            self.sentences[key] = value
                            sentence_bytes += value.numel() * 4
                            sent_h.update(key.encode())
                            sent_h.update(value.numpy().tobytes())
                self.progress(phase='building_real_frozen_text_RAM', modality=modality, texts=min(start + 32, len(text_items)),
                    token_RAM_bytes=token_bytes, sentence_RAM_bytes=sentence_bytes)
            del model, tokenizer, result, encoded
            torch.cuda.empty_cache()
        assert set(self.tokens) == set(self.sentences) == set(data['texts'])
        assert len(self.tokens) == 4043 and sentence_bytes == 4043 * 768 * 4
        self.signature = dict(scope_receipt_sha256=spec['scope_receipt']['sha256'], numeric_receipt_sha256=spec['numeric_receipt']['sha256'],
            text_receipt_sha256=spec['text_receipt']['sha256'], original_train_sources=5361, gradient_samples=32462,
            internal_samples=1606, unique_texts=4043, motion_RAM_bytes=total_bytes, token_RAM_bytes=token_bytes,
            sentence_RAM_bytes=sentence_bytes, token_cache_sha256=token_h.hexdigest(), sentence_cache_sha256=sent_h.hexdigest(),
            maximum_actual_text_tokens=max_tokens, token_truncation_performed=False, mirror_augmentation=False,
            actual_original_validation_sources_read=0, feature_transform_sha256=numeric['codec_sha256'], normalization_sha256=sha(norm))
        self.signature_sha256 = hashlib.sha256(json.dumps(self.signature, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def own_data(self, p):
        from ldm100k_verified_input import ROOT
        path = pathlib.Path(p)
        shm = pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
        roots = [x for x in [ROOT, shm] if path == x or x in path.parents]
        assert len(roots) == 1
        return self.owned(path, roots[0])

    @staticmethod
    def fingerprint(st):
        return dict(device=st.st_dev, inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns, uid=st.st_uid)

    def batch(self, indices, split='gradient_train'):
        torch = self.torch
        samples = self.gradient if split == 'gradient_train' else self.internal
        assert split in ['gradient_train', 'internal_validation'] and len(indices) > 1
        rows = [samples[i] for i in indices]
        motions = [self.features[x['canonical_source']][x['start_feature']:x['end_feature_exclusive']] for x in rows]
        texts = [self.tokens[x['text_id']] for x in rows]
        ml, tl = max(len(x) for x in motions), max(len(x) for x in texts)
        motion = torch.zeros(len(rows), ml, 69)
        text = torch.zeros(len(rows), tl, 768)
        mm = torch.zeros(len(rows), ml, dtype=torch.bool)
        tm = torch.zeros(len(rows), tl, dtype=torch.bool)
        for i, (m, t) in enumerate(zip(motions, texts)):
            assert 1 <= len(m) <= 200 and t.shape[1] == 768
            motion[i, :len(m)] = m
            text[i, :len(t)] = t
            mm[i, :len(m)] = True
            tm[i, :len(t)] = True
        sent = torch.stack([self.sentences[x['text_id']] for x in rows])
        result = dict(motion_x_dict=dict(x=motion.to(self.device), mask=mm.to(self.device)),
            text_x_dict=dict(x=text.to(self.device), mask=tm.to(self.device)), sent_emb=sent.to(self.device))
        assert all(torch.isfinite(x).all() for x in [motion, text, sent])
        h = hashlib.sha256()
        for row in rows:
            h.update(row['sample_id'].encode())
        for tensor in [motion, text, mm, tm, sent]:
            h.update(tensor.numpy().tobytes())
        return result, h.hexdigest()

    def verify_unchanged(self):
        for row in self.data['sources']:
            path = self.own_data(row['output_file'])
            assert self.fingerprint(path.stat()) == self.source_fingerprints[str(path)]
            assert self.sha(path) == row['output_sha256']
        norm = self.own_data(self.data['normalization']['path'])
        assert self.sha(norm) == self.data['normalization']['sha256']
        for item in self.public_files:
            path = self.owned(item['path'])
            assert path.stat().st_size == item['bytes'] and self.sha(path) == item['sha256']
            assert self.fingerprint(path.stat()) == item['fingerprint']
