"""Apply the public DreamControl HumanML3D networks and metrics to existing G1 clips."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from evaluate_generated_reference_geometry_20261105 import ROOT, UPSTREAM, MANIFEST, MANIFEST_SHA, METRICS_SHA, TASKS, sha, checked_npz

GROUND_TRUTH = ROOT / 'grab_omni_29dof_weak_baseline_v1_20261103/manifest.json'
WEIGHTS_SHA = '569bbd5526cb4b5fdda48a08abf7dd0b5a1e9a39ea0be381730771ac18063981'
WRAPPER_SHA = '7a3578bae99c75c11ab741d12d95b3048f3d7f768b2508a7a54d2f5f5288c330'
SEED = 20261105
MAX_FRAMES = 196
BATCH = 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--tokens', type=Path)
    parser.add_argument('--resume-motion-from', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    assert sha(MANIFEST) == MANIFEST_SHA
    assert sha(UPSTREAM / 'data_loaders/humanml/utils/metrics.py') == METRICS_SHA
    assert sha(UPSTREAM / 'data_loaders/humanml/networks/evaluator_wrapper.py') == WRAPPER_SHA
    sys.path.insert(0, str(UPSTREAM))
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.networks.modules import MovementConvEncoder, MotionEncoderBiGRUCo, TextEncoderBiGRUCo
    from data_loaders.humanml.utils.metrics import (
        calculate_activation_statistics, calculate_diversity, calculate_frechet_distance,
        euclidean_distance_matrix,
    )
    from data_loaders.humanml.utils.word_vectorizer import POS_enumerator
    from feature263_signfix import encode_signfixed

    assets = args.assets.resolve()
    model_path = assets / 'model.safetensors'
    assert sha(model_path) == WEIGHTS_SHA
    asset_receipt = json.loads((assets / 'source_assets.json').read_text())
    for entry in asset_receipt:
        p = assets / entry['member']
        assert p.stat().st_size == entry['size'] and sha(p) == entry['sha256']
    mean, std = [np.load(assets / 'stats' / name, allow_pickle=False) for name in ('mean.npy', 'std.npy')]
    assert mean.shape == std.shape == (263,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    assert sha(assets / 'stats/mean.npy') == sha(UPSTREAM / 'dataset/t2m_mean.npy')
    assert sha(assets / 'stats/std.npy') == sha(UPSTREAM / 'dataset/t2m_std.npy')
    weights = load_file(model_path, device='cpu')
    evaluator = EvaluatorMDMWrapper.__new__(EvaluatorMDMWrapper)
    evaluator.device = 'cpu'
    evaluator.opt = {'unit_length': 4}
    evaluator.movement_encoder = MovementConvEncoder(259, 512, 512)
    evaluator.motion_encoder = MotionEncoderBiGRUCo(input_size=512, hidden_size=1024, output_size=512, device='cpu')
    evaluator.text_encoder = TextEncoderBiGRUCo(word_size=300, pos_size=len(POS_enumerator),
                                                hidden_size=512, output_size=512, device='cpu')
    used = set()
    for name in ('movement_encoder', 'motion_encoder', 'text_encoder'):
        prefix = name + '.'
        module = getattr(evaluator, name)
        subset = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        module.load_state_dict(subset, strict=True)
        module.eval().requires_grad_(False)
        used.update(prefix + k for k in subset)
    assert used == weights.keys() and len(used) == 42
    del weights

    def embed(generator):
        result = []
        batch = []
        truncated = 0
        def flush():
            samples = np.zeros((len(batch), MAX_FRAMES, 263), np.float32)
            lengths = np.array([len(x) for x in batch], dtype=np.int64)
            assert lengths.min() >= 4
            for i, feat in enumerate(batch):
                samples[i, :len(feat)] = (feat - mean) / std
            align = np.argsort(lengths)[::-1].copy()
            values = evaluator.get_motion_embeddings(torch.from_numpy(samples), torch.from_numpy(lengths))
            return values.numpy()[np.argsort(align)]
        for feat in generator:
            assert feat.ndim == 2 and feat.shape[1] == 263 and np.isfinite(feat).all()
            truncated += len(feat) > MAX_FRAMES
            batch.append(feat[:MAX_FRAMES])
            if len(batch) == BATCH:
                result.append(flush())
                batch.clear()
        if batch:
            result.append(flush())
        result = np.concatenate(result)
        assert result.shape[1] == 512 and np.isfinite(result).all()
        return result, truncated

    motions = json.loads(MANIFEST.read_text())['records']
    gt = json.loads(GROUND_TRUTH.read_text())['records']
    test = [row for row in gt if row['split'] == 'test']
    assert len(test) == 586
    for key in ('group_id', 'source_id', 'feature_sha256'):
        assert not ({r[key] for r in test} & {r[key] for r in gt if r['split'] != 'test'})
    stages = ('raw_diffusion', 'final_reference')
    if args.resume_motion_from:
        assert args.tokens is not None
        previous = args.resume_motion_from
        report = json.loads((previous / 'report.json').read_text())
        assert report['status'] == 'motion_embeddings_completed_retrieval_pending'
        assert report['manifest_sha256'] == MANIFEST_SHA and report['ground_truth_manifest_sha256'] == sha(GROUND_TRUTH)
        assert report['assets'] == asset_receipt
        with checked_npz(previous / 'motion_embeddings.npz', report['motion_embeddings_sha256']) as data:
            embeddings = {stage: {task: data[stage + '__' + task] for task in TASKS} for stage in stages}
        add_retrieval(args.tokens, evaluator, assets, motions, test, embeddings, report, euclidean_distance_matrix)
        report['status'] = 'generated_reference_embedding_metrics_completed'
        report['motion_result_source'] = str(previous.resolve())
        with (args.output / 'report.json').open('x') as file:
            json.dump(report, file, indent=2, allow_nan=False)
        return
    def ground_features():
        for row in test:
            with checked_npz(ROOT / row['feature_path'], row['feature_sha256']) as item:
                yield item['features']

    report = {'status': 'motion_embeddings_running', 'manifest_sha256': MANIFEST_SHA,
              'ground_truth_manifest_sha256': sha(GROUND_TRUTH), 'assets': asset_receipt,
              'wrapper_sha256': WRAPPER_SHA, 'metrics_sha256': METRICS_SHA, 'seed': SEED,
              'ground_truth': {'split': 'test', 'clips': len(test),
                               'source_groups': len({x['source_id'] for x in test}),
                               'dataset_sources': dict(Counter(x['dataset_source'] for x in test)),
                               'scope': 'shared, source-group-held-out GRAB+Omni robot-space reference distribution; task scores use the same pool'},
              'encoding': {'raw_diffusion': 'original denormalized 263D generator output',
                           'final_reference': 'stored 20Hz G1 FK 22 points to existing sign-continuous HumanML 263D encoder',
                           'ground_truth': 'archived 20Hz G1 263D features',
                           'normalization': 'published HumanML3D matching-network mean and std; max 196 frames'},
              'diversity': 'public calculate_diversity with 32 sampled pairs, mean of 20 deterministic replicates',
              'fid': 'public activation statistics and Frechet distance using original unnormalized embeddings',
              'tasks': {}, 'retrieval': None}
    embeddings = {}
    clipped = {}

    def write_report():
        tmp = args.output / 'report.tmp'
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        tmp.replace(args.output / 'report.json')

    def diversity(values):
        samples = []
        for i in range(20):
            np.random.seed(SEED + i)
            samples.append(calculate_diversity(values, 32))
        return {'mean': float(np.mean(samples)), 'sd_across_pair_draws': float(np.std(samples))}

    reference, gt_truncated = embed(ground_features())
    gt_stats = calculate_activation_statistics(reference)
    report['ground_truth'].update(truncated=gt_truncated, diversity=diversity(reference))
    write_report()

    def generated_features(rows, stage):
        for row in rows:
            if stage == 'raw_diffusion':
                with checked_npz(row['source_sample'], row['source_sample_sha256']) as raw:
                    feature = raw['features'].reshape(row['source_frames'], 263)
                    assert hashlib.sha256(feature.tobytes()).hexdigest() == row['feature_content_sha256']
                    yield feature
            else:
                with checked_npz(row['source'], row['source_sha256']) as final:
                    points = final['keypoints']
                    assert points.shape == (row['source_frames'], 22, 3) and float(final['fps']) == 20
                    yield encode_signfixed(points, foot_velocity_squared_threshold=.002)

    for stage in stages:
        embeddings[stage] = {}
        clipped[stage] = {}
        for task in TASKS:
            rows = [row for row in motions if row['task'] == task]
            assert len(rows) == 500
            values, n_truncated = embed(generated_features(rows, stage))
            embeddings[stage][task] = values
            clipped[stage][task] = n_truncated
            fid = float(calculate_frechet_distance(*gt_stats, *calculate_activation_statistics(values)))
            assert np.isfinite(fid) and fid >= 0
            report['tasks'].setdefault(task, {})[stage] = {
                'clips': len(values), 'fid_to_shared_heldout': fid,
                'diversity': diversity(values), 'truncated': n_truncated}
            print(task, stage, report['tasks'][task][stage], flush=True)
            write_report()
        all_values = np.concatenate([embeddings[stage][task] for task in TASKS])
        report['tasks'].setdefault('all', {})[stage] = {
            'clips': len(all_values),
            'fid_to_shared_heldout': float(calculate_frechet_distance(*gt_stats, *calculate_activation_statistics(all_values))),
            'diversity': diversity(all_values), 'truncated': sum(clipped[stage].values())}
    # Cache only the 512D embeddings to allow text scoring to resume without rerunning motion encoding.
    np.savez_compressed(args.output / 'motion_embeddings.npz', **{
        stage + '__' + task: embeddings[stage][task] for stage in stages for task in TASKS})
    report['motion_embeddings_sha256'] = sha(args.output / 'motion_embeddings.npz')
    report['status'] = 'motion_embeddings_completed_retrieval_pending'
    write_report()
    if args.tokens:
        add_retrieval(args.tokens, evaluator, assets, motions, test, embeddings, report, euclidean_distance_matrix)
        report['status'] = 'generated_reference_embedding_metrics_completed'
        write_report()


def add_retrieval(token_file, evaluator, assets, motions, test, embeddings, report, distance):
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

    tokenizer = json.loads(token_file.read_text())
    tokens = tokenizer['tokens']
    positives = sorted({r['caption'].strip() for r in motions})
    def key(caption):
        return ' '.join(caption.casefold().split())
    excluded = set(map(key, positives))
    pool = {}
    for row in sorted(test, key=lambda item: item['caption']):
        caption = row['caption'].strip()
        if caption and key(caption) not in excluded:
            pool.setdefault(key(caption), caption)
    distractors = sorted(pool.values())
    assert len(distractors) >= 31 and all(c in tokens for c in positives + distractors)
    captions = positives + distractors
    ids = {caption: i for i, caption in enumerate(captions)}
    vectorizer = WordVectorizer(str(assets / 'glove'), 'our_vab')
    word_pos = []
    lengths = []
    for caption in captions:
        seq = ['sos/OTHER', *tokens[caption][:20], 'eos/OTHER']
        lengths.append(len(seq))
        word_pos.append([vectorizer[t] for t in seq + ['unk/OTHER'] * (22 - len(seq))])
    order = np.argsort(np.array(lengths))[::-1].copy()
    word = torch.tensor(np.array([[pair[0] for pair in word_pos[i]] for i in order]), dtype=torch.float32)
    pos = torch.tensor(np.array([[pair[1] for pair in word_pos[i]] for i in order]), dtype=torch.float32)
    with torch.inference_mode():
        text = evaluator.text_encoder(word, pos, torch.tensor(np.array(lengths)[order])).numpy()[np.argsort(order)]
    assert np.isfinite(text).all()
    rng = np.random.default_rng(SEED)
    for task in TASKS:
        rows = [r for r in motions if r['task'] == task]
        ranks = {}
        for stage in embeddings:
            ranks[stage] = []
        for i, row in enumerate(rows):
            chosen = rng.choice(len(distractors), 31, replace=False) + len(positives)
            indices = rng.permutation(np.r_[ids[row['caption'].strip()], chosen])
            target = int(np.where(indices == ids[row['caption'].strip()])[0][0])
            for stage in embeddings:
                ranked = np.argsort(distance(embeddings[stage][task][i:i+1], text[indices])[0], kind='stable')
                ranks[stage].append(int(np.where(ranked == target)[0][0]) + 1)
        for stage in embeddings:
            rr = np.array(ranks[stage])
            report['tasks'][task][stage]['r_precision'] = {f'top_{k}': float((rr <= k).mean()) for k in (1, 2, 3)}
    for stage in embeddings:
        report['tasks']['all'][stage]['r_precision'] = {f'top_{k}': float(np.mean([
            report['tasks'][task][stage]['r_precision'][f'top_{k}'] for task in TASKS])) for k in (1, 2, 3)}
    report['retrieval'] = {'protocol': 'one conditioning prompt and 31 distinct held-out captions without replacement; same seeded candidates for raw and final for each clip',
                           'positives': positives, 'distractor_caption_count': len(distractors),
                           'tokens_sha256': sha(token_file), 'tokenizer': tokenizer['tokenizer'],
                           'scope': 'four repeated conditioning prompts, distinct weak-label test captions, not HumanML3D batch retrieval on this skill subset'}


if __name__ == '__main__':
    main()
