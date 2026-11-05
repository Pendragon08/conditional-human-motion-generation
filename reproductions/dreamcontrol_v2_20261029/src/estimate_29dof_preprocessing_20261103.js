// Count indexed source-rate frames and project one measured window benchmark.
// This is a what-if workload, not an ETA, an acceptance decision or a train job.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const files = {
  bml: 'data/bmlmovi_smplx_inventory_20261102/records.jsonl',
  eyes: 'data/eyesjapan_smplx_inventory_20261102/records.jsonl',
  kit: 'data/kit_smplx_stream_inventory_20261102/records.jsonl',
  kit_large: 'data/kit_smplx_large_members_20261102/records.jsonl',
  grab: 'reports/grab_source_registry_20261030.jsonl',
  benchmark: 'reports/gmr29_pyroki_full_airplane_audit_20261103.json',
};
const bytes = p => fs.readFileSync(path.join(root, p));
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const inputs = Object.fromEntries(Object.entries(files).map(([k, p]) => [k, bytes(p)]));
const lines = b => b.toString('utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const benchmark = JSON.parse(inputs.benchmark);
const measuredGpuSecondsPerWindow = (benchmark.elapsed_s - benchmark.first_window_s) / (benchmark.windows - 1);
const measuredGmrSecondsPerFrame = 25.256877349689603 / 1198;
function windowCount(n) {
  if (!Number.isInteger(n) || n < 8) return 0;
  return Math.floor((n - 8) / 4) + 1 + ((n - 8) % 4 === 0 ? 0 : 1);
}
function count(rows, frameKey) {
  const vals = rows.map(frameKey).filter(n => Number.isInteger(n) && n > 0);
  const frames = vals.reduce((a, b) => a + b, 0);
  const windows = vals.reduce((a, n) => a + windowCount(n), 0);
  return {matched_sources: rows.length, usable_frame_counts: vals.length,
    sequences_shorter_than_8: vals.filter(n => n < 8).length,
    raw_source_frames: frames, estimated_windows: windows,
    projected_gmr_cpu_hours: frames * measuredGmrSecondsPerFrame / 3600,
    projected_gpu_hours_after_one_compilation: windows * measuredGpuSecondsPerWindow / 3600};
}
const kit = new Map(lines(inputs.kit).filter(x => x.poses_shape).map(x => [x.member, x]));
for (const x of lines(inputs.kit_large)) {
  if (kit.has(x.member)) throw Error('KIT duplicate in supplemental members');
  kit.set(x.member, x);
}
const groups = {
  BMLmovi: count(lines(inputs.bml), x => x.poses_shape?.[0]),
  EyesJapan: count(lines(inputs.eyes), x => x.poses_shape?.[0]),
  KIT_matched: count([...kit.values()], x => x.poses_shape?.[0]),
  GRAB: count(lines(inputs.grab), x => x.frames),
};
if (groups.BMLmovi.matched_sources !== 1801 || groups.EyesJapan.matched_sources !== 749 ||
    groups.KIT_matched.matched_sources !== 4231 || groups.GRAB.matched_sources !== 1335 ||
    groups.BMLmovi.raw_source_frames !== 1216725 || groups.EyesJapan.raw_source_frames !== 2857698)
  throw Error('inventory coverage unexpected');
const result = {
  kind: 'illustrative_from_scratch_whole_matched_source_workload_not_eta',
  input_sha256: Object.fromEntries(Object.entries(inputs).map(([k, b]) => [k, sha(b)])),
  benchmark: {source_frames: 1198, windows: benchmark.windows,
    total_seconds: benchmark.elapsed_s, first_window_seconds: benchmark.first_window_s,
    mean_subsequent_solver_seconds: benchmark.mean_subsequent_window_s,
    effective_seconds_per_window_including_observed_overhead: measuredGpuSecondsPerWindow,
    gmr_seconds_per_frame_ik_loop_only: measuredGmrSecondsPerFrame},
  groups,
  limits: ['Includes full matched raw recordings, which can be more than text-labelled clips; excludes other AMASS/HumanML sources.',
    'One 1198-frame GRAB sample, existing 4-frame-stride JAX solver, serial one GPU and one CPU IK worker; no quality acceptance.',
    'First GPU compilation excluded from projected per-window hours; cache reuse across clips is assumed, not proven.',
    'Source reads, SMPL construction, data transfer, cropping, review, repair, interruptions, checkpoints, training and evaluation excluded.',
    'GMR and GPU may pipeline; do not automatically add CPU and GPU hours as wall time.',
    'Unreliable until the continuity/limits defect is fixed and a representative multi-clip batch is measured.'],
};
const output = path.join(root, 'reports', 'preprocessing_workload_estimate_20261103.json');
fs.writeFileSync(output, JSON.stringify(result, null, 2), {flag: 'wx'});
console.log(JSON.stringify(result, null, 2));
