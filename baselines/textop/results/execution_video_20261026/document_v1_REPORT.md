# Independent actual CPU software-rendering document witness

Executed the exact PowerShell invocation in EXECUTION_VIDEO_ENVIRONMENT.md once. Actual PowerShell/helper exit0, actual remote supervisor/worker exit0, stderr empty. Command, stdout, stderr and actual exit are preserved alongside witness.json.

- Status: `SOFTWARE_RENDERER_SYNTHETIC_WITNESS_PASSED`.
-24 frames encoded and actually decoded at50fps; closed MP4 is462654bytes.
- GL vendor `Mesa/X.org`, renderer `llvmpipe (LLVM 12.0.0, 256 bits)`, loaded NVIDIA libraries: empty.
- SDK runs0; new physics steps0; optimizer updates0; training signals0. Original16 and live quick-controller bytes unchanged. No original job, flag, package, dependency or shared model changed.
- Actual policy result video: false. These are synthetic renderer fixtures, not robot tracking, filtering, generator, closed-loop or sim-to-sim results.

## Media and receipt hashes

- MP4 `2a69103f2c0c201f7d49a25ad0e808d124a16a54371d7a50a732d62a0e8d92a2`; matches actual local file and author media.
- First frame `fc2127ce7c1bbc6964b846de190bd1123a50dfefade242c56dc8ea8e42231869`; matches actual local PNG and author image.
- Actual renderer receipt `ef05b8f169798f9b0f323ccea467b7b069d28135d87eda49290e4d3b9d3b8981`.
- Actual supervisor exit `97338dd447732445fa6e3737509eafb2fb0b0715383e9a79bc416486a0303557`.
- Source `2225692cbe72c230f68355eead9a3e1c0ed5a4ea9e0b62e6a1b0c1520a013945`; render signature `dbb4301b20d56abf03619cad25854a8d53769eea980950fb39ecc0de0a9cabc5`.

## Visual/document findings

The PNG was actually inspected. G1 is recognizable and the SYNTHETIC POSES / NOT AN EXPERIMENT RESULT watermark is clear.

A meaningful visual issue remains: the ground is substantially mottled, consistent with z-fighting. Read-only inspection of the original GMR MJCF confirms an existing world plane named floor at defaultz0; the renderer adds another video_floor plane atz0. Reuse/remove one visual ground in the process-local scene before actual result videos. Do not alter the shared MJCF or discard this witness. No fix or rerun was performed by this agent.

Minor document wording difference: the actual-mode overlay labels G1 tracker/5000 and pose replay/no MuJoCo simulation, while the receipt explicitly names recorded IsaacLab poses. The document's statement that video text explicitly distinguishes IsaacLab versus MuJoCo is stronger than the current overlay. Correct the wording or label before claiming this distinction in an actual video.

The documented command matched reality and the CPU rendering/encoding/decoding checks passed. Actual experiment videos remain pending; this pass should not be described as visually issue-free.
