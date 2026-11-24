# Model and media provenance

Weights are downloaded separately and are not committed to this repository.

| Asset | Version/source | SHA-256 |
| --- | --- | --- |
| MediaPipe Pose Landmarker Full | Google float16 model bundle, version 1 | `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1` |
| TrackNet V1 tennis checkpoint | [vishnushenoy09/tracknet-v1-tennis](https://huggingface.co/vishnushenoy09/tracknet-v1-tennis), revision `c53bc2f8d5524596e70737c5e6aa9b35bfb52e9b` | `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076` |

The download command verifies digests. TrackNet uses `torch.load(..., weights_only=True)` and strict architecture-key matching. Its publisher labels the checkpoint MIT; this does not grant rights to unrelated videos or datasets. Consult MediaPipe's distribution and model documentation for applicable terms.

Architecture: Huang et al., *TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications*, AVSS 2019 ([paper](https://arxiv.org/abs/1907.03698)); key layout follows [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet). This project did not train the checkpoint.

## Smoke-test footage

`scripts/fetch_sample.py` downloads a 7.13-second clip from [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis/blob/2595ef08a38d00288bcdf21f873b602ffbb2798d/input_videos/input_video.mp4), SHA-256 `5f2584a2010864d5d8624c325d7f0571d48bd0756164657ef603dc8a97bab22d`.

The footage bears ATP/Tennis TV branding. It is downloaded for local technical tests, not redistributed as an application dataset. The README screenshot contains an attributed still from this test. No license for the underlying footage is asserted. Its broadcast scoreboard is ignored. The smoke configuration begins at love-all to exercise the scoring API and does not reproduce the real match score.

For a meaningful amateur-tennis benchmark, collect authorized fixed-camera clips across matches, surfaces, lighting, occlusion, and player scales. Record match/camera IDs for leakage-free splits. Human ball, visibility, contact, and point-ending labels must stay separate from corrected model predictions. Access/licensing for external training datasets must be checked with their original providers.
