# The InterAct route (reference)

ParaHome, NeuralDome, CoRoleHOI and IMHD² reached this pipeline through
**[InterAct](https://github.com/wzyabcas/InterAct)**, whose toolchain emits the
InterMimic `.pt` format that OMOMO arrives in, so every dataset's contact labels come
from InterAct's own computation.

**To use these four datasets, download the released corpus:**
<https://huggingface.co/datasets/shinben0327/hoi-retarget>. This page records how it was
produced; the conversion tooling is not part of this repository.

## Licence

InterAct is **CC BY-NC-SA 4.0**, covering the dataset *and* its software; request
non-commercial access through the form linked from their repository. Sub-datasets carry
their own terms: OMOMO is MIT; ParaHome, IMHD², CHAIRS and HODome are CC BY-NC-SA 4.0;
GRAB, BEHAVE, INTERCAP and ARCTIC use MPI custom licences.

## How the released rows were made

1. Raw data from each project site (CoRoleHOI from
   <https://huggingface.co/datasets/RayZuo/CoRoleHOI>).
2. ParaHome and CoRoleHOI converted to InterAct's canonical format; NeuralDome and IMHD²
   are native InterAct sub-datasets.
3. InterAct's `interact2mimic.py` wrote the `.pt` clips.
4. Long sequences were cut into windows, named `<sequence>__<start>_<end>`.
5. Each object was staged with `hoi-retarget-stage-object` ([DATA.md](DATA.md)), and each
   clip was retargeted with an explicit `--object_model_path`.

## Names

InterAct picks its conversion branch from the first `_` token of a dataset's name, so the
converted datasets carry different names from the released rows:

| released rows | InterAct |
|---|---|
| `parahome` | `omomo_parahome` |
| `corolehoi` | `omomo_corole` |
| `neuraldome` | `neuraldome` |
| `imhd2` | `imhd` |

Released rows record object paths as `<released name>/_assets/<object>.urdf`, resolved
under `HOI_RETARGET_OBJECT_ROOTS` ([DATA.md](DATA.md)).
