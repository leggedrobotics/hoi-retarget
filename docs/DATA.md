# Getting the data

Nothing on this page ships with the repository. Every source carries its own
licence, several require registration, and one (InterAct) requires written
authorisation before any sub-dataset is redistributed — which is why we
redistribute none of them. See `THIRD_PARTY_NOTICES.md` for the rights analysis.

Paths below are the defaults. All are overridable:

```bash
export HOI_RETARGET_ASSETS=/somewhere/assets     # robots, objects, body models
```

For object meshes specifically there is a third, and it is the one to reach for when
working from the released corpus:

```bash
export HOI_RETARGET_OBJECT_ROOTS=/somewhere/meshes   # os.pathsep-separated, searched in order
```

It only adds places to look for object meshes, so unlike `HOI_RETARGET_ASSETS` it cannot
hide the robot models. Lay the directory out as the corpus records its paths:

```
<root>/assets/objects/<object>.urdf          OMOMO (ships with this repo and with the dataset)
<root>/<dataset>/_assets/<object>.urdf       ParaHome, NeuralDome, CoRoleHOI, IMHD2
```

A released row whose `object_model_path` is an absolute path from the machine that produced
it also resolves, by matching its tail under these roots.

---

## 1. Body models — required for everything

### SMPL-X (always)

The human source motion is a SMPL-X body, so nothing runs without these.

1. Register at <https://smpl-x.is.tue.mpg.de/> and accept the licence.
2. Download the SMPL-X models.
3. Place them so the tree is:

```
assets/body_models/smplx/SMPLX_NEUTRAL.npz
assets/body_models/smplx/SMPLX_MALE.npz
assets/body_models/smplx/SMPLX_FEMALE.npz
```

The download ships each model as both `.npz` and `.pkl`; either set works.

These may not be redistributed by anyone, including you. `assets/body_models/`
is gitignored and a test fails the build if anything under it is ever committed.

### SMPL-H (only for CARI4D)

CARI4D fits an **SMPL-H** body, and reading it as SMPL-X without correction puts
the subject about 10 cm out of place. The converter bridges the two models, which
means it needs SMPL-H as well:

1. Register at <https://mano.is.tue.mpg.de/download.php> and accept the licence.
2. Download the SMPL+H model **for the `smplx` python package** (`smplx.zip`); it contains
   `smplh/SMPLH_{FEMALE,MALE}.pkl`. The AMASS variant ships `model.npz` files instead,
   which this pipeline does not read.
3. Place the models as:

```
assets/body_models/smplh/SMPLH_MALE.pkl
assets/body_models/smplh/SMPLH_FEMALE.pkl
```

Skip this unless you are using CARI4D. The error message says the same thing if
you forget.

---

## 2. Motion sources

Every InterMimic-format clip (`.pt`) is retargeted from its own recorded skeleton, i.e. the
subject's body model and shape, so contact targets sit at the subject's real palms. SMPL-X
only supplies joint orientations.

### OMOMO via InterMimic — the main benchmark

The paper's quantitative results use 4,421 OMOMO clips across 13 object
categories. The pipeline reads InterMimic's re-processed `OMOMO_new` release
directly, with no conversion step.

Obtain it from <https://github.com/Sirui-Xu/InterMimic> under its terms, and
place the `.pt` clips at:

```
data/InterMimic/OMOMO_new/sub10_whitechair_049.pt
```

Then:

```bash
hoi-retarget --input_file data/InterMimic/OMOMO_new/sub10_whitechair_049.pt \
             --out_dir outputs/sub10_whitechair_049
```

The **object meshes** for these 13 categories already ship in `assets/objects/`
(OMOMO is MIT-licensed, so they can be). Only the motion needs fetching.


### CARI4D — monocular video

Human-object interaction reconstructed from a single RGB camera, used in
Sec. IV-C. Run CARI4D on your own footage — see
<https://github.com/NVlabs/CARI4D> — and point the pipeline at its
`opt`-stage output.

```bash
hoi-retarget --cari4d_sequence    Date03_Sub01_OrangeTable9-wild \
             --cari4d_opt_dir     /path/to/cari4d/output/opt \
             --cari4d_meshes_root /path/to/cari4d/data/meshes_metric \
             --out_dir outputs/orangetable9
```

Both roots are needed: CARI4D writes poses under `output/opt/` and object meshes
under `data/meshes_*/`, which are siblings. `--cari4d_meshes_root` defaults to
`data/CARI4D/meshes`, so leaving it out fails unless your layout happens to match.
Each sequence's mesh is found as `<meshes_root>/<sequence>*{_rgba,_align}/*_align.obj`.

Object URDFs and 500 farthest-point surface samples are staged on first use, into
`--cari4d_assets_root`. Gender is read from the subject code in the filename; pass
`--gender` only to override it. The G1 object scale follows the subject's height
(0.83 for a 1.66 m subject), so CARI4D clips vary around 0.83.

### ParaHome, NeuralDome, CoRoleHOI, IMHD²

These went through InterAct's toolchain, so their contact labels come from InterAct's own
computation, like OMOMO's. Use the released corpus (section 3); [`INTERACT.md`](INTERACT.md)
records how it was produced and what of that route is not shipped. CoRoleHOI is the only
two-actor source: every motion is half of a pair, and the two halves are solved
independently.

### Object meshes

Only OMOMO's 13 objects ship, here under `assets/objects/` and in the released corpus; the
other datasets' meshes are not redistributable. To use released rows from those datasets,
place their meshes, obtained through InterAct, as `<root>/<dataset>/_assets/<object>.urdf`
under a `HOI_RETARGET_OBJECT_ROOTS` root, matching the rows' recorded paths.

For your own objects, stage each mesh (below) and pass the URDF as `--object_model_path`.
Pass it explicitly for anything outside OMOMO: several datasets name an object OMOMO also
has (`monitor`, `trashcan`, `suitcase`).

#### `sample_points.npy`

A cloud of points on the object's surface. The pipeline rotates it each frame and takes
the lowest point, which is how the object is kept from sinking through the floor. It sits
beside the mesh, and **without it the object rests on a zero floor** — visibly wrong for
anything that is not flat-bottomed. InterAct's own URDF generator does not write one.

To stage an object:

```bash
hoi-retarget-stage-object mug.obj --name mug --assets-root data/my_dataset/_assets
```

That writes the mesh, the URDF and 1024 surface samples, and prints the URDF path to pass
as `--object_model_path`. Object poses rotate about the mesh's centroid, so the tool warns
if yours is not at the origin; `--recenter` moves it. Two exceptions to the 1024: OMOMO's
shipped clouds have 340 and came that way, and CARI4D stages 500 of its own on first use.

---

## 3. What the released corpus already contains

If you want retargeted trajectories rather than the means to produce them, they
are published separately:

<https://huggingface.co/datasets/leggedrobotics/hoi-retarget>

6,952 motions across 5 source datasets and 2 robots, with per-link contact flags
and QC metrics, under CC BY-NC-SA 4.0. `examples/to_pkl.py` in that repository
converts a row back into the `contact_window.pkl` this pipeline writes, which the
viewer and renderers read directly. Editing contacts needs the row's source clip.
