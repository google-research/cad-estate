This is not an officially supported Google product.
# CAD-Estate dataset

Current state-of-the-art methods for 3D scene understanding are driven by large
annotated datasets. To address this, we propose CAD-Estate, a large dataset of
complex multi-object RGB videos, each annotated with a globally-consistent 3D
representation of its objects, as well as with a room layout, consisting of
structural elements in 3D, such as wall, floor, and ceiling.

We annotate each object with a CAD model from a database, and place it in the 3D
coordinate frame of the scene with a 9-DoF pose transformation. Our method [1]
works on commonly-available RGB videos, without requiring a depth sensor. Many
steps are performed automatically, and the tasks performed by humans are simple,
well-specified, and require only limited reasoning in 3D. This makes them
feasible for crowd-sourcing and has allowed us to construct a large-scale
dataset. CAD-Estate offers 101K instances of 12K unique CAD models placed in the
3D representations of 20K videos. The videos of CAD-Estate offer wide complex
views of real estate properties. They pose a difficult challenge for automatic
scene understanding methods, as they contain numerous objects in each frame,
many of which are far from the camera and thus appear small. In comparison to
Scan2CAD, the largest existing dataset with CAD model annotations on real
scenes, CAD-Estate has 8x more instances and 4x more unique CAD models.

We produce generic 3D room layouts from 2D segmentation masks, which are easy to
annotate for humans. Our method [2], automatically reconstructs 3D plane
equations and spatial extents for the structural elements from the annotations,
and connects adjacent elements at the appropriate contact edges. CAD-Estate
offers room layouts for 2246 videos. The videos contain complex topologies, with
multiple rooms connected by open doors, multiple floors connected by stairs, and
generic geometry with slanted structural elements. Our automatic quality control
procedure guarantees high quality of the resulting 3D room layouts.

<p style="text-align: center;">Example of objects dataset</p>
<p align="center"><img src="doc/objects_1.gif" align="center" width=480 height=auto/></p>

<p style="text-align: center;">Example of layouts dataset</p>
<p align="center"><img src="doc/structures_4.gif" align="center" width=480 height=auto/></p>

## How to use the dataset
You need to first download the dataset and the accompanying source code,
following the instructions [here](./downloading_the_dataset.md). The text below
assumes that the code lives in `${WORKDIR}/cad_estate` and the dataset in
`${WORKDIR}/cad_estate/data`. Please set the environmental variable `WORKDIR` first,
according to the instructions.

You can visualize individual scenes with included Jupyter notebooks:
[room_structure_notebook.ipynb](./src/cad_estate/notebooks/objects_notebook.ipynb)
and
[room_structure_notebook.ipynb](./src/cad_estate/notebooks/room_structure_notebook.ipynb).
To start a Jupyter kernel for them, use:
```bash
cd ${WORKDIR}/cad_estate/src
jupyter notebook
```
The kernel requires a CUDA capable GPU.

There are also two PyTorch dataset classes for reading video frames and
their object or room structure annotations. You can find more details in the
[source code](./src/cad_estate/datasets.py)

Finally, [this file](./src/cad_estate/input_file_structures.py) describes the
structure of the CAD-Estate annotation files.

## How to cite
If you use the object annotations, please cite [1,3,4]. CAD-Estate contains
object annotations [1] that align ShapeNet [3] models over RealEstate10K
videos[4]. If you use the 3D room layouts, please cite [2,4]. CAD-Estate
contains 3D room layouts [2] over RealEstate10K videos [4].

[[1] K.-K. Maninis, S. Popov, M. Nießner, and V. Ferrari. CAD-Estate: Large-scale CAD Model Annotation in RGB Videos. In ICCV, 2023 (to appear).](https://arxiv.org/abs/2306.09011)\
[[2] D. Rozumnyi, S. Popov, K.-K. Maninis, M. Nießner, V. Ferrari. Estimating Generic 3D Room Structures from 2D Annotations. arXiv preprint, 2023.](https://arxiv.org/abs/2306.09077) \
[[3] A. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi, and Fisher Yu. ShapeNet: An Information-Rich 3D Model Repository. arXiv preprint, 2015.](https://arxiv.org/abs/1512.03012) \
[[4] T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely. Stereo Magnification: Learning view synthesis using multiplane images. In SIGGRAPH, 2018](https://research.google/pubs/pub46965).


