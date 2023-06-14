# CAD-Estate

CAD-Estate is a dataset consisting of two types of 3D annotations over RGB
videos from YouTube. The first type contains a globally-consistent 3D
representation of the objects within a video. The second type applies to videos
of indoor rooms and contains their 3D structure. It consists of structural
elements, such as walls, floors, and ceilings. The videos come from
the [RealEstate-10K](https://google.github.io/realestate10k) dataset.

This page explains the data structure and how to download CAD-Estate. It also
provides example code for reading and visualizing the data. All instructions
have been tested on Ubuntu 22.04 with Python 3.10.

## Object annotations
Coming soon.

## Room structure annotations
Link to the paper describing how the annotations were made will be available
shortly here.

### Data structure
[RealEstate-10K](https://google.github.io/realestate10k) partitions videos
into clips. We annotate each clip separately.
The room structure for a clip is stored in a Numpy NPZ file, with name
`{video-id-on-youtube}_{timestamp-of-first-frame}.npz`.
The fields contained in the NPZ are:

| Field | Type | Description |
| ------| ---- | ------------|
|clip_name             | `str`                           | Matches the filename, without the extension |
|layout_triangles      | `float32[NUM_LAYOUT_TRI, 3, 3]` | The room structure geometry |
|layout_triangle_flags | `int64[NUM_LAYOUT_TRI, 3, 3]`   | Additional per-triangle flags, used for visualization |
|layout_num_tri        | `int64[NUM_ELEMENTS]`           | Number of triangles for each structural element. The geometry of the structural elements is stored sequentially in `layout_triangles`, so the first `layout_num_tri[0]` triangles correspond to element 0, the next `layout_num_tri[1]` -- to element 1, and so on.|
|layout_labels         | `int64[NUM_ELEMENTS]`           | Semantic labels of the structural elements |

### Downloading and using the dataset
Create a directory for the code and the data and clone this repository there:
```bash
mkdir -p ~/prj/cad_estate
cd ~/prj/cad_estate
git clone https://github.com/google-research/cad-estate .
```

Download and extract the room structures from CAD-estate:
```bash
mkdir contrib
wget https://storage.googleapis.com/gresearch/cad-estate/cad_estate__lkuzgl8q.zip -P contrib/
unzip contrib/cad_estate__lkuzgl8q.zip -d contrib/
```

Download and extract `RealEstate10K.tar.gz` from the official site of
[RealEstate10K](https://google.github.io/realestate10k/):
```bash
# Download RealEstate10K.tar.gz into some directory (e.g. /some/download/dir )
tar xzvf /some/download/dir/RealEstate10K.tar.gz -C ~/prj/cad_estate/contrib
```

Create a virtual environment and install the Python requirements. The
instructions below rely on [pyenv](https://github.com/pyenv/pyenv) together
with the [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugin,
however other virtual environment managers should work as well
(e.g. [conda](https://docs.conda.io/en/latest/)). Another requirement is
`ffmpeg`.
```bash
pyenv install 3.10.9  # In case Python 3.10.9 is not installed
pyenv virtualenv 3.10.9 cad_estate
pyenv shell cad_estate && pip install -r src/requirements.txt
sudo apt install ffmpeg
```

You can now start a jupyter notebook kernel and use
[src/room_structure_notebook.ipynb](src/room_structure_notebook.ipynb)
to load and visualize room structures. The notebook requires a CUDA capable GPU.
```bash
cd src
jupyter notebook
```
