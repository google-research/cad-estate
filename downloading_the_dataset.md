# Downloading the dataset

First, set your working directory:

```bash
export WORKDIR=/path/to/work/dir
```

All instructions below rely on the source code in this repository. You need to
download the code, create a Python virtual
environment, and install the code's dependencies in it. You also need to
install the [ffmpeg](https://ffmpeg.org/) package.
The following instructions have been tested on Ubuntu 22.04, with
[pyenv](https://github.com/pyenv/pyenv) and
[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

```bash
# Install ffmpeg.
sudo apt install ffmpeg
````

```bash
# Set up work directory and environment.
mkdir -p ${WORKDIR}/cad_estate
cd ${WORKDIR}/cad_estate
git clone https://github.com/google-research/cad-estate.git .
pyenv install 3.10.12  # Only in case Python 3.10.12 is not installed already
pyenv virtualenv 3.10.12 cad_estate
pyenv shell cad_estate && pip install -r src/requirements.txt
```

Other linux distributions and virtual environment managers
(e.g. [conda](https://docs.conda.io/en/latest/)) can be used as well.

## Download and extract the CAD-Estate annotations

CAD-Estate is divided in three parts: frame annotations, which contain
camera positions and what frames were used for manual track completion
(sec. 3.1 of this [paper](https://arxiv.org/abs/2306.09011)); object
annotations, which contain 3D object poses, and 2D object tracks; and room
structure annotations, described in this
[paper](https://arxiv.org/abs/2306.09077). To download these, use:

```bash
mkdir -p ${WORKDIR}/cad_estate/data
cd ${WORKDIR}/cad_estate/data
wget https://storage.googleapis.com/gresearch/cad-estate/frame_annotations__gq3timzs.tgz -O - | tar xzf -
wget https://storage.googleapis.com/gresearch/cad-estate/room_structures__mftdmmzw.tgz -O - | tar xzf -
wget https://storage.googleapis.com/gresearch/cad-estate/object_annotations__mfqwemtc.tgz -O - | tar xzf -
```

The dataset of objects contains 21452 scenes (`<scene_name>/objects.json`), out of which 19512 have at least one successfully aligned 3D object.
The dataset of layouts contains 2246 scenes (`<scene_name>/room_structure.npz`).

## Download and prepare ShapeNet
The CAD models in CAD-Estate are provided by [ShapeNet](https://shapenet.org/).
You need to download `ShapeNetCore.v2.zip` from ShapeNet's original site,
unpack it, and then convert the 3D meshes to a binary format. You also need
to download the class and shape symmetry information.
```bash
echo "Please download ShapeNetCore.v2.zip from ShapeNet's original site and "
echo "place it in ${WORKDIR}/cad_estate/ before running the commands below."

# Extract and process ShapeNet.
cd ${WORKDIR}/cad_estate
mkdir -p data/shape_net_raw
unzip ShapeNetCore.v2.zip -d data/shape_net_raw

# Create npz files from the obj files of ShapeNet.
# After this step, we don't use the shape_net_raw/ data anymore.
PYTHONPATH=src python -m cad_estate.preprocess_shapenet \
  --shapenet_root=data/shape_net_raw/ShapeNetCore.v2 \
  --output_root=data/shape_net_npz

# Download class and shape symmetry information.
cd ${WORKDIR}/cad_estate/data/shape_net_npz
wget https://storage.googleapis.com/gresearch/cad-estate/class_and_shape_symmetry__mvrtiobw.tgz -O - | tar xzf -
```

## Download and extract the video frames
CAD-Estate provides only annotations. To download the corresponding
videos and to extract their frames, use the
`cad_estate.download_and_extract_frames` script. Alternatively, our partner
(Technical University of Munich) also hosts video frames extracted with the
`download_and_extract_frames` script
[here](https://kaldir.vc.in.tum.de/cadestate/readme.txt).
The data visualization notebooks can download videos and extract frames on
the fly.

### Download the videos
To download the videos, use
```bash
cd ${WORKDIR}/cad_estate; pyenv shell cad_estate
PYTHONPATH=src python -m cad_estate.download_and_extract_frames \
  --cad_estate_dir="$(realpath data/)" \
  --skip_extract
```

This will download the videos into `${WORKDIR}/cad_estate/data/raw_videos` and
will log the results into:
```
${WORKDIR}/cad_estate/data/log_<date-and-time>.txt
${WORKDIR}/cad_estate/data/download_results_<date-and-time>.csv
```
The former captures all python `logging` logs, the latter contains the download
status for each video, and `<date-and-time>` is the point at which
the program was started.

The download script reports the number of failures at the end. In case this
number is not 0, inspect `download_results_<date-and-time>.csv`, otherwise
proceed to the next section (frame extraction).

The first column in the CSV file is the video ID, the second indicates
whether the download was successful, and the last once contains the reason for
failure.
Videos that have failed because they are private or no longer available can
no longer be downloaded. At the time of writing, there are no such videos.
Some videos fail because of timeout or bandwidth restrictions. Running
`download_and_extract_frames` again will fix these. Note that
`download_and_extract_frames` will only attempt to download the failed videos.
For the remaining failed videos, check if you can view them on YouTube,
using `https://www.youtube.com/watch?v=<video-id>`.
If this is the case, try upgrading the `yt-dlp` Python package
(this is what `download_and_extract_frames` uses to download videos).
Also, try to download the videos manually into
`${WORKDIR}/cad_estate/data/raw_videos/<video-id>.mp4`, using `yt-dlp` from the
command line. This can help videos that fail due to
an "HTTP Error 500". You have to specify the video format explicitly in the
command line of `yt-dlp` in this case, rather than
using `-f bestvideo`, as `download_and_extract_frames` does.

### Extract the video frames
The next step is to extract the video frames, using:

```bash
cd ${WORKDIR}/cad_estate; pyenv shell cad_estate
PYTHONPATH=src python -m cad_estate.download_and_extract_frames \
  --cad_estate_dir="$(realpath data/cad_estate)" \
  --parallel_extract_tasks 4 \
  --skip_download
```
In the arguments above, `parallel_extract_tasks` specifies how many instances
of `ffmpeg` to run in parallel (4 is a good number for a 12 core CPU). Like
above, the script produces two log:
```
${WORKDIR}/cad_estate/data/log_<date-and-time>.txt
${WORKDIR}/cad_estate/data/process_results_<date-and-time>.csv
```
Again, the former contains Python logs. The latter is a CSV, containing
the video ID, the frame extraction status, the maximum timestamp discrepancy,
and in case of failure -- the reason.

For successfully processed videos, inspect the maximum timestamp discrepancy
(third column). It contains the maximum difference between the time stamp of any
frame annotation and the corresponding nearest extracted frame.
The discrepancy should not exceed 2000-3000 µs for any video.

If the there are any new errors (i.e. the extract script reports more
errors than the download script), inspect the fourth column.
All videos that failed to download will fail here with a
reason "No such file or directory". For any other failures,
run `ffmpeg` on the downloaded video, try to find options that work for it,
and patch `download_and_extract_frames` to use this options.
The log file (`log_<date-and-time>.txt`) contains the `ffmpeg` command executed
by `download_and_extract_frames`.
If frame extraction is interrupted, you can run `download_and_extract_frames`
again, which will resume the frame extraction process from where it is left.

### Create the splits
The script will output a list of CAD-Estate scenes with successfully
processed videos in `${WORKDIR}/cad_estate/data/scene_list_<date-and-time>.txt`.
Use this to create the dataset splits, which are later consumed by the provided
PyTorch dataset classes.

```bash
export LC_ALL=C  # traditional sort order, using native byte values.

# Create the object splits
cat scene_list.txt | awk \
  '{ if(($0 <= "0gBW3RDqyMc_122914456") \
        || ($0 >= "5FpLWL4jcsQ_170003333" && $0 <= "6radJfppz70_89856000") \
       ) print > "obj_test.txt"; else print > "obj_train.txt"}'

# Create the room structure split
join \
  <(find annotations | grep room_structure.npz$ | awk -F/ '{print $2}' | sort) \
  <(cat scene_list.txt | sort) > struct_all.txt
```
