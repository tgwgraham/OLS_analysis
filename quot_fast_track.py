"""
Track TIF(F) and ND2 files by:
    1. Running detection and subpixel localization 
       on each frame of a movie in parallel
    2. Running tracking (trajectory linking) on the 
       outputs of the previous step in parallel

By Vinson Fan, 7/18/2024

usage:

python quot_fast_track.py [dir_to_track] [settings_file] [options]

options:
-e --ext                file name extension (default nd2)
-n --n_threads          number of threads
-o --out_dir            output directory
-c --contains           string that file name must contain



"""
# Filepaths
import os
from glob import glob

# CLI
import argparse

# DataFrames
import pandas as pd

# Parallelization and progress
import dask
from dask.diagnostics import ProgressBar

# quot tools
from quot.subpixel import localize_frame
from quot.chunkFilter import ChunkFilter
from quot.findSpots import detect
from quot.read import read_config
from quot.core import retrack_file


ACCEPTABLE_EXTS = [".nd2", ".tif", ".tiff"]     # If passed just one file


def localize_frames(movie_path: str, 
                    n_threads: int=4, 
                    out_dir: str=None, 
                    **kwargs):
    """
    Run detection and subpixel localization by frame-wise parallelization.

    args
    ----
    movie_path  :   str, path to movie to track
    n_threads :   int, the number of threads to use
    out_dir     :   str, output directory. If None, save the output
                    to the same directory as the movie_path
    kwargs      :   tracking configuration, as read with 
                    quot.read.read_config

    output
    ------
    write       :   CSV file with subpixel-localized spots
    return      :   str, path to the CSV file, so we can track this later

    """
    # Check that the movie path is a file
    assert os.path.isfile(movie_path), f"{movie_path} is not a file!"

    # Create the output directory if it doesn't exist
    if out_dir is not None and not os.path.isdir(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    
    # If the config file does not contain a 
    # "filter" section, don't worry about it
    kwargs["filter"] = kwargs.get("filter", {})

    # Open an image file reader with some filtering settings, if provided
    with ChunkFilter(movie_path, **kwargs['filter']) as fh:
        # Driver function to run spot detection and subpixel localization
        @dask.delayed 
        def driver(frame_idx, frame):
            detections = detect(frame, **kwargs['detect'])
            return localize_frame(frame, 
                                  detections, 
                                  **kwargs['localize']).assign(frame=frame_idx)

        # Run the driver function on each frame lazily
        tasks = [driver(i, frame) for i, frame in enumerate(fh)]
        scheduler = "single-threaded" if n_threads == 1 else "processes"
        with ProgressBar():
            print(f"Detecting and localizing spots in {movie_path} with {n_threads} threads.")
            result = dask.compute(*tasks, 
                                  scheduler=scheduler, 
                                  num_workers=n_threads)

    locs = pd.concat(result, ignore_index=True, sort=False)

    # Adjust for start index
    locs['frame'] += kwargs['filter'].get('start', 0)

    # Save to a file
    if out_dir is not None:
        out_csv_path = os.path.join(out_dir, 
                                    f"{os.path.splitext(os.path.basename(movie_path))[-2]}_trajs.csv")
    else:
        out_csv_path = os.path.splitext(movie_path)[0] + "_trajs.csv"
    
    locs.to_csv(out_csv_path, index=False)
    
    return out_csv_path


def retrack_files_threads(paths, out_suffix=None, num_workers=1, **kwargs):
    """
    Given paths to localized spots, connect the spots into trajectories
    parallelized over the number of threads.

    args
    ----
        paths       :   list of str, a set of CSV files encoding trajectories
        out_suffix  :   str, the suffix to use when generating the output 
                        paths. If *None*, then the output trajectories are 
                        saved to the original file path.
        num_workers :   int, the number of threads to use
        kwargs      :   tracking configuration

    """
    # Avoid redundant extensions
    if (not out_suffix is None) and (not ".csv" in out_suffix):
        out_suffix = "{}.csv".format(out_suffix)

    @dask.delayed 
    def task(fn):
        """
        Retrack one file.

        """
        out_csv = fn if out_suffix is None else \
            "{}_{}".format(os.path.splitext(fn)[0], out_suffix)
        retrack_file(fn, out_csv=out_csv, **kwargs["track"])

    # Run retracking on all files
    scheduler = "single-threaded" if num_workers == 1 else "threads"
    tasks = [task(fn) for fn in paths]
    with ProgressBar():
        dask.compute(*tasks, num_workers=num_workers, scheduler=scheduler)


def quot_fast_track(target_path: str, 
                    config_path: str, 
                    ext: str=".nd2", 
                    n_threads: int=1, 
                    out_dir: str=None, 
                    contains: str="*"):
    # If passed an image file, track without checking for ext or contains
    if os.path.isfile(target_path) and os.path.splitext(target_path)[-1] in ACCEPTABLE_EXTS:
        files_to_track = [target_path]
    # Otherwise, track all files matching ext and contains in the directory
    elif os.path.isdir(target_path):
        files_to_track = glob(os.path.join(target_path, f"*{contains}*{ext}"))
    # Else raise an error
    else:
        raise RuntimeError(f"Not a file or directory: {target_path}")
    
    # Try to load config
    try:
        config = read_config(config_path)
    except:
        raise RuntimeError(f"Could not load config file {config_path}")
    
    # Run detection and localization on each file, 
    # keeping track of the output files
    output_files = []
    for file in files_to_track:
        output_files.append(localize_frames(file, 
                                            n_threads=n_threads, 
                                            out_dir=out_dir, 
                                            **config))
    
    # Do tracking parallelized over the output files
    print(f"Tracking {len(output_files)} files...")
    retrack_files_threads(output_files, 
                          out_suffix=None, 
                          num_workers=n_threads, 
                          **config)    


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('target_path')
    parser.add_argument('config_path')
    parser.add_argument("-e", "--ext", nargs='?', type=str, const=".nd2", default=".nd2")
    parser.add_argument("-n", "--n_threads", nargs='?', type=int, const=1, default=1)
    parser.add_argument("-o", "--out_dir", nargs='?', type=str, const=None, default=None)
    parser.add_argument("-c", "--contains", nargs='?', type=str, const="*", default="*")
    args = parser.parse_args()

    # Run the fast tracking
    quot_fast_track(args.target_path, 
                    args.config_path, 
                    args.ext, 
                    args.n_threads, 
                    args.out_dir, 
                    args.contains)