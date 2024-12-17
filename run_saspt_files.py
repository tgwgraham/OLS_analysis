# Run SASPT on specific sets of input files.
# You can specify the output directory, start frame, and prefix

import os, numpy as np, pandas as pd
from glob import glob
from saspt import StateArray, StateArrayDataset, RBME, load_detections
import re
from matplotlib import pyplot as plt
import sys

# read in all files from the dataset, and perform a StateArray analysis on those with the same prefix

def saspt_file_list(files,outdir=None,start_frame=0,prefix=""):
                    
        if outdir is None:
            outdir = f"saspt_output/"
            
        os.makedirs(outdir,exist_ok=True)

        detections = load_detections(*files)
        settings = dict(
                    likelihood_type = RBME,
                    pixel_size_um = 0.108,
                    frame_interval = 0.0100433,
                    focal_depth = 1.1,  # 2*0.670*1.33/(1.27^2)
                    start_frame = start_frame,
                    progress_bar = True,
                    sample_size = 100000,
                    num_workers = 4,
                    )
        SA = StateArray.from_detections(detections, **settings)
        print(SA)
        print("Trajectory statistics:")
        with open(f'{outdir}/{prefix}_stats.txt','w') as fh:
            for k, v in SA.trajectories.processed_track_statistics.items():
                    print(f"{k : <20} : {v}")
                    fh.write(f"{k : <20} : {v}\n")


        # make some output plots, and write the overall posterior occupations to a CSV file

        SA.occupations_dataframe.to_csv(f"{outdir}/{prefix}_pocsv.csv", index=False)
        SA.plot_occupations(f"{outdir}/{prefix}_po.png")
        SA.plot_assignment_probabilities(f"{outdir}/{prefix}_trajpo.png")
        SA.plot_temporal_assignment_probabilities(f"{outdir}/{prefix}_framepo.png")


        po = pd.read_csv(f"{outdir}/{prefix}_pocsv.csv")

        grouped = po.groupby('diff_coef',as_index=False).sum()
        D = grouped['diff_coef'].to_numpy()
        mpo = grouped['mean_posterior_occupation']
        plt.close()
        plt.semilogx(D,mpo)
        plt.xlabel('Diffusion coefficient (µm$^2$/s)')
        plt.ylabel('Fraction of molecules');
        plt.savefig(f'{outdir}/{prefix}_mpo.png',dpi=300,bbox_inches='tight')
        plt.savefig(f'{outdir}/{prefix}_mpo.pdf',bbox_inches='tight')


if __name__ == "__main__":
        saspt_file_list(["sorted_trajectories/all_g.csv"],outdir='saspt_sorted',start_frame=80,prefix="g")
        saspt_file_list(["sorted_trajectories/all_v.csv"],outdir='saspt_sorted',start_frame=40,prefix="v")
