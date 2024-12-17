# run a separate state array on each unique non-numeric prefix of CSV files in a directory

import os, numpy as np, pandas as pd
from glob import glob
from saspt import StateArray, StateArrayDataset, RBME, load_detections
import re
from matplotlib import pyplot as plt
import sys

# read in all files from the dataset, and perform a StateArray analysis on those with the same prefix

def saspt_by_prefix(directory,outdir=None,start_frame=0):
    
        files = glob(directory + "/*.csv")
        files = [os.path.basename(f) for f in files]
        
        pattern = re.compile(r'^([a-zA-Z_]+)')

        prefixes = ['']
        
        if outdir is None:
            outdir = f"saspt_output/"
            
        os.makedirs(outdir,exist_ok=True)
        
        for prefix in prefixes:
            
            print(f'Analyzing {prefix}...')

            # This assumes that you have classified cells into just one category. The code will need to be modified for additional categories.
            input_files = glob(f'{directory}/{prefix}*csv')

            detections = load_detections(*input_files)
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
            with open(f'{outdir}/stats.txt','w') as fh:
                for k, v in SA.trajectories.processed_track_statistics.items():
                        print(f"{k : <20} : {v}")
                        fh.write(f"{k : <20} : {v}\n")


            # make some output plots, and write the overall posterior occupations to a CSV file

            SA.occupations_dataframe.to_csv(f"{outdir}/pocsv_{prefix}.csv", index=False)
            SA.plot_occupations(f"{outdir}/po_{prefix}.png")
            SA.plot_assignment_probabilities(f"{outdir}/trajpo_{prefix}.png")
            SA.plot_temporal_assignment_probabilities(f"{outdir}/framepo_{prefix}.png")


            po = pd.read_csv(f"{outdir}/pocsv_{prefix}.csv")

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
        saspt_by_prefix("../tracking_v/",outdir='saspt_v',start_frame=40)
        saspt_by_prefix("../tracking_g/",outdir='saspt_g',start_frame=80)
