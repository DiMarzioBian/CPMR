# CPMR
This is the official PyTorch implementation of CPMR from paper 
*"CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning"* accepted by CIKM'23 (Oral).

Alias: *"Modeling Context-Aware Temporal Dynamics via Pseudo-Multi-Task Learning"*

Link: [Arxiv](https://arxiv.org/abs/2309.04802), [ACM](https://dl.acm.org/doi/10.1145/3583780.3615512).

<img alt="Model" src="figs/CPMR.png" title="Model"/>


## Data
Processed data are stored in `/data/core_5/`. If you wanna process your own data, please put the data under `/data/raw`, 
and check the preprocess scripts `/util/preprocess_raw.py` and `/util/process_data.py`.

## Usage

Please check `demo.sh` on how to run the project.

For more information about CoPE (CIKM'21), please check the official [implementation](https://github.com/FDUDSDE/CoPE) 
and my [implementation](https://github.com/DiMarzioBian/CoPE/) as well.


