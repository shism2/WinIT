# tsi-research
Time Series Interpretability Research

## Environment

This requires:
- Python 3.7
- Anaconda
- (GPU) CUDA 10.1

Tested on Ubuntu 18.04.

### Anaconda

Install the individual edition:

```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
chmod +x Anaconda3-2021.05-Linux-x86_64.sh
./Anaconda3-2021.05-Linux-x86_64.sh
```

Restart your shell after installing so that you are using the base conda environment.

### Setup Conda Environment

```
conda env create -f environment.yml
```

# Run paper experiments

## Generate synthetic datasets

Generate spike datasets:

```
python -m FIT.data_generator.simulations_threshold_spikes
```

This will generate five datasets and store them in `data/`:
 - The original spike dataset (`data/simulated_spike_data`)
 - Four spike datasets with delays of 1 through 4 (`data/simulated_spike_data_delay_X`).
 
## Run benchmarks

```
python paper_experiments_2.py
```

And will store the results in the csv file specified at the beginning of the script.