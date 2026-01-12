This python project contains a (I would say fairily standard) ML code powered by torch lightning. The code was created to run experiments on a newly developed SSL pipeline for dynamical systems. The experiments are related to the physical sciences, and in the src/data.py module you can find the dataloaders for each experiments. 

Among them, the most relevant for what I'm about to ask you is the DESRESDataset (standing for de-shaw research), implementing the logic to load data coming from their classical 2011 paper about fast folding proteins. I've heavily modified the original data from the paper, and the one I actually use are now stored into an LMDB dataset, which I produced thanks to the script pin scripts/to_lmdb.py file (more on that later). I'm using the mlcolvar library from a friend of mine (and a specifically a still unmerged branch implementing graph-nn models) for the implementations of the graph-nn architectures I'm using as backbones. 

I'm now in the process of creating a dataloader for the newer MDCATH dataset, and I've done quite a bit of work so far. You can find the current implementation in src/mdcath.py. The logic for downloading the dataset is done, and also most of the logic for loading the data from the raw .h5 file. What I need to do right now, is to construct a `Configuration` (standing for atomistic configuration), defined as such

```python
@dataclass
class Configuration:
    """
    Internal helper class that describe a given configuration of the system.
    """
    atomic_numbers: np.ndarray          # shape: [n_atoms]
    positions: np.ndarray               # shape: [n_atoms, 3], units: Ang
    cell: np.ndarray                    # shape: [n_atoms, 3], units: Ang
    pbc: Optional[tuple]                # shape: [3]
    node_labels: Optional[np.ndarray]   # shape: [n_atoms, n_node_labels]
    graph_labels: Optional[np.ndarray]  # shape: [n_graph_labels, 1]
    weight: Optional[float] = 1.0       # shape: []
    system: Optional[np.ndarray] = None       # shape: [n_system_atoms]
    environment: Optional[np.ndarray] = None  # shape: [n_environment_atoms]
```

when loading each frame. Right now, instead, my code returns a raw dictionary. The Configuration object is needed, as it is the one accepted by the mlcolvar library, and I don't want to tweak that part right now. Something I'm not currently taking care of, but you should, is to include the cell size (and if needed the pbc).

As you can see in the DESRES example in scripts/to_lmdb.py, I explicitly saved configuration files in the LMDB dataset. For DESRES I first loaded the trajectory with mdtraj, and then used the `src.data.traj_to_confs` function to convert it into configurations. I don't know if this approach is viable also for MDCATH. To explore the MDCATH dataset I've also created a small script inside exps/mdcath/read_h5.py, which recursively prints the name of the groups and elements of the .h5 file. Here's an excerpt from output of this so you maybe don't have to run it by yourself

```
-320
--0
---box
---coords
---dssp
---forces
---gyrationRadius
---rmsd
---rmsf

...

--4
---box
---coords
---dssp
---forces
---gyrationRadius
---rmsd
---rmsf
-chain
-element
-pdb
-pdbProteinAtoms
-psf
-resid
-resname
-z
```

I'm using `uv` and I launch all my commands as `uv run --env-file .env -- python -m <my modules>`. 


---

Ok, now I want to run scaling benchmark on the leonardo HPC facility. You can find the relevant code so far inside the exps/mdcath/benchmark.py file. 

The MD on the MDCATH datasets are evaluated for 5 temperatures and 5 replicas each. The total size of the dataset is ~3.6TB. What I want to measure is:

throughput in atoms/sec as a function of the #of GPUs max 4 A100s per node. Let's test it on 4 nodes maximum so far. 

Using this number, and given the total amount of "atoms" in the dataset (that is #atoms in configuration X #configurations) I would also like an estimate of the GPU hours needed to complete 1 epoch. 

Right now I only want to process _a single_ temperature (so the total size of the dataset gets trimmed down by a factor 5 to ~700GB). 

To evaluate the total number of atoms in the whole MDCATH you don't have to download it, I think that there is a master file `mdcath_source.h5` which should contain info about the number of atoms for each protein structure, and the number of frames produced in each replica. Create a short script to loop through it, compute the grand total of atoms, and save it somewhere for later. 

Remove all the unnecessary fluff from the exps/mdcath/benchmark.py file (no wandb, no checkpointing). Just focus on get the throughput on a subset of structures, and save it in readable form (yaml, toml, json, whatever you prefer). I obviously don't need microbenchmarks, so perf_counter should suffice. In the multigpu case (which is handled by lightning) you should correcly handle the logic to compute the number of atoms processed by each gpu. Everything clear? If not feel free to ask.