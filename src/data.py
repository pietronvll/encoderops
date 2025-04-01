from typing import Union

import mdtraj
import numpy as np
import torch
from mlcolvar.data import DictDataset
from mlcolvar.data.graph.utils import create_dataset_from_configurations
from mlcolvar.utils.io import (
    _configures_from_trajectory,
    _names_from_top,
    _z_table_from_top,
)
from mlcolvar.utils.timelagged import find_timelagged_configurations, tprime_evaluation

from src.configs import DataArgs


def get_dataset(
    trajectory_files: list[str],
    top: str,
    data_args: DataArgs,
):
    traj = mdtraj_load(trajectory_files, top, data_args.stride)
    # Create configs
    configs, z_table, atom_names = traj_to_confs(
        traj, system_selection=data_args.system_selection
    )
    # Create dataset (?)
    dataset = create_dataset_from_configurations(
        configs,
        z_table,
        data_args.cutoff,
        0.0,
        atom_names,
        data_args.remove_isolated_nodes,
        False,
    )
    # Lagged dataset (?)
    timelagged_dataset = create_timelagged_dataset(dataset, progress_bar=False)
    return timelagged_dataset


def traj_to_confs(traj: mdtraj.Trajectory, system_selection: str | None = None):
    configs = _configures_from_trajectory(traj, system_selection=system_selection)
    z_table = _z_table_from_top([traj.top])
    atom_names = _names_from_top([traj.top])
    return configs, z_table, atom_names


def mdtraj_load(trajectory_files: list[str], top: str, stride: int = 1000):
    traj = mdtraj.load(trajectory_files, top=top, stride=stride)
    traj.top = mdtraj.core.trajectory.load_topology(top)
    return traj


def create_timelagged_dataset(
    X: Union[torch.Tensor, np.ndarray, DictDataset],
    t: torch.Tensor = None,
    lag_time: float = 1,
    reweight_mode: str = None,
    logweights: torch.Tensor = None,
    tprime: torch.Tensor = None,
    interval: list = None,
    progress_bar: bool = False,
):
    """
    Create a DictDataset of time-lagged configurations.

    In case of biased simulations the reweight can be performed in two different ways (``reweight_mode``):

    1. ``rescale_time`` : the search for time-lagged pairs is performed in the accelerated time (dt' = dt*exp(logweights)), as described in [1]_ .
    2. ``weights_t`` : the weight of each pair of configurations (t,t+lag_time) depends only on time t (logweights(t)), as done in [2]_ , [3]_ .

    If reweighting is None and tprime is given the `rescale_time` mode is used. If instead only the logweights are specified the user needs to choose the reweighting mode.

    References
    ----------
    .. [1] Y. I. Yang and M. Parrinello, “Refining collective coordinates and improving free energy
        representation in variational enhanced sampling,” JCTC 14, 2889–2894 (2018).
    .. [2] J. McCarty and M. Parrinello, "A variational conformational dynamics approach to the selection
        of collective variables in meta- dynamics,” JCP 147, 204109 (2017).
    .. [3] H. Wu, et al. "Variational Koopman models: Slow collective variables and molecular kinetics
        from short off-equilibrium simulations." JCP 146.15 (2017).

    Parameters
    ----------
    X : array-like
        input descriptors
    t : array-like, optional
        time series, by default np.arange(len(X))
    reweight_mode: str, optional
        how to do the reweighting, see documentation, by default none
    lag_time: float, optional
        lag between configurations, by default = 10
    logweights : array-like,optional
        logweight of each configuration (typically beta*bias)
    tprime : array-like,optional
        rescaled time estimated from the simulation. If not given and `reweighting_mode`=`rescale_time` then `tprime_evaluation(t,logweights)` is used
    interval : list or np.array or tuple, optional
        Range for slicing the returned dataset. Useful to work with batches of same sizes. Recall that with different lag_times one obtains different datasets, with different lengths
    progress_bar: bool
        Display progress bar with tqdm

    Returns
    -------
    dataset: DictDataset
        Dataset with keys 'data', 'data_lag' (data at time t and t+lag), 'weights', 'weights_lag' (weights at time t and t+lag).

    """

    # check reweigthing mode if logweights are given:
    # 1) if rescaled time tprime is given
    if tprime is not None:
        if reweight_mode is None:
            reweight_mode = "rescale_time"
        elif reweight_mode != "rescale_time":
            raise ValueError(
                "The `reweighting_mode` needs to be equal to `rescale_time`, and not {reweight_mode} if the rescale time `tprime` is given."
            )
    # 2) if logweights are given
    elif logweights is not None:
        if reweight_mode is None:
            reweight_mode = "rescale_time"
            # TODO output warning or error if mode not specified?
            # warnings.warn('`reweight_mode` not specified, setting it to `rescale_time`.')

    # define time if not given
    if t is None:
        t = torch.arange(0, len(X))
    else:
        if len(t) != len(X):
            raise ValueError(
                f"The length of t ({len(t)}) is different from the one of X ({len(X)}) "
            )

    # define tprime if not given:
    if reweight_mode == "rescale_time":
        if tprime is None:
            tprime = tprime_evaluation(t, logweights)
    else:
        tprime = t

    # find pairs of configurations separated by lag_time
    if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
        x_t, x_lag, w_t, w_lag = find_timelagged_configurations(
            X,
            tprime,
            lag_time=lag_time,
            logweights=logweights if reweight_mode == "weights_t" else None,
            progress_bar=progress_bar,
        )
    elif isinstance(X, DictDataset):
        index = torch.arange(len(X), dtype=torch.long)
        x_t, x_lag, w_t, w_lag = find_timelagged_configurations(
            index,
            tprime,
            lag_time=lag_time,
            logweights=logweights if reweight_mode == "weights_t" else None,
            progress_bar=progress_bar,
        )

    # return only a slice of the data (N. Pedrani)
    if interval is not None:
        # convert to a list
        data = list(x_t, x_lag, w_t, w_lag)
        # assert dimension of interval
        assert len(interval) == 2
        # modifies the content of data by slicing
        for i in range(len(data)):
            data[i] = data[i][interval[0] : interval[1]]
        x_t, x_lag, w_t, w_lag = data

    if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
        dataset = DictDataset(
            {"data": x_t, "data_lag": x_lag, "weights": w_t, "weights_lag": w_lag},
            data_type="descriptors",
        )
        return dataset

    elif isinstance(X, DictDataset):
        # we use deepcopy to avoid editing the original dataset
        dataset = DictDataset(
            dictionary={
                "data_list": X[x_t.numpy().tolist()]["data_list"],
                "data_list_lag": X[x_lag.numpy().tolist()]["data_list"],
            },
            metadata={"z_table": X.metadata["z_table"], "cutoff": X.metadata["cutoff"]},
            data_type="graphs",
        )
        # update weights
        for i in range(len(dataset)):
            dataset["data_list"][i]["weight"] = w_t[i]
            dataset["data_list_lag"][i]["weight"] = w_lag[i]

        return dataset
        return dataset
