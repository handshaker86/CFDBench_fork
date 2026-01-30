from tap import Tap


class Args(Tap):
    seed: int = 0
    output_dir: str = "result"
    """The directory to save the results to"""
    lr: float = 1e-3  # Initial learning rate
    lr_step_size: int = 20  # LR decays every lr_step_size epochs
    num_epochs: int = 100  # Number of epochs to train for
    eval_interval: int = 10
    """Evaluate every eval_interval epochs, and save checkpoint."""
    log_interval: int = 50  # Log training progress every log_interval batches

    loss_name: str = "nmse"
    """
    The loss function to use for training.
    Choices: ['mse', 'nmse', 'mae', 'nmae'].
    """

    mode: str = "train_test"
    """"train" or "test" for train/test only"""

    model: str = "auto_deeponet"
    """
    For autoregressive modeling (`train_auto.py`), it must be one of: ['auto_ffn', 'auto_deeponet', 'auto_edeeponet', 'auto_deeponet_cnn', 'unet', 'fno', 'cno', 'uno', 'resnet'],
    for non-autoregressive modeling (`train.py`), it must be one of: ['ffn', 'deeponet'].
    """
    in_chan: int = 2
    """Number of input channels, only applicable to autoregressive models"""
    out_chan: int = 2
    """Number of output channels, only applicable to autoregressive models"""

    batch_size: int = 128
    eval_batch_size: int = 16

    velocity_dim: int = 1
    """
    Choose the velocity dimension for training and evaluation. 
    0 for u (horizontal velocity), 1 for v (vertical velocity).
    """
    visualize: bool = False
    """Whether to visualize the dataset."""
    data_to_visualize: str = "geo"
    """
    One of 'bc', 'geo', 'prop'. Used to visualize the dataset.
    """
    cal_case_accuracy: bool = True
    """Whether to calculate the case accuracy."""

    measure_predict_time: bool = False
    """Whether to measure the prediction time."""
    measure_predict_num_frames: int = 100
    """Number of frames to measure prediction time on."""

    # Dataset hyperparamters
    data_name: str = "cylinder_bc_geo_prop"
    """
    One of: 'laminar_*', 'cavity_*', 'karman_*', where * is used to
    indicate the subset to use. E.g., 'laminar_prop_geo' trains
    on the subset of laminar task with varying geometry and physical
    properties.
    """
    data_dir: str = "data/cfdbench_dataset"
    """The directory that contains the CFDBench."""
    norm_props: int = 1
    """Whether to normalize the physical properties."""
    norm_bc: int = 1
    """Whether to normalize the boundary conditions."""
    num_rows = 64
    """Number of rows in the lattice that represents the field."""
    num_cols = 64
    """Number of columns in the lattice that represents the field."""
    # The time interval between two time step is 0.1s except for the cylinder task.
    # For the cylinder task, it is 0.001s.
    delta_time: float = 1.0
    """The time step size."""

    # robustness_test parameters
    robustness_test: bool = False
    """Whether to run the robustness test."""
    mask_range: str = "edge"
    """One of 'global', 'edge'. Used to generate the mask for robustness test."""
    noise_mode: str = "noise"
    """One of 'noise', 'zero'. noise is to add noise to the input, zero is to set the parts of input to zero."""
    noise_std: float = 0.1
    """The standard deviation of the noise."""
    edge_width: int = 1
    """The width of the edge to mask."""
    block_size: int = 0
    """The size of the block to mask."""
    num_blocks: int = 0
    """The number of blocks to mask."""

    # FFN hyperparameters
    ffn_depth: int = 8
    ffn_width: int = 100

    # DeepONet hyperparameters
    deeponet_width: int = 100
    branch_depth: int = 2
    trunk_depth: int = 1
    act_fn: str = "relu"
    act_scale_invariant: int = 1
    act_on_output: int = 0

    # Auto-FFN hyperparameters
    autoffn_depth: int = 8
    autoffn_width: int = 200

    # Auto-EDeepONet hyperparameters
    autoedeeponet_width: int = 100
    autoedeeponet_depth: int = 8
    autoedeeponet_act_fn: str = "relu"
    # autoedeeponet_act_scale_invariant: int = 1
    # autoedeeponet_act_on_output: int = 0

    # FNO hyperparameters
    fno_depth: int = 4
    fno_hidden_dim: int = 32
    fno_modes_x: int = 12
    fno_modes_y: int = 12

    # CNO hyperparameters
    cno_depth: int = 6
    cno_hidden_dim: int = 64
    cno_kernel_size: int = 5
    cno_padding: int = 2

    # U-NO hyperparameters
    uno_base_dim: int = 64
    uno_levels: int = 4
    uno_depth_per_level: int = 2
    uno_bottleneck_depth: int = 4
    uno_kernel_size: int = 5

    # CNO tiny sweep (in-process, avoids repeated data loading)
    cno_sweep: bool = False
    """If True and model=='cno', run a tiny hyperparameter sweep before full training."""
    cno_sweep_epochs: int = 5
    """Number of epochs to train each sweep config (proxy training)."""
    cno_sweep_depths: str = "4,6,8"
    """Comma-separated list of depths to sweep, e.g. '4,6,8'."""
    cno_sweep_hidden_dims: str = "64,96,128"
    """Comma-separated list of hidden dims to sweep, e.g. '64,96,128'."""
    cno_sweep_kernel_sizes: str = "3,5"
    """Comma-separated list of kernel sizes to sweep, e.g. '3,5'."""

    # UNet
    unet_dim: int = 12
    unet_insert_case_params_at: str = "input"

    # ResNet hyperparameters
    resnet_depth: int = 4
    resnet_hidden_chan: int = 16
    resnet_kernel_size: int = 7
    resnet_padding: int = 3


def is_args_valid(args: Args):
    assert any(key in args.data_name for key in ["poiseuille", "cavity", "karman"])
    assert args.batch_size > 0
    assert args.model in [
        "deeponet",
        "unet",
        "fno",
        "resnet",
        "cno",
        "uno",
        "auto_ffn",
        "auto_deeponet",
        "auto_edeeponet",
        "auto_deeponet_cnn",
    ]
