"""An example of using Ray RLlib to train a PPO agent on EnergyPlus."""
import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import ResultGrid
from tempfile import TemporaryDirectory

from rleplus.examples.registry import register_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="The gym environment to use.", required=False, default="AmphitheaterEnv", )
    parser.add_argument("--csv", help="Generate eplusout.csv at end of simulation", required=False, default=False,
        action="store_true")
    parser.add_argument("--verbose", help="In verbose mode, EnergyPlus will print to stdout", required=False,
        default=False, action="store_true", )
    parser.add_argument("--output", help="EnergyPlus output directory. Default is a generated one in /tmp/",
        required=False, default=TemporaryDirectory().name, )
    parser.add_argument("--timesteps", "-t", help="Number of timesteps to train", required=False, default=1e6)
    parser.add_argument("--num-workers", type=int, default=2, help="The number of workers to use", )

    parser.add_argument("--num-gpus-per-worker", type=float, default=0.5, help="The number of GPUs to use", )
    parser.add_argument("--num-gpus", type=int, default=0, help="The number of GPUs to use", )
    parser.add_argument("--alg", default="PPO", choices=["APEX", "DQN", "IMPALA", "PPO", "R2D2"],
        help="The algorithm to use", )
    parser.add_argument("--use-lstm", action="store_true",
        help="Whether to auto-wrap the model with an LSTM. Only valid option for " "--run=[IMPALA|PPO|R2D2]", )
    built_args = parser.parse_args()
    print(f"Running with following CLI args: {built_args}")
    return built_args


def main():
    args = parse_args()

    ray.init()

    register_all()

    # Ray configuration. See Ray docs for tuning
    config = (
        PPOConfig().environment(env=args.env, env_config=vars(args), ).training(gamma=0.95, lr=0.003, kl_coeff=0.3,
            train_batch_size=4000, sgd_minibatch_size=128, vf_loss_coeff=0.01, use_critic=True, use_gae=True,
            model={"use_lstm": args.use_lstm, "vf_share_layers": False, }, _enable_learner_api=True, ).rl_module(
            _enable_rl_module_api=True).framework(# to use tensorflow, you'll need install it first,
            # then set framework="tf2" and eager_tracing=True (for fast exec)
            framework="torch", ).resources(num_gpus=args.num_gpus,
                                           num_gpus_per_worker=args.num_gpus_per_worker).rollouts(
            num_rollout_workers=args.num_workers, rollout_fragment_length="auto", ))

    print("PPO config:", config.to_dict())

    # my_ppo = config.build()

    # .. train one iteration ..
    #    my_ppo.train()
    # .. and call `save()` to create a checkpoint.
    #    save_result = my_ppo.save()
    #    path_to_checkpoint = save_result.checkpoint.path
    #    print(
    #        "An Algorithm checkpoint has been created inside directory: "
    #        f"'{path_to_checkpoint}'."
    #    )

    # Let's terminate the algo for demonstration purposes.
    #    my_ppo.stop()
    # Doing this will lead to an error.
    # my_ppo.train()

    experiment_path = os.path.join("/home/bbrangeo/ray_results", "PPO_2024-03-27_08-54-05")
    print(f"Loading results from {experiment_path}...")
    restored_tuner = tune.Tuner.restore(experiment_path, trainable="PPO")
    result_grid = restored_tuner.get_results()
    # Check if there have been errors
    if result_grid.errors:
        print("One of the trials failed!")
    else:
        print("No errors!")
    num_results = len(result_grid)
    print("Number of results:", num_results)
    # Iterate over results
    for i, result in enumerate(result_grid):
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        # print(f"Trial #{i} finished successfully with a mean accuracy metric of:", result.metrics["mean_accuracy"])

    results_df = result_grid.get_dataframe()
    print(results_df.head())
    results_df[["training_iteration", "mean_accuracy"]]
    print("Shortest training time:", results_df["time_total_s"].min())
    print("Longest training time:", results_df["time_total_s"].max())
    best_result_df = result_grid.get_dataframe(filter_metric="mean_accuracy", filter_mode="max")
    best_result_df[["training_iteration", "mean_accuracy"]]

    from ray.train import Result

    # Get the result with the maximum test set `mean_accuracy`
    best_result: Result = result_grid.get_best_result()

    # Get the result with the minimum `mean_accuracy`
    worst_performing_result: Result = result_grid.get_best_result(metric="mean_accuracy", mode="min")
    print(best_result.config)
    print(best_result.path)
    # Get the last Checkpoint associated with the best-performing trial
    best_result.checkpoint
    # Get the last reported set of metrics
    best_result.metrics
    result_df = best_result.metrics_dataframe
    result_df[["training_iteration", "mean_accuracy", "time_total_s"]]
    best_result.metrics_dataframe.plot("training_iteration", "mean_accuracy")

    ax = None
    for result in result_grid:
        label = f"lr={result.config['lr']:.3f}, momentum={result.config['momentum']}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "mean_accuracy", label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", "mean_accuracy", ax=ax, label=label)
    ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
    ax.set_ylabel("Mean Test Accuracy")

    import torch

    with best_result.checkpoint.as_directory() as checkpoint_dir:
        # The model state dict was saved under `model.pt` by the training function
        # imported from `ray.tune.examples.mnist_pytorch`
        my_ppo.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))


if __name__ == "__main__":
    main()
