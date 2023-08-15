import argparse
import json
import numpy as np
import itertools

from ray.rllib.algorithms import Algorithm
from multigrid.utils.training_utilis import algorithm_config, get_checkpoint_dir, policy_mapping_fn



def visualize(algorithm: Algorithm, num_episodes: int = 100,  teams: dict[str, int] = {"red": 1}, training_scheme: str = "CTCE", num_agents: int = 2) -> list[np.ndarray]:
    """
    Visualize trajectories from trained agents.
    """
    frames = []
    env = algorithm.env_creator(algorithm.config.env_config)
    agent_index_dict = {agent_id: next(team for team, count in teams.items() if sum(teams[t] for t in itertools.takewhile(lambda x: x != team, teams)) + count > agent_id) for agent_id in range(num_agents)}

    for episode in range(num_episodes):
        print('\n', '-' * 32, '\n', 'Episode', episode, '\n', '-' * 32)

        episode_rewards = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminations, truncations = {'__all__': False}, {'__all__': False}
        observations, infos = env.reset()
        states = {
            agent_id: algorithm.get_policy(agent_id).get_initial_state()
            for agent_id in env.get_agent_ids()
        }
        while not terminations['__all__'] and not truncations['__all__']:
            frames.append(env.get_frame())

            actions = {}
            for agent_id in env.get_agent_ids():

                # Single-agent
                actions[agent_id] = algorithm.compute_single_action(
                    observations[agent_id],
                    states[agent_id],
                    policy_id=agent_id
                )

                # # Multi-agent
                # actions[agent_id], states[agent_id], _ = algorithm.compute_single_action(
                #     observations[agent_id],
                #     states[agent_id],
                #     policy_id=policy_mapping_fn(agent_id)
                # )

            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in rewards:
                episode_rewards[agent_id] += rewards[agent_id]

        frames.append(env.get_frame())
        print('Rewards:', episode_rewards)

    env.close()
    return frames



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algo', type=str, default='PPO',
        help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        '--framework', type=str, choices=['torch', 'tf', 'tf2'], default='torch',
        help="Deep learning framework to use.")
    parser.add_argument(
        '--lstm', action='store_true', help="Use LSTM model.")
    parser.add_argument(
        '--env', type=str, default='MultiGrid-CompetativeRedBlueDoor-v3-CTCE-Red',
        help="MultiGrid environment to use.")
    parser.add_argument(
        '--env-config', type=json.loads, default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')")
    parser.add_argument(
        '--num-agents', type=int, default=2, help="Number of agents in environment.") # FIXME - streamline this with
    parser.add_argument(
        '--num-episodes', type=int, default=10, help="Number of episodes to visualize.")
    parser.add_argument(
        '--load-dir', type=str,
        help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        '--gif', type=str, help="Store output as GIF at given path.")
    parser.add_argument(
        '--our-agent-ids', nargs="+", type=int, default=[0,1],
        help="List of agent ids to evaluate")
    parser.add_argument(
        '--teams', type=json.loads, default={"red": 1, "blue": 1}, #  "blue": 2 # TODO - map this with env config
        help='A dictionary containing team name and counts, e.g. \'{"red": 2, "blue": 2}\'')
    parser.add_argument(
        '--training-scheme', type=str, default='CTDE',
        help="Can be either 'CTCE', 'DTDE' or 'CTDE'")

    args = parser.parse_args()
    args.env_config.update(render_mode='human')
    config = algorithm_config(
        **vars(args),
        num_workers=0,
        num_gpus=0,
    )
    config.environment(disable_env_checking=True)
    algorithm = config.build()
    checkpoint = get_checkpoint_dir(args.load_dir)

    if checkpoint:
        from ray.rllib.policy.policy import Policy

        print(f"Loading checkpoint from {checkpoint}")
        algorithm.restore(checkpoint)

        # # TODO update checkpoint loading method
        # # New way
        # policy_name = f"policy_{args.our_agent_ids[1]}"
        # restored_policy_0 = Policy.from_checkpoint(checkpoint)
        # restored_policy_0_weights = restored_policy_0[policy_name].get_weights()
        # algorithm.set_weights({policy_name: restored_policy_0_weights})
   
    frames = visualize(algorithm, num_episodes=args.num_episodes,teams=args.teams, training_scheme=args.training_scheme, num_agents=args.num_agents)
    if args.gif:
        import imageio
        filename = args.gif if args.gif.endswith('.gif') else f'{args.gif}.gif'
        print(f"Saving GIF to {filename}")

        # Define your desired frames per second
        fps = 60  # or any other number

        # Calculate the duration for each frame to achieve the desired fps
        duration = 1.0 / fps

        # write to file
        imageio.mimsave(filename, frames, duration=duration)