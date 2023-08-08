#!/usr/bin/env python3
import configparser
import envs
import torch.nn.functional as F
import torch.optim as optim
import torch
import models.model as model
from lib import common, test_net, make_logger
import random
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import time
import gym
import ptan
from typing import List, Optional
from copy import deepcopy
from pathlib import Path
import pickle
import os
import pprint
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
GAMMA = 0.99
# https://www.reddit.com/r/reinforcementlearning/comments/ookni2/sac_critic_loss_increases/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-v", "--version", required=True,
                        type=str, help="config")
    parser.add_argument("-s", "--source", default=0, type=int,
                        help="whether load previous model and buffer")
    parser.add_argument("-r", "--seed", required=True,
                        type=int, help="random seed")
    # parser.add_argument("-s", "--load_config", ENVIRONMENT=None, help="whether load previous model and buffer")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "RL-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    current_lesson: int = 1
    version = "v" + args.version
    seed = args.seed

    # Set random seed
    torch.manual_seed(seed + current_lesson)
    np.random.seed(seed + current_lesson)
    random.seed(seed + current_lesson)

    model_folder = Path(save_path, version, "seed-" + str(seed))
    model_folder.mkdir(exist_ok=True, parents=True)
    save_model_folder = Path(model_folder, "save_models")
    save_model_folder.mkdir(exist_ok=True)

    logger = make_logger('trace', Path(model_folder, "rewards.log"))

    config = configparser.ConfigParser()
    config_path = Path("configs", args.name, version+".ini")
    print('source of config:', config_path)
    config.read(config_path)
    gym_id = config["ENVIRONMENT"]["gym_id"]
    # for key in config["ENVIRONMENT"]:
    #     print(f"{key} = {config['ENVIRONMENT'][key]}")
    for section in config.sections():
        print("-----", section, "-----")
        pprint.pprint(dict(config[section]))

    test_file_name = config["ENVIRONMENT"].get("test_file")

    # Create Environment
    env = gym.make(gym_id, configs=config["ENVIRONMENT"])
    test_env = gym.make(gym_id, configs=config["ENVIRONMENT"])

    # Model hyperparameters
    n_head = config["MODEL"].getint("n_head")
    hidden_size = config["MODEL"].getint("hidden_size")

    # Model hyperparameters
    TOTAL_FRAMES = config["TRAINING"].getfloat("total_frames")
    BATCH_SIZE = config["TRAINING"].getint("batch_size")
    REPLAY_SIZE = config["TRAINING"].getint("replay_size")
    REPLAY_INITIAL = config["TRAINING"].getint("replay_initial_size")
    SAC_ENTROPY_ALPHA = config["TRAINING"].getfloat("sac_entropy_alpha")
    LR_ACTS = config["TRAINING"].getfloat("learning_rate_actor")
    LR_VALS = config["TRAINING"].getfloat("learning_rate_vals")
    TEST_ITERS = config["TRAINING"].getint("test_iters")

    act_net = model.ModelActor(
        input_size=env.observation_space.shape[-1],
        act_size=env.action_space.shape[-1],
        hidden_size=hidden_size,
        n_head=n_head,
    ).to(device)

    crt_net = model.ModelCritic(
        input_size=env.observation_space.shape[-1],
        hidden_size=hidden_size,
        n_head=n_head,
    ).to(device)

    twinq_net = model.ModelSACTwinQ(
        input_size=env.observation_space.shape[-1],
        act_size=env.action_space.shape[-1],
        hidden_size=hidden_size,
        n_head=n_head,
    ).to(device)

    print(act_net)
    print(crt_net)
    print(twinq_net)

    # 如果有預設的modle, loading previous model
    # if privous_lesson > 0:
    #     previous_model_folder = Path(save_path, "lesson"+str(privous_lesson), version, "seed-" + str(seed))
    #     INITIAL_MODEL = Path(previous_model_folder, "best_model.pt")
    #     checkpoint = torch.load(INITIAL_MODEL, map_location = device)
    #     act_net.load_state_dict(checkpoint["act_net"])
    #     crt_net.load_state_dict(checkpoint["crt_net"])
    #     twinq_net.load_state_dict(checkpoint["twinq_net"])

    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(
        Path("runs", "RL-" + args.name, version, "seed-" + str(seed)))

    agent = model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=REPLAY_SIZE)

    # Load Buffer
    # if privous_lesson > 0:
    #     buffer_path = Path(previous_model_folder, "sac_replay_buffer.pkl")
    #     if buffer_path.exists():
    #         with open(buffer_path, "rb")as fs:
    #             d = pickle.load(fs)
    #         print("load buffer model:", buffer_path)
    #         buffer.buffer = deepcopy(d)
    #         print("buffer state", buffer.buffer[0].state.shape)
    #         print("buffer action", buffer.buffer[0].action.shape)
    #     else:
    #         print("no load buffer")

    act_opt = optim.Adam(act_net.parameters(), lr=LR_ACTS)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LR_VALS)
    twinq_opt = optim.Adam(twinq_net.parameters(), lr=LR_VALS)

    with open(f"data/validation/unper-{test_file_name}.pk", "rb")as fs:
        test_unper_file: List[np.ndarray] = pickle.load(fs)
    with open(f"data/validation/per-{test_file_name}.pk", "rb")as fs:
        test_per_file: List[np.ndarray] = pickle.load(fs)

    frame_idx = 0
    best_reward = None
    # try:
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
            writer, batch_size=10
        ) as tb_tracker:
            while frame_idx < TOTAL_FRAMES:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue
                print("frame", frame_idx)
                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, ref_vals_v, ref_q_v = \
                    common.unpack_batch_sac(
                        batch, tgt_crt_net.target_model,
                        twinq_net, act_net, GAMMA,
                        SAC_ENTROPY_ALPHA, device)

                tb_tracker.track("ref_v", ref_vals_v.mean(), frame_idx)
                tb_tracker.track("ref_q", ref_q_v.mean(), frame_idx)

                # train TwinQ
                twinq_opt.zero_grad()
                q1_v, q2_v = twinq_net(states_v, actions_v)
                q1_loss_v = F.mse_loss(q1_v.squeeze(),
                                       ref_q_v.detach())
                q2_loss_v = F.mse_loss(q2_v.squeeze(),
                                       ref_q_v.detach())
                q_loss_v = q1_loss_v + q2_loss_v
                q_loss_v.backward()
                twinq_opt.step()
                tb_tracker.track("loss_q1", q1_loss_v, frame_idx)
                tb_tracker.track("loss_q2", q2_loss_v, frame_idx)

                # Critic
                crt_opt.zero_grad()
                val_v = crt_net(states_v)
                v_loss_v = F.mse_loss(val_v.squeeze(),
                                      ref_vals_v.detach())

                v_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_v", v_loss_v, frame_idx)

                # Actor
                act_opt.zero_grad()
                acts_v = act_net(states_v)
                q_out_v, _ = twinq_net(states_v, acts_v)

                act_loss = -q_out_v.mean()
                act_loss.backward()

                act_opt.step()
                tb_tracker.track("loss_act", act_loss, frame_idx)

                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    print("Testing...")
                    # Test perturbation molecules
                    ts = time.time()
                    rewards_per, steps_per = test_net(
                        act_net, test_env, test_per_file, device=device)
                    logger.info(
                        f"rewards_per Test done in {time.time() - ts: .2f} sec, reward {rewards_per:.3f}, steps {steps_per}")
                    print("Test done in %.2f sec, rewards_per %.3f, steps_per %d" % (
                        time.time() - ts, rewards_per, steps_per))
                    writer.add_scalar("test_per_reward",
                                      rewards_per, frame_idx)
                    writer.add_scalar("test_per_steps", steps_per, frame_idx)

                    # Test unperturbation molecules
                    ts = time.time()
                    rewards_unper, steps_unper = test_net(
                        act_net, test_env, test_unper_file, device=device)
                    logger.info(
                        f"rewards_unper Test done in {time.time() - ts: .2f} sec, reward {rewards_unper:.3f}, steps {steps_unper}")
                    print("Test done in %.2f sec, rewards_unper %.3f, steps_unper %d" % (
                        time.time() - ts, rewards_unper, steps_unper))
                    writer.add_scalar("test_unper_reward",
                                      rewards_unper, frame_idx)
                    writer.add_scalar("test_unper_steps",
                                      steps_unper, frame_idx)

                    rewards = round((rewards_per + rewards_unper) / 2, 3)

                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" %
                                  (best_reward, rewards))
                            name = "best_%+.3f_%d.pt" % (rewards, frame_idx)
                            # fname = os.path.join(models_folder, name)
                            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-multiple-models-in-one-file
                            # torch.save(act_net.state_dict(), fname)
                            torch.save({
                                'act_net': act_net.state_dict(),
                                'crt_net': crt_net.state_dict(),
                                'twinq_net': twinq_net.state_dict(),
                            }, Path(save_model_folder, name))

                            torch.save({
                                'act_net': act_net.state_dict(),
                                'crt_net': crt_net.state_dict(),
                                'twinq_net': twinq_net.state_dict(),
                            }, Path(model_folder, "best_model.pt"))

                        best_reward = rewards
                # raise Exception

            else:
                # Dumps Buffer
                with open(Path(model_folder, "sac_replay_buffer.pkl"), 'wb')as fs:
                    pickle.dump(buffer.buffer, fs)
