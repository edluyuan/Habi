import os
from copy import deepcopy
# os.environ['MUJOCO_GL'] = 'egl'

import d4rl
import gym
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn.functional import softplus

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, loop_two_dataloaders
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition, IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DAMlp, VAEDistillMLP
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters, NormalCritic, BasicCritic, DQLCritic, DAHorizonCritic, FreezeModules
from cleandiffuser.utils import JoshMLP
from cleandiffuser.nn_diffusion import DQLMlp
from utils import set_seed
from tqdm import tqdm
from omegaconf import OmegaConf
import pickle

from cleandiffuser.dataset.base_dataset import BaseDataset
class FakePlannerDataset(BaseDataset):
    def __init__(self, state_normalizer):
        super().__init__()

        self.normalizers = {
            "state": state_normalizer
        }
        self.size = 0
        self.append_planner = False

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        assert self.append_planner, "Planner's plan is not appended to horizon dataset"
        data = {
            'obs': {'state': self.obs[idx], },
            'planner_act': self.planner_act[idx],
            'planner_best_act': self.planner_best_act[idx],
            'plan_value': self.plan_value[idx],
            'plan_best_value': self.plan_best_value[idx],
        }

        return data

class Synergy(nn.Module):
    def __init__(self, obs_dim, act_dim, posterior_input_dim, latent_dim, conditional_decoder=False):
        super(Synergy, self).__init__()
        self.prior_mu_encoder = JoshMLP(
            dims=[obs_dim] + [latent_dim] * 3,
            activation=nn.Mish(),
            out_activation=nn.Identity(),
        )
        
        self.prior_aspstd_encoder = JoshMLP(
            dims=[obs_dim] + [latent_dim] * 3,
            activation=nn.Mish(),
            out_activation=nn.Identity(),
        )
        
        self.posterior_mu_encoder = JoshMLP(
            dims=[posterior_input_dim] + [latent_dim] * 3,
            activation=nn.Mish(),
            out_activation=nn.Identity(),
        )
        
        self.posterior_aspstd_encoder = JoshMLP(
            dims=[posterior_input_dim] + [latent_dim] * 3,
            activation=nn.Mish(),
            out_activation=nn.Identity(),
        )
        
        if conditional_decoder:
            self.z_to_action = JoshMLP(
                dims=[latent_dim + obs_dim] + [latent_dim] * 2 + [act_dim],
                activation=nn.Mish(),
                out_activation=nn.Identity(),
            )
        else:
            self.z_to_action = JoshMLP(
                dims=[latent_dim] + [latent_dim] * 2 + [act_dim],
                activation=nn.Mish(),
                out_activation=nn.Identity(),
            )

    def get_prior_mu_sigma(self, obs):
        prior_mu = self.prior_mu_encoder(obs)
        prior_aspstd = self.prior_aspstd_encoder(obs)
        return prior_mu, softplus(prior_aspstd) + 1e-2
    
    def get_posterior_mu_sigma(self, planner_inputs):
        posterior_mu = self.posterior_mu_encoder(planner_inputs)
        posterior_aspstd = self.posterior_aspstd_encoder(planner_inputs)
        return posterior_mu, softplus(posterior_aspstd) + 1e-2

@hydra.main(config_path="../configs/habi/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    args.pipeline_name = args.pipeline_name
    if args.enable_wandb:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(args, resolve=True)
        )

    set_seed(args.seed)
    
    dql_base_path = f'dql_d4rl_antmaze'
    
    # for distill policy
    base_path = dql_base_path.replace("dql", args.pipeline_name)
    base_path += f"_ETA{args.task.eta}"
    base_path += f"_recon{args.reconstruct_eta}"
    base_path += f"_kl_eta{args.kl_eta}"
    base_path += f"_fixedz{args.deterministic_latent}"
    base_path += f"_dimz{args.latent_dim}"
    base_path += f"_qtype{args.planner_input_type}"
    base_path += f"_learnkleta{args.learn_kl_eta}"
    base_path += f"_seed{args.seed}"
    base_path += f"_randimitate{args.use_random_imitation}"
    base_path += f"_datasettype{args.dataset_type}"
    base_path += f"_uncondratio{args.uncondition_sample_ratio}"
    base_path += f"_modeltype{args.model_type}"
    base_path += f"_conddecoder{args.conditional_decoder}"
    base_path += f"_lr{args.policy_learning_rate}"
    
    dql_save_path = f"{args.save_dir}/" + dql_base_path + f"/{args.task.env_name}/"
    save_path = f"{args.save_dir}/" + base_path + f"/{args.task.env_name}/"
    
    if os.path.exists(dql_base_path) is False:
        os.makedirs(dql_base_path)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    
    from cleandiffuser.utils import GaussianNormalizer
    
    raw_dataset = env.get_dataset()
    # state normalizer
    state_normalizer = GaussianNormalizer(raw_dataset["observations"].astype(np.float32))
    
    # td_dataset
    planner_td_dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env), state_normalizer=state_normalizer)
    
    # fake planner dataset
    fake_planner_dataset = FakePlannerDataset(state_normalizer)
    
    # dims
    obs_dim, act_dim = planner_td_dataset.o_dim, planner_td_dataset.a_dim
    
    # --------------- Prior and VAE encoders -------------------
    bb_model = Synergy(obs_dim, act_dim, obs_dim + act_dim, args.latent_dim, args.conditional_decoder).to(args.device)
    bb_optimizer = torch.optim.Adam(bb_model.parameters(), lr=args.policy_learning_rate)

    # ----------------- Masking -------------------

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)
    
    dql_actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.dql_predict_noise, optim_params={"lr": args.policy_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.dql_diffusion_steps, ema_rate=args.dql_ema_rate, device=args.device)
    
    # ------------------ Critic ---------------------
    dql_critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.dql_hidden_dim).to(args.device)
    dql_critic_target = deepcopy(dql_critic).requires_grad_(False).eval()
    
    # ------------------ Critic ---------------------
    habi_critic = NormalCritic(args.latent_dim, act_dim, hidden_dim=args.latent_dim).to(args.device)
    critic_optim = torch.optim.Adam(habi_critic.parameters(), lr=args.critic_learning_rate)
    
    # ------------------ learn kl_eta ---------------------
    if args.learn_kl_eta:
        log_kl_eta = nn.Parameter(torch.tensor(np.log(1e-6).astype(np.float32), requires_grad=True).to(device=args.device))
        kl_eta_optim = torch.optim.Adam([log_kl_eta], lr=3e-4)
    
    def load_planner():
        dql_actor.load(dql_save_path + f"diffusion_ckpt_{args.dql_ckpt}.pt")
        dql_critic_ckpt = torch.load(dql_save_path + f"critic_ckpt_{args.dql_ckpt}.pt")
        dql_critic.load_state_dict(dql_critic_ckpt["critic"])
        dql_critic_target.load_state_dict(dql_critic_ckpt["critic_target"])

        dql_actor.eval()
        dql_critic.eval()
        dql_critic_target.eval()
    
    # ---------------------- Training ----------------------
    if args.mode == "generate_goal_direct_behavior":
        
        assert args.dataset_type == "td", "Only support td dataset for now"
        
        load_planner()        
        # --------------- Append Planner's Plan to Original Dataset -------------------
        
        def append_planner_plan(obs):
            B = obs.shape[0]
            prior = torch.zeros((B * args.planner_num_candidates, act_dim), device=args.device)
            obs = torch.tensor(obs, device=args.device, dtype=torch.float32)
            obs_repeat_reshape = obs.unsqueeze(1).repeat(1, args.planner_num_candidates, 1)
            obs_repeat = obs_repeat_reshape.view(-1, obs_dim)
            acts, log = dql_actor.sample(
                prior,
                solver=args.dql_solver,
                n_samples=B * args.planner_num_candidates,
                sample_steps=args.dql_sampling_steps,
                condition_cfg=obs_repeat, w_cfg=1.0,
                use_ema=args.dql_use_ema, temperature=args.dql_temperature)
            
            with torch.no_grad():
                q = dql_critic.q_min(obs_repeat, acts)
                q = q.view(B, args.planner_num_candidates)
                idx = torch.argmax(q, -1)

                acts = acts.view(B, args.planner_num_candidates, act_dim)
                best_act = acts[torch.arange(B), idx]
                best_act_repeat = best_act.unsqueeze(1).repeat(1, args.planner_num_candidates, 1)
                best_value = q[torch.arange(B), idx]
                best_value_repeat = best_value.unsqueeze(1).repeat(1, args.planner_num_candidates)
                        
            obs_data = obs_repeat_reshape[:, :args.uncondition_sample_ratio]
            act_data = acts[:, :args.uncondition_sample_ratio]
            best_act_data = best_act_repeat[:, :args.uncondition_sample_ratio]
            value_data = q[:, :args.uncondition_sample_ratio]
            best_value_data = best_value_repeat[:, :args.uncondition_sample_ratio]
            
            append_data = {
                "obs": obs_data.reshape(-1, obs_dim).cpu().numpy(),
                "planner_act": act_data.reshape(-1, act_dim).cpu().numpy(),
                "planner_best_act": best_act_data.reshape(-1, act_dim).cpu().numpy(),
                "value": value_data.reshape(-1, 1).cpu().numpy(),
                "best_value": best_value_data.reshape( -1, 1).cpu().numpy(),
            }
                        
            return append_data
           
        print("Appending Planner's plan to horizon dataset...")

        APPEND_BATCH_SIZE = 128
        num_transitions = planner_td_dataset.size
        
        fake_planner_dataset.size = num_transitions * args.uncondition_sample_ratio
        fake_planner_dataset.obs = np.zeros((num_transitions * args.uncondition_sample_ratio, obs_dim))
        fake_planner_dataset.planner_act = np.zeros((num_transitions * args.uncondition_sample_ratio, act_dim))
        fake_planner_dataset.planner_best_act = np.zeros((num_transitions * args.uncondition_sample_ratio, act_dim))
        fake_planner_dataset.plan_value = np.zeros((num_transitions * args.uncondition_sample_ratio, 1))
        fake_planner_dataset.plan_best_value = np.zeros((num_transitions * args.uncondition_sample_ratio, 1))
        
        append_num = 0
        for start_idx in tqdm(range(0, num_transitions, APPEND_BATCH_SIZE)):
            end_idx = min(start_idx + APPEND_BATCH_SIZE, num_transitions)
                    
            obs = planner_td_dataset.obs[start_idx: end_idx].cpu().numpy()
            
            # append planner's plan
            append_data = append_planner_plan(obs)

            # append planner's plan (obs)
            new_data_num = append_data["obs"].shape[0]
            fake_planner_dataset.obs[append_num: append_num+new_data_num] = append_data["obs"]
            fake_planner_dataset.planner_act[append_num: append_num+new_data_num] = append_data["planner_act"]
            fake_planner_dataset.planner_best_act[append_num: append_num+new_data_num] = append_data["planner_best_act"]
            fake_planner_dataset.plan_value[append_num: append_num+new_data_num] = append_data["value"]
            fake_planner_dataset.plan_best_value[append_num: append_num+new_data_num] = append_data["best_value"]
            
            append_num += new_data_num

        print("Appending Planner's plan to horizon dataset... Done")
        fake_planner_dataset.append_planner = True
        
        # save planner_td_dataset
        dataset_save_dir = os.path.join(dql_save_path, f"fake_planner_{args.dataset_type}_dataset_bb_x{args.uncondition_sample_ratio}")
        if not os.path.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        
        save_tasks = [
            ("obs", fake_planner_dataset.obs),
            ("planner_act", fake_planner_dataset.planner_act),
            ("planner_best_act", fake_planner_dataset.planner_best_act),
            ("plan_value", fake_planner_dataset.plan_value),
            ("plan_best_value", fake_planner_dataset.plan_best_value),
        ]

        # 使用 tqdm 显示进度条
        for filename, data in tqdm(save_tasks, desc="Saving datasets"):
            np.save(os.path.join(dataset_save_dir, filename), data)
            
        print("Planner's plan is saved to horizon dataset")
        print("Data Path is:", dataset_save_dir)
            
    elif args.mode == "train":

        dataset_save_dir = os.path.join(dql_save_path, f"fake_planner_{args.dataset_type}_dataset_bb_x{args.uncondition_sample_ratio}_dql")
        
        load_tasks = [
            "obs", 
            "planner_act", 
            "planner_best_act", 
            "plan_value", 
            "plan_best_value"
        ]
        for task in tqdm(load_tasks, desc="Loading datasets"):
            setattr(fake_planner_dataset, task, np.load(os.path.join(dataset_save_dir, f"{task}.npy")).astype(np.float32))
        fake_planner_dataset.size = fake_planner_dataset.obs.shape[0]
        print(f"Size of Planner's plan dataset: {fake_planner_dataset.size}")
        fake_planner_dataset.append_planner = True
        
        fake_planner_dataloader = DataLoader(
            fake_planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
        # --------------- Train  -------------------        
        print("Start training ...")
        
        habi_lr_scheduler = CosineAnnealingLR(bb_optimizer, T_max=args.gradient_steps)
        critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)

        bb_model.train()
        habi_critic.train()
        
        n_gradient_step = 0
        log = {
            "bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.,
            "dv_imitate_loss": 0., "kl_loss": 0.
        }
        
        pbar = tqdm(total=args.gradient_steps / args.log_interval)
        for td_batch in loop_dataloader(fake_planner_dataloader):
            
            obs = td_batch["obs"]["state"].to(args.device)
            planner_act = td_batch["planner_act"].to(args.device)
            planner_best_act = td_batch["planner_best_act"].to(args.device)
            plan_value = td_batch["plan_value"].to(args.device).reshape(-1, 1)
            with torch.no_grad():
                posterior_mu, posterior_sigma = bb_model.get_posterior_mu_sigma(torch.cat([obs, planner_act], dim=-1))
                dist_posterior = torch.distributions.Normal(posterior_mu, posterior_sigma)
                z_posterior = dist_posterior.rsample() 
            
            critic_loss = F.mse_loss(habi_critic(z_posterior.detach(), planner_act), plan_value)
            target_q = plan_value.detach()
            
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            
            # ---- Policy Training ----
            
            prior_mu, prior_sigma = bb_model.get_prior_mu_sigma(obs)
            if args.use_random_imitation:
                posterior_mu, posterior_sigma = bb_model.get_posterior_mu_sigma(torch.cat([obs, planner_act], dim=-1))
            else:
                posterior_mu, posterior_sigma = bb_model.get_posterior_mu_sigma(torch.cat([obs, planner_best_act], dim=-1))
            dist_posterior = torch.distributions.Normal(posterior_mu, posterior_sigma)
            z_posterior = dist_posterior.rsample()
            
            if np.random.rand() < 0.0002:
                size_sigma = prior_sigma.shape[0]
                sampled_idx = np.random.choice(size_sigma, 10)
                print("prior_sigma samples:", np.array2string(prior_sigma.detach().cpu().numpy().reshape([-1])[sampled_idx], precision=3, separator=","))
                print("postr_sigma samples:", np.array2string(posterior_sigma.detach().cpu().numpy().reshape([-1])[sampled_idx], precision=3, separator=","))
                print("----------------------------------------------------")
            
            if args.conditional_decoder:
                new_act = bb_model.z_to_action(torch.cat([z_posterior, obs], dim=-1))
            else:
                new_act = bb_model.z_to_action(z_posterior)
            
            # discrepency from DV action
            if args.use_random_imitation:
                dv_imitate_loss = (new_act - planner_act) ** 2
            else:
                dv_imitate_loss = (new_act - planner_best_act) ** 2
            
            if args.learn_kl_eta:
                kl_eta = torch.exp(log_kl_eta).detach()
            else:
                kl_eta = args.kl_eta
            
            kl_loss = (torch.log(prior_sigma) - torch.log(posterior_sigma) + (posterior_sigma ** 2 + (posterior_mu - prior_mu) ** 2) / (2 * prior_sigma ** 2) - 0.5).sum(-1, keepdim=True)

            with FreezeModules([habi_critic, ]):
                q_new_action = habi_critic(z_posterior.detach(), new_act)
                q_loss = - q_new_action.mean()
                
            actor_loss = args.task.eta * q_loss \
                + dv_imitate_loss \
                + kl_eta * kl_loss
            
            actor_loss = actor_loss.mean()
            
            bb_optimizer.zero_grad()
            actor_loss.backward()
            bb_optimizer.step()

            habi_lr_scheduler.step()
            critic_lr_scheduler.step()
            
            # -- eta Training
            if args.learn_kl_eta:
                loss_kl_eta = - torch.mean(log_kl_eta * (
                   torch.log10(torch.clamp(kl_loss.detach(), 1e-9, np.inf)) - np.log10(float(args.kld_target))).detach())
                
                kl_eta_optim.zero_grad()
                loss_kl_eta.backward()
                kl_eta_optim.step()
                
            # # ----------- Logging ------------
            log["dv_imitate_loss"] += dv_imitate_loss.mean().item()
            log["kl_loss"] += kl_loss.mean().item()
            log["q_loss"] += q_loss.mean().item()
            log["critic_loss"] += critic_loss.item()
            log["target_q_mean"] += target_q.mean().item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["dv_imitate_loss"] /= args.log_interval
                log["kl_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["critic_loss"] /= args.log_interval
                log["target_q_mean"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                log = {
                    "bc_loss": 0., "q_loss": 0., "critic_loss": 0., "target_q_mean": 0.,
                    "dv_imitate_loss": 0., "kl_loss": 0.
                }

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                torch.save({
                    "bb_model": bb_model.state_dict(),
                }, save_path + f"bb_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "bb_model": bb_model.state_dict(),
                }, save_path + f"bb_ckpt_latest.pt")
                
                torch.save({
                    "habi_critic": habi_critic.state_dict(),
                }, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "habi_critic": habi_critic.state_dict(),
                }, save_path + f"critic_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    elif args.mode == "inference_prior" or args.mode == "inference":
        # --------------- Load Policy and Critic -------------------
        
        load_planner()
        
        bb_model_ckpt = torch.load(save_path + f"bb_ckpt_{args.bb_ckpt}.pt")
        bb_model.load_state_dict(bb_model_ckpt["bb_model"])
        
        bb_model.eval()
        
        habi_critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.bb_ckpt}.pt")
        habi_critic.load_state_dict(habi_critic_ckpt["habi_critic"])
        habi_critic.eval()
        
        # --------------- Test Policy -------------------
        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        episode_rewards = []
        normalizer = state_normalizer
        
        for i in range(args.num_episodes):
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
            
            while not np.all(cum_done) and t < args.task.max_path_length + 1:
                
                with torch.no_grad():
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    obs = obs.unsqueeze(1).repeat(1, args.bb_num_candidates, 1).view(-1, obs_dim)
                    prior_mu, prior_sigma = bb_model.get_prior_mu_sigma(obs)
                    dist = torch.distributions.Normal(prior_mu, prior_sigma)
                    
                    z_prior = dist.rsample()
                    
                    if args.conditional_decoder:
                        act = bb_model.z_to_action(torch.cat([z_prior, obs], dim=-1))
                    else:
                        act = bb_model.z_to_action(z_prior)

                    # select best action
                    if args.use_pretrained_dql_critic == 1:
                        q = dql_critic_target.q_min(obs, act)
                    elif args.use_pretrained_dql_critic == 2:
                        q = dql_critic.q_min(obs, act)
                    else:
                        q = habi_critic(z_prior, act)

                    q = q.view(args.num_envs, args.bb_num_candidates)
                    idx = torch.argmax(q, -1)
                        
                    act = act.reshape(args.num_envs, args.bb_num_candidates, act_dim)
                    sampled_act = act[torch.arange(args.num_envs), idx].clip(-1., 1.).cpu().numpy()
                
                obs, rew, done, info = env_eval.step(sampled_act)
                
                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                if t % 10 == 9:
                    print(f'[episode{i}, t={t}] rew: {ep_reward}')
            
            episode_rewards.append(np.clip(ep_reward, 0., 1.))
            
        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).reshape(-1) * 100
        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        print(mean, err)
        
        if args.enable_wandb:
            wandb.log({'Mean Reward': mean, 'Error': err})
            wandb.finish()
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    print("~~~~~~~~~~~~~~~~~~~~~~~~~ Start ~~~~~~~~~~~~~~~~~~~~~~~~~")
    pipeline()
