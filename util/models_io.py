import torch
import numpy as np
import os, json

def save_RLmodels(save_dir, actor, critic, vae_encoder,
                    actor_opt, critic_opt, vae_opt,
                    bocpd_cfg, meta, step=None):
    """
    Save RL models for BOCPD+VAE+RL hybrid model

    Input:
      save_dir: saving directory name
      actor: actor model to be saved
      critic: critic model to be saved
      vae_encoder: vae model to be saved
      actor_opt: actor optimizer to be saved
      critic_opt: critic optimizer to be saved
      vae_opt: VAE optimizer to be saved
      bocpd_cfg: Configuration meta file for BOCPD
      meta: meta file to store the log details of the save model state

    Exit:
      Save all the models and their optimizer to the save_dir folder
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_step{step}" if step is not None else ""

    torch.save(actor.state_dict(), os.path.join(save_dir, f"actor{tag}.pth"))
    torch.save(critic.state_dict(), os.path.join(save_dir, f"critic{tag}.pth"))
    torch.save(vae_encoder.state_dict(), os.path.join(save_dir, f"vae{tag}.pth"))
    torch.save(actor_opt.state_dict(), os.path.join(save_dir, f"actor_opt{tag}.pth"))
    torch.save(critic_opt.state_dict(), os.path.join(save_dir, f"critic_opt{tag}.pth"))
    torch.save(vae_opt.state_dict(), os.path.join(save_dir, f"vae_opt{tag}.pth"))
    with open(os.path.join(save_dir, f"bocpd_cfg{tag}.json"), "w") as f: json.dump(bocpd_cfg, f)
    with open(os.path.join(save_dir, f"meta{tag}.json"), "w") as f: json.dump(meta, f)
    if step == None:
        print(f"Saved all models + optimizers")
    else:
        print(f"Saved all models + optimizers at step {step}")


def load_RLmodels(load_dir, actor, critic, vae_encoder,
                    actor_opt, critic_opt, vae_opt,
                    device, step=None):
    """
    Load RL models for BOCPD+VAE+RL hybrid model

    Input:
      load_dir: loading directory name
      actor: actor model to be loaded
      critic: critic model to be loaded
      vae_encoder: vae model to be loaded
      actor_opt: actor optimizer to be loaded
      critic_opt: critic optimizer to be loaded
      vae_opt: VAE optimizer to be loaded
      bocpd_cfg: Configuration meta file for BOCPD
      meta: meta file to store the log details of the save model state
    
    Exit:
      Load all the models and their optimizer from the load_dir folder
    """
    tag = f"_step{step}" if step is not None else ""
    actor.load_state_dict(torch.load(os.path.join(load_dir, f"actor{tag}.pth"), map_location=device))
    critic.load_state_dict(torch.load(os.path.join(load_dir, f"critic{tag}.pth"), map_location=device))
    vae_encoder.load_state_dict(torch.load(os.path.join(load_dir, f"vae{tag}.pth"), map_location=device))
    actor_opt.load_state_dict(torch.load(os.path.join(load_dir, f"actor_opt{tag}.pth"), map_location=device))
    critic_opt.load_state_dict(torch.load(os.path.join(load_dir, f"critic_opt{tag}.pth"), map_location=device))
    vae_opt.load_state_dict(torch.load(os.path.join(load_dir, f"vae_opt{tag}.pth"), map_location=device))

    with open(os.path.join(load_dir, f"bocpd_cfg{tag}.json")) as f: bocpd_cfg = json.load(f)
    with open(os.path.join(load_dir, f"meta{tag}.json")) as f: meta = json.load(f)
    if step == None:
        print(f"Loaded models/opts")
    else:
        print(f"Loaded models/opts at step {step}")
    return bocpd_cfg, meta

def save_models(save_dir, model, vae_encoder,
                    model_opt, vae_opt,
                    bocpd_cfg, meta, step=None):
    """
    Save models for all hybrid model except for BOCPD+VAE+RL

    Input:
      save_dir: saving directory name
      vae_encoder: vae model to be saved
      model_opt: model optimizer to be saved
      vae_opt: VAE to be saved
      bocpd_cfg: Configuration meta file for BOCPD
      meta: meta file to store the log details of the save model state

    Exit:
      Save all the models and their optimizer to the save_dir folder
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_step{step}" if step is not None else ""

    torch.save(model.state_dict(), os.path.join(save_dir, f"model{tag}.pth"))
    torch.save(vae_encoder.state_dict(), os.path.join(save_dir, f"vae{tag}.pth"))
    torch.save(model_opt.state_dict(), os.path.join(save_dir, f"model_opt{tag}.pth"))
    torch.save(vae_opt.state_dict(), os.path.join(save_dir, f"vae_opt{tag}.pth"))
    with open(os.path.join(save_dir, f"bocpd_cfg{tag}.json"), "w") as f: json.dump(bocpd_cfg, f)
    with open(os.path.join(save_dir, f"meta{tag}.json"), "w") as f: json.dump(meta, f)
    if step == None:
        print(f"Saved all models + optimizers")
    else:
        print(f"Saved all models + optimizers at step {step}")


def load_models(load_dir, model, vae_encoder,
                    model_opt, vae_opt,
                    device, step=None):
    """
    Load models for all hybrid models except BOCPD+VAE+RL

    Input:
      load_dir: loading directory name
      model: model to be loaded
      vae_encoder: vae model to be loaded
      model_opt: model optimizer to be loaded
      vae_opt: VAE optimizer to be loaded
    
    Exit:
      Load all the models and their optimizer from the load_dir folder
    """

    tag = f"_step{step}" if step is not None else ""
    model.load_state_dict(torch.load(os.path.join(load_dir, f"model{tag}.pth"), map_location=device))
    vae_encoder.load_state_dict(torch.load(os.path.join(load_dir, f"vae{tag}.pth"), map_location=device))
    model_opt.load_state_dict(torch.load(os.path.join(load_dir, f"model_opt{tag}.pth"), map_location=device))
    vae_opt.load_state_dict(torch.load(os.path.join(load_dir, f"vae_opt{tag}.pth"), map_location=device))

    with open(os.path.join(load_dir, f"bocpd_cfg{tag}.json")) as f: bocpd_cfg = json.load(f)
    with open(os.path.join(load_dir, f"meta{tag}.json")) as f: meta = json.load(f)
    if step == None:
        print(f"Loaded models/opts")
    else:
        print(f"Loaded models/opts at step {step}")
    return bocpd_cfg, meta
