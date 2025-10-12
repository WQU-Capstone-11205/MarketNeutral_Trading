import torch
import numpy as np
import os, json

def save_models(save_dir, actor, critic, vae_encoder,
                    actor_opt, critic_opt, vae_opt,
                    bocpd_cfg, meta, step=None):
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
    print(f"Saved all models + optimizers at step {step}")


def load_models(load_dir, actor, critic, vae_encoder,
                    actor_opt, critic_opt, vae_opt,
                    device, step=None):
    tag = f"_step{step}" if step is not None else ""
    actor.load_state_dict(torch.load(os.path.join(load_dir, f"actor{tag}.pth"), map_location=device))
    critic.load_state_dict(torch.load(os.path.join(load_dir, f"critic{tag}.pth"), map_location=device))
    vae_encoder.load_state_dict(torch.load(os.path.join(load_dir, f"vae{tag}.pth"), map_location=device))
    actor_opt.load_state_dict(torch.load(os.path.join(load_dir, f"actor_opt{tag}.pth"), map_location=device))
    critic_opt.load_state_dict(torch.load(os.path.join(load_dir, f"critic_opt{tag}.pth"), map_location=device))
    vae_opt.load_state_dict(torch.load(os.path.join(load_dir, f"vae_opt{tag}.pth"), map_location=device))

    with open(os.path.join(load_dir, f"bocpd_cfg{tag}.json")) as f: bocpd_cfg = json.load(f)
    with open(os.path.join(load_dir, f"meta{tag}.json")) as f: meta = json.load(f)
    print(f"Loaded models/opts at step {step}")

def save_lstm_models(save_dir, lstm, vae_encoder,
                    lstm_opt, vae_opt,
                    bocpd_cfg, meta, step=None):
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_step{step}" if step is not None else ""

    torch.save(lstm.state_dict(), os.path.join(save_dir, f"lstm{tag}.pth"))
    torch.save(vae_encoder.state_dict(), os.path.join(save_dir, f"vae{tag}.pth"))
    torch.save(lstm_opt.state_dict(), os.path.join(save_dir, f"lstm_opt{tag}.pth"))
    torch.save(vae_opt.state_dict(), os.path.join(save_dir, f"vae_opt{tag}.pth"))
    with open(os.path.join(save_dir, f"bocpd_cfg{tag}.json"), "w") as f: json.dump(bocpd_cfg, f)
    with open(os.path.join(save_dir, f"meta{tag}.json"), "w") as f: json.dump(meta, f)
    print(f"Saved all models + optimizers at step {step}")


def load_lstm_models(load_dir, lstm, vae_encoder,
                    lstm_opt, vae_opt,
                    device, step=None):
    tag = f"_step{step}" if step is not None else ""
    lstm.load_state_dict(torch.load(os.path.join(load_dir, f"lstm{tag}.pth"), map_location=device))
    vae_encoder.load_state_dict(torch.load(os.path.join(load_dir, f"vae{tag}.pth"), map_location=device))
    lstm_opt.load_state_dict(torch.load(os.path.join(load_dir, f"lstm_opt{tag}.pth"), map_location=device))
    vae_opt.load_state_dict(torch.load(os.path.join(load_dir, f"vae_opt{tag}.pth"), map_location=device))

    with open(os.path.join(load_dir, f"bocpd_cfg{tag}.json")) as f: bocpd_cfg = json.load(f)
    with open(os.path.join(load_dir, f"meta{tag}.json")) as f: meta = json.load(f)
    print(f"Loaded models/opts at step {step}")
    return bocpd_cfg, meta

    return bocpd_cfg, meta
