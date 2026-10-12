# BPX Quadruped Reinforcement Learning on GCP with Isaac Sim

This repository contains Infrastructure-as-Code (Terraform), automation scripts, and an end-to-end guide to train reinforcement learning locomotion policies for the [MirrorMe BPX Quadruped Robot](https://github.com/mirrormerobotics) remotely on **Google Cloud Platform (GCP)** and visualize the results locally. It sets up a remote VM with an Nvidia L4, clone and executes the [MirrorMe BPX RL Training Baseline](https://github.com/mirrormerobotics/mirrorme_rl_train). You do not need this repo if you already have a machine to run training; You can use the MirrorMe repo directly.

---

## Architecture Overview

```
 ┌──────────────────────┐                                       ┌──────────────────────────────────────────────┐
 │   Local Client       │  ◄── SSH Tunnel: Port 6006 ─────────► │         GCP Compute Engine (G2)              │
 │  (MacBook Pro)       │      (TensorBoard Metrics)            │  • Machine: g2-standard-8 (8 vCPU, 32GB RAM) │
 │                      │                                       │  • GPU: 1x NVIDIA L4 (24GB VRAM)             │
 │ • Terminal / Browser │  ◄── Direct WebRTC / UDP: 47998 ────► │  • Ubuntu 22.04 LTS (NVIDIA Driver 550+)     │
 │ • Streaming Client   │      (Interactive 3D Video & Control) │  • Isaac Sim / Isaac Lab 2.3.0 + RSL-RL      │
 └──────────────────────┘                                       └──────────────────────────────────────────────┘
```

---

## Directory Structure

```text
bpx_rl/
├── terraform/                   # GCP Infrastructure as Code
│   ├── main.tf                  # Compute instance, firewall, IAM
│   ├── variables.tf             # Configurable input variables
│   ├── outputs.tf               # IP addresses and SSH helper commands
│   └── terraform.tfvars.example # Example configuration
├── scripts/
│   ├── setup_remote_vm.sh       # Automated remote environment setup
│   ├── setup_bpx_repo.sh        # Submodules & BPX task contract check
│   └── start_stream.sh          # WebRTC livestream launcher
└── README.md                    # Step-by-step guide
```

---

## Step-by-Step Guide

### Phase 1: Provision Cloud GPU VM (from Local Mac)

1. **Install Terraform** (if not already installed):
   ```bash
   brew tap hashicorp/tap
   brew install hashicorp/tap/terraform
   ```

2. **Configure GCP Variables**:
   Navigate to the `terraform/` folder and copy the example variables:
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   ```
   Edit `terraform.tfvars` with your GCP Project ID:
   ```hcl
   project_id = "your-gcp-project-id"
   use_spot   = true  # Optional: Set to true for ~70% cost reduction (~$0.25/hr vs $0.85/hr)
   ```

3. **IAM Permissions**:
   If you are using an account that isn't admin, you will need to grant the following permissions to this account before the terraform will succeed (and destroy cleanly):
   ```
   Compute Instance Admin (v1)
   Compute Network Admin (required to create custom VPCs and subnets)
   Compute Security Admin
   Create Service Accounts
   Delete Service Accounts (or Service Account Admin)
   IAP-secured Tunnel User
   Project IAM Admin
   Service Account User
   Storage Object Admin
   ```

4. **Deploy Infrastructure**:
   ```bash
   terraform init
   terraform apply
   ```

   ```bash
   Apply complete! Resources: 3 added, 0 changed, 0 destroyed.

   Outputs:

   instance_external_ip = "xxx.xxx.xxx.xxx"
   instance_name = "isaac-sim-trainer"
   ssh_command = "ssh xxx.xxx.xxx.xxx"
   tensorboard_tunnel_command = "ssh -i ~/.ssh/google_compute_engine -N -L 6006:localhost:6006 xxx.xxx.xxx.xxx"
   webrtc_client_connection = {
     "server_ip"      = "xxx.xxx.xxx.xxx"
     "signaling_port" = 49100
     "streaming_port" = 47998
   }
   ```

   Take note of the output `instance_external_ip` and `instance_name`.

---

### Phase 2: Setup Remote Training Environment

1. **SSH into the Instance**:
   ```bash
   gcloud compute ssh [USER]@isaac-sim-trainer --tunnel-through-iap
   ```

2. **Run Host System Setup**:
   Clone your repository and prepare the Ubuntu VM host:
   ```bash
   git clone https://github.com/gkcng/bpx_rl.git
   cd bpx_rl
   bash scripts/setup_remote_vm.sh
   ```
   This script:
   - Installs OpenGL, Vulkan, and headless display dependencies (`libglu1-mesa`, `libegl1`, etc.).
   - Verifies NVIDIA drivers (and installs them if missing).
   - Installs Miniconda and configures `conda-forge` as default to avoid Anaconda TOS prompts.

   Reboot to make sure NVIDIA drivers are active in the kernel:
   ```bash
   sudo reboot
   ```

   *(After reconnecting, verify GPU with `nvidia-smi`)*.

3. **Setup BPX RL Repository & Isaac Lab Environment**:
   ```bash
   cd ~/bpx_rl
   bash scripts/setup_bpx_repo.sh
   ```
   This script:
   - Clones `mirrorme_rl_train` and initializes submodules (URDF & mesh assets).
   - Runs the official upstream `./bpx.sh setup bpx_train` which creates the `bpx_train` environment, installs Isaac Lab 2.3.0, PyTorch CUDA 12.8, RSL-RL, and verifies the 45-D Actor / 48-D Critic contract:
   ```text
   [PASS] Actor 45-D, Critic 48-D, Action 12-D runtime contract.
   ```

---

### Phase 3: Train the Locomotion Policy

On the remote VM, launch training inside a persistent `tmux` session:

```bash
tmux new -s train
conda activate bpx_train
cd ~/mirrorme_rl_train

# 1. Quick Smoke Test (verifies GPU simulation pipeline in under 1 minute)
./bpx.sh train \
  --task bpx_flat \
  --num_envs 64 \
  --max_iterations 20 \
  --run_name smoke_test \
  --headless

# 2. Full Baseline Run (trains flat-ground velocity tracking in ~20-30 minutes)
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```
*Press `Ctrl+B` then `D` to detach from tmux. You can safely disconnect your Mac.*
Reattach anytime with `tmux attach -t train`.

#### Monitor Training Live via TensorBoard
From your **Mac terminal**, establish the secure IAP tunnel (zero public port 22 needed):
```bash
gcloud compute ssh isaac-sim-trainer --tunnel-through-iap -- -N -L 6006:localhost:6006
```

On the **remote VM**:
```bash
tensorboard --logdir ~/mirrorme_rl_train/logs/bpx_flat --port 6006
```
Open [http://localhost:6006](http://localhost:6006) on your Mac to track rewards and training convergence.

---

### Phase 4: Visualize the Walking Robot on your Mac

You have two methods to view the results:

#### Method A: Interactive 3D Stream via Isaac Sim WebRTC Streaming Client (Recommended)

In Isaac Sim 5.x, WebRTC streaming is handled by the official **Isaac Sim WebRTC Streaming Client** desktop app for macOS:
- **Download for macOS**: [Isaac Sim WebRTC Streaming Client (NVIDIA Official Download)](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release)

> **How WebRTC works with the client**:
> - **Signaling Port**: **`49100` (TCP)** handles session handshake and SDP negotiation.
> - **Streaming / Media Port**: **`47998` (UDP)** handles low-latency 60 FPS video frames and user input events.
>
> Direct connection to the VM's public IP over UDP is required because standard SSH tunnels only forward TCP.

1. **Firewall Requirement**:
   Ensure WebRTC ports (`tcp:49100`, `udp:47998-48012`) are permitted in your GCP firewall. (The Terraform configuration in `terraform/main.tf` creates this rule automatically as `allow_webrtc`).

2. **Launch the player on the remote VM**:
   ```bash
   conda activate bpx_train
   cd ~/mirrorme_rl_train
   ./bpx.sh play --checkpoint logs/bpx_flat/<RUN_FOLDER>/model_2999.pt --livestream 1 --num_envs 1
   ```

3. **Connect from your Mac**:
   Open the **Isaac Sim WebRTC Streaming Client** app on your Mac:
   * **Server / IP**: Enter your VM's public IP address (`<EXTERNAL_IP>`)
   * **Signaling Port**: `49100`
   * **Streaming Port**: `47998`
   * Click **Connect**

   You will have full real-time interactive 60 FPS 3D control to orbit, pan, and zoom around the walking BPX quadruped.

#### Method B: Render Headless Video (MP4)
1. On the **remote VM**:
   ```bash
   ./bpx.sh play --checkpoint logs/bpx_flat/<RUN_FOLDER>/model_2999.pt --headless --video --video_length 500
   ```
2. Download to your **Mac**:
   ```bash
   scp user@<EXTERNAL_IP>:~/mirrorme_rl_train/logs/bpx_flat/<RUN_FOLDER>/videos/*.mp4 ~/Downloads/
   ```
   Open the MP4 file in QuickTime.

---

### Phase 5: Policy Export & Deployment

Export the trained model to standard TorchScript and ONNX formats for physical robot deployment:
```bash
./bpx.sh export --checkpoint logs/bpx_flat/<RUN_FOLDER>/model_2999.pt
```
This produces:
- `exported/bpx_flat_45.pt` (TorchScript policy)
- `exported/bpx_flat_45.onnx` (ONNX policy for C++ runtime on robot)

---

### Phase 6: Clean Up / Manage VM Costs

- **Pause the VM** (stops compute and GPU charges, keeps SSD disk):
  ```bash
  gcloud compute instances stop isaac-sim-trainer --zone=us-central1-a
  ```
- **Destroy Infrastructure** when finished:
  ```bash
  cd terraform
  terraform destroy
  ```

---

## References

- [MirrorMeTech Robotics GitHub Organization](https://github.com/mirrormerobotics)
- [MirrorMe BPX RL Training Repo](https://github.com/mirrormerobotics/mirrorme_rl_train)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
