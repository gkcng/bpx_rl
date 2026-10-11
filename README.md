# BPX Quadruped Reinforcement Learning on GCP with Isaac Sim

This repository contains Infrastructure-as-Code (Terraform), automation scripts, and an end-to-end guide to train reinforcement learning locomotion policies for the [MirrorMe BPX Quadruped Robot](https://github.com/mirrormerobotics) remotely on **Google Cloud Platform (GCP)** and visualize the results locally on a Mac or browser.

---

## Architecture Overview

```
 ┌──────────────────────┐              SSH Tunnels              ┌──────────────────────────────────────────────┐
 │   Local Client       │  ◄──────────────────────────────────► │         GCP Compute Engine (G2)              │
 │  (MacBook Pro)       │     • Port 8011/8211 (WebRTC Stream)  │  • Machine: g2-standard-8 (8 vCPU, 32GB RAM) │
 │                      │     • Port 6006 (TensorBoard)         │  • GPU: 1x NVIDIA L4 (24GB VRAM)             │
 │ • Safari / Chrome    │                                       │  • Ubuntu 22.04 LTS (NVIDIA Driver 550+)     │
 │ • Terminal (SSH)     │                                       │  • Isaac Sim / Isaac Lab 2.3.0 + RSL-RL      │
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

3. **Deploy Infrastructure**:
   ```bash
   terraform init
   terraform apply
   ```
   Take note of the output `instance_external_ip`.

---

### Phase 2: Setup Remote Training Environment

1. **SSH into the Instance**:
   ```bash
   ssh user@<EXTERNAL_IP>
   ```

2. **Run Automated System Setup**:
   Upload or run the setup script on the remote VM:
   ```bash
   # Run the environment setup script
   curl -sSL https://raw.githubusercontent.com/.../setup_remote_vm.sh | bash
   # Or run directly if copied:
   bash scripts/setup_remote_vm.sh
   ```
   This script:
   - Installs OpenGL, Vulkan, and headless display dependencies.
   - Installs Miniconda and creates the `bpx_train` Python 3.11 environment.
   - Installs Isaac Lab 2.3.0, PyTorch 2.7.0 with CUDA 12.8, and RSL-RL.

3. **Setup BPX RL Repository**:
   ```bash
   conda activate bpx_train
   bash scripts/setup_bpx_repo.sh
   ```
   This clones `mirrorme_rl_train`, initializes BPX URDF/meshes, installs the package, and verifies the 45-D Actor / 48-D Critic contract:
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

# Train flat-ground velocity tracking (1024 parallel environments, 3000 iterations)
./bpx.sh train \
  --task bpx_flat \
  --num_envs 1024 \
  --max_iterations 3000 \
  --run_name baseline \
  --headless
```
*Press `Ctrl+B` then `D` to detach from tmux. You can safely disconnect your Mac.*

#### Monitor Training Live via TensorBoard
From your **Mac terminal**:
```bash
ssh -N -L 6006:localhost:6006 user@<EXTERNAL_IP>
```
On the **remote VM**:
```bash
tensorboard --logdir ~/mirrorme_rl_train/logs/bpx_flat --port 6006
```
Open [http://localhost:6006](http://localhost:6006) on your Mac to track rewards and training convergence.

---

### Phase 4: Visualize the Walking Robot on your Mac

You have two methods to view the results:

#### Method A: Live WebRTC 3D Viewport in your Browser (Recommended)
1. From your **Mac terminal**, establish port tunnels:
   ```bash
   ssh -N -L 8011:localhost:8011 -L 8211:localhost:8211 user@<EXTERNAL_IP>
   ```
2. On the **remote VM**, start the stream:
   ```bash
   conda activate bpx_train
   cd ~/mirrorme_rl_train
   ./bpx.sh play --checkpoint logs/bpx_flat/<RUN_FOLDER>/model_2999.pt --livestream 1 --num_envs 1
   ```
3. Open Chrome or Safari on your Mac and browse to:
   ```text
   http://localhost:8011/streaming/client/
   ```
   You will have full real-time interactive 3D control to orbit and zoom around the walking BPX quadruped.

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
