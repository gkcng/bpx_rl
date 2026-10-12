terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ==============================================================================
# Networking: Dedicated Custom VPC & Subnet
# ==============================================================================
resource "google_compute_network" "training_vpc" {
  name                    = var.vpc_name
  auto_create_subnetworks = false
  description             = "Dedicated VPC network for Isaac Sim GPU training"
}

resource "google_compute_subnetwork" "training_subnet" {
  name                     = "${var.vpc_name}-subnet"
  ip_cidr_range            = var.subnet_cidr
  region                   = var.region
  network                  = google_compute_network.training_vpc.id
  private_ip_google_access = true
  description              = "Dedicated subnet for Isaac Sim training instances"
}

# ==============================================================================
# IAM: Service Account following Principle of Least Privilege
# ==============================================================================
resource "google_service_account" "isaac_sim_sa" {
  account_id   = "isaac-sim-runner"
  display_name = "Isaac Sim Training VM Service Account"
}

resource "google_project_iam_member" "logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.isaac_sim_sa.email}"
}

resource "google_project_iam_member" "monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.isaac_sim_sa.email}"
}

# ==============================================================================
# Firewall Rules: Zero public port 22, IAP SSH only, and restricted WebRTC
# ==============================================================================

# Allow SSH strictly via Google Cloud Identity-Aware Proxy (IAP)
# Port 22 is NEVER exposed to 0.0.0.0/0
resource "google_compute_firewall" "allow_iap_ssh" {
  name        = "${var.vpc_name}-allow-iap-ssh"
  network     = google_compute_network.training_vpc.name
  description = "Allow SSH ingress strictly via GCP Identity-Aware Proxy (IAP)"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # Official GCP IAP ingress CIDR range
  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["isaac-sim-node"]
}

# Allow WebRTC streaming strictly from whitelisted client IP addresses (var.allowed_client_ips)
resource "google_compute_firewall" "allow_webrtc" {
  name        = "${var.vpc_name}-allow-webrtc"
  network     = google_compute_network.training_vpc.name
  description = "Allow WebRTC streaming strictly from whitelisted client IP addresses"

  allow {
    protocol = "tcp"
    ports    = ["49100"]
  }

  allow {
    protocol = "udp"
    ports    = ["47998-48012"]
  }

  # Whitelist strictly your IP addresses from terraform.tfvars
  source_ranges = var.allowed_client_ips
  target_tags   = ["isaac-sim-node"]
}

# ==============================================================================
# Compute Engine GPU Instance
# ==============================================================================
resource "google_compute_instance" "isaac_sim_node" {
  name                      = var.instance_name
  machine_type              = var.machine_type
  zone                      = var.zone
  tags                      = ["isaac-sim-node"]
  allow_stopping_for_update = true

  boot_disk {
    auto_delete = true
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.boot_disk_size_gb
      type  = "pd-ssd"
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    preemptible         = var.use_spot
    provisioning_model  = var.use_spot ? "SPOT" : "STANDARD"
    automatic_restart   = var.use_spot ? false : true
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  network_interface {
    network    = google_compute_network.training_vpc.name
    subnetwork = google_compute_subnetwork.training_subnet.name
    access_config {
      # Ephemeral public IP address for WebRTC UDP streaming
    }
  }

  service_account {
    email  = google_service_account.isaac_sim_sa.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    install-nvidia-driver = "true"
  }
}
