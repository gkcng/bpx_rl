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

# Service Account following least privilege principle
resource "google_service_account" "isaac_sim_sa" {
  account_id   = "isaac-sim-runner"
  display_name = "Isaac Sim Training VM Service Account"
}

# Minimal role: Logging and Monitoring metrics writer
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

# Firewall rule for SSH ingress
resource "google_compute_firewall" "allow_ssh" {
  name    = "isaac-sim-allow-ssh"
  network = var.network

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allowed_ssh_cidr
  target_tags   = ["isaac-sim-node"]
  description   = "Allow SSH ingress to Isaac Sim training node"
}

# Compute Engine GPU Instance
resource "google_compute_instance" "isaac_sim_node" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["isaac-sim-node"]

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
    network = var.network
    access_config {
      # Ephemeral public IP address
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
