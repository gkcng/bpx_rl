variable "project_id" {
  description = "The GCP project ID to deploy resources into."
  type        = string
}

variable "region" {
  description = "The GCP region for deployment."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone for the compute instance with GPU capacity."
  type        = string
  default     = "us-central1-a"
}

variable "instance_name" {
  description = "Name of the compute engine instance."
  type        = string
  default     = "isaac-sim-trainer"
}

variable "machine_type" {
  description = "GCP machine type with NVIDIA L4 support."
  type        = string
  default     = "g2-standard-8" # 8 vCPUs, 32 GB RAM
}

variable "gpu_type" {
  description = "GPU accelerator type."
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach."
  type        = number
  default     = 1
}

variable "boot_disk_size_gb" {
  description = "Size of the root SSD disk in GB. Isaac Sim caches & Docker/Conda require at least 100-150 GB."
  type        = number
  default     = 150
}

variable "use_spot" {
  description = "Use Spot VM pricing for significant cost reduction (~60-70% savings)."
  type        = bool
  default     = false
}

variable "network" {
  description = "VPC network name."
  type        = string
  default     = "default"
}

variable "allowed_ssh_cidr" {
  description = "CIDR range allowed to SSH into the instance."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
