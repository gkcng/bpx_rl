output "instance_name" {
  description = "The name of the Compute Engine instance."
  value       = google_compute_instance.isaac_sim_node.name
}

output "instance_external_ip" {
  description = "Public IP address of the training instance."
  value       = google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip
}

output "ssh_command_direct" {
  description = "Direct SSH command using external IP (allowed when whitelisted)."
  value       = "ssh ${google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip}"
}

output "ssh_command_iap" {
  description = "Secure SSH command tunneling through GCP Identity-Aware Proxy (zero open public port 22)."
  value       = "gcloud compute ssh ${google_compute_instance.isaac_sim_node.name} --zone=${google_compute_instance.isaac_sim_node.zone} --tunnel-through-iap"
}

output "tensorboard_tunnel_command" {
  description = "SSH tunnel command to forward TensorBoard port (6006) to your Mac via IAP."
  value       = "gcloud compute ssh ${google_compute_instance.isaac_sim_node.name} --zone=${google_compute_instance.isaac_sim_node.zone} --tunnel-through-iap -- -N -L 6006:localhost:6006"
}

output "webrtc_client_connection" {
  description = "Connection settings for the Isaac Sim WebRTC Streaming Client desktop app."
  value = {
    server_ip      = google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip
    signaling_port = 49100
    streaming_port = 47998
  }
}
