output "instance_name" {
  description = "The name of the Compute Engine instance."
  value       = google_compute_instance.isaac_sim_node.name
}

output "instance_external_ip" {
  description = "Public IP address of the training instance."
  value       = google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  description = "Convenience SSH command to connect to the VM."
  value       = "ssh ${google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip}"
}

output "tensorboard_tunnel_command" {
  description = "SSH tunnel command to forward TensorBoard port (6006) to your Mac."
  value       = "ssh -i ~/.ssh/google_compute_engine -N -L 6006:localhost:6006 ${google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip}"
}

output "webrtc_client_connection" {
  description = "Connection settings for the Isaac Sim WebRTC Streaming Client desktop app."
  value = {
    server_ip      = google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip
    signaling_port = 49100
    streaming_port = 47998
  }
}
