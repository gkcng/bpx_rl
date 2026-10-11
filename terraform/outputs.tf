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

output "livestream_tunnel_command" {
  description = "SSH tunnel command to forward WebRTC streaming and TensorBoard ports to your Mac."
  value       = "ssh -N -L 8011:localhost:8011 -L 8211:localhost:8211 -L 6006:localhost:6006 ${google_compute_instance.isaac_sim_node.network_interface[0].access_config[0].nat_ip}"
}
