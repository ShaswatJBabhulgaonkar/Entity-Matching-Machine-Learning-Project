output "repository" {
  description = "The repository"
  value       = try(aws_ecr_repository.repository, null)
}