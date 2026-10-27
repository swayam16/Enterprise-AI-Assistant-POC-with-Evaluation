output "url" {
  value = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "init_command" {
  description = "Run once before first use (indexes the docs, seeds the CRM, trains the reranker)."
  value = join(" ", [
    "aws ecs run-task --cluster ${aws_ecs_cluster.main.name}",
    "--launch-type FARGATE --task-definition ${aws_ecs_task_definition.init.family}",
    "--network-configuration 'awsvpcConfiguration={subnets=[${join(",", local.subnets)}],securityGroups=[${aws_security_group.app.id}],assignPublicIp=ENABLED}'",
  ])
}
