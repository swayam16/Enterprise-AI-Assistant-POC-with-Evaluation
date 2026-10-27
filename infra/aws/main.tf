# POC deployment on AWS: ECS Fargate (API + one-off init task) behind an ALB, RDS PostgreSQL 16
# with pgvector, EFS for model artifacts, Secrets Manager for credentials, CloudWatch logs.
# Uses the account's default VPC to keep the POC small; see README for the production delta.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = { Project = var.project, Environment = "poc" } }
}

data "aws_vpc" "default" { default = true }

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  name    = var.project
  subnets = data.aws_subnets.default.ids
  image   = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
}

# --- Container registry ------------------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
}

# --- Networking --------------------------------------------------------------------------------

resource "aws_security_group" "alb" {
  name   = "${local.name}-alb"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name   = "${local.name}-app"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "db" {
  name   = "${local.name}-db"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
}

resource "aws_security_group" "efs" {
  name   = "${local.name}-efs"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
}

# --- PostgreSQL 16 with pgvector ----------------------------------------------------------------

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_db_subnet_group" "db" {
  name       = local.name
  subnet_ids = local.subnets
}

resource "aws_db_instance" "db" {
  identifier              = local.name
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.db_instance_class
  allocated_storage       = 20
  storage_encrypted       = true
  db_name                 = "assistant"
  username                = "assistant"
  password                = random_password.db.result
  db_subnet_group_name    = aws_db_subnet_group.db.name
  vpc_security_group_ids  = [aws_security_group.db.id]
  publicly_accessible     = false
  backup_retention_period = 1
  skip_final_snapshot     = true
  # `CREATE EXTENSION vector` is run by the init task (pgvector ships with RDS PostgreSQL 16).
}

resource "aws_secretsmanager_secret" "app" {
  name                    = "${local.name}/app"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({
    DATABASE_URL      = "postgresql://assistant:${random_password.db.result}@${aws_db_instance.db.address}:5432/assistant?sslmode=require"
    ANTHROPIC_API_KEY = var.anthropic_api_key
  })
}

# --- Shared artifacts (embedder, reranker, thresholds) on EFS -----------------------------------

resource "aws_efs_file_system" "data" {
  encrypted = true
  tags      = { Name = "${local.name}-data" }
}

resource "aws_efs_mount_target" "data" {
  for_each        = toset(local.subnets)
  file_system_id  = aws_efs_file_system.data.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "data" {
  file_system_id = aws_efs_file_system.data.id
  posix_user {
    uid = 10001
    gid = 10001
  }
  root_directory {
    path = "/assistant"
    creation_info {
      owner_uid   = 10001
      owner_gid   = 10001
      permissions = "750"
    }
  }
}

# --- IAM -----------------------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.app.arn]
    }]
  })
}

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy" "task_efs" {
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"]
      Resource = [aws_efs_file_system.data.arn]
    }]
  })
}

# --- ECS -----------------------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = 14
}

locals {
  secrets = [
    { name = "DATABASE_URL", valueFrom = "${aws_secretsmanager_secret.app.arn}:DATABASE_URL::" },
    { name = "ANTHROPIC_API_KEY", valueFrom = "${aws_secretsmanager_secret.app.arn}:ANTHROPIC_API_KEY::" },
  ]
  environment = [
    { name = "LLM_PROVIDER", value = var.llm_provider },
    { name = "ANTHROPIC_MODEL", value = var.anthropic_model },
    { name = "ASSISTANT_DATA_DIR", value = "/app/data" },
  ]
  volume_mount = [{ sourceVolume = "data", containerPath = "/app/data" }]
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  volume {
    name = "data"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.data.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.data.id
        iam             = "ENABLED"
      }
    }
  }
  container_definitions = jsonencode([{
    name         = "api"
    image        = local.image
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment  = local.environment
    secrets      = local.secrets
    mountPoints  = local.volume_mount
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

# One-off: fetch + index docs, seed CRM, train reranker, tune thresholds. Run it with the
# command in the `init_command` output before the first deploy, and after a corpus change.
resource "aws_ecs_task_definition" "init" {
  family                   = "${local.name}-init"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  volume {
    name = "data"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.data.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.data.id
        iam             = "ENABLED"
      }
    }
  }
  container_definitions = jsonencode([{
    name        = "init"
    image       = local.image
    essential   = true
    command     = ["sh", "-c", "assistant ingest && assistant seed && assistant train && assistant tune"]
    environment = local.environment
    secrets     = local.secrets
    mountPoints = local.volume_mount
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "init"
      }
    }
  }])
}

# --- Load balancer + service ----------------------------------------------------------------------

resource "aws_lb" "main" {
  name               = local.name
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = local.subnets
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name}-api"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = data.aws_vpc.default.id
  health_check {
    path                = "/healthz"
    healthy_threshold   = 2
    unhealthy_threshold = 5
    interval            = 15
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = local.subnets
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true # default VPC has no NAT; needed to pull from ECR and reach the LLM API
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
  depends_on = [aws_lb_listener.http, aws_efs_mount_target.data]
}
