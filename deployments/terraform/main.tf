# Terraform Infrastructure as Code for GAN Cyber Range Simulator
# Comprehensive cloud infrastructure deployment on AWS

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    # Configure S3 backend for state management
    # bucket = "your-terraform-state-bucket"
    # key    = "gan-cyber-range/terraform.tfstate"
    # region = "us-west-2"
    # encrypt = true
    # dynamodb_table = "terraform-locks"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "GAN-Cyber-Range-Simulator"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Repository  = "gan-cyber-range-sim"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
  
  # Security group rules for cyber range
  cyber_range_ingress_rules = [
    {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "HTTP access"
    },
    {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
      description = "HTTPS access"
    },
    {
      from_port   = 8080
      to_port     = 8090
      protocol    = "tcp"
      cidr_blocks = [var.vpc_cidr]
      description = "Cyber range services"
    }
  ]
}

# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-vpc"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-igw"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = var.az_count
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-${count.index + 1}"
    Type = "Public"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = var.az_count
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-${count.index + 1}"
    Type = "Private"
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb" = "1"
  })
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat" {
  count = var.az_count
  
  domain = "vpc"
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-eip-${count.index + 1}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# NAT Gateways
resource "aws_nat_gateway" "main" {
  count = var.az_count
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nat-${count.index + 1}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables - Public
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-public-rt"
  })
}

# Route Tables - Private
resource "aws_route_table" "private" {
  count = var.az_count
  
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-private-rt-${count.index + 1}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = var.az_count
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = var.az_count
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name        = "${local.cluster_name}-cluster-sg"
  description = "Security group for EKS cluster"
  vpc_id      = aws_vpc.main.id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-cluster-sg"
  })
}

resource "aws_security_group" "eks_nodes" {
  name        = "${local.cluster_name}-nodes-sg"
  description = "Security group for EKS worker nodes"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    description = "Node to node communication"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }
  
  ingress {
    description     = "Cluster to node communication"
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-nodes-sg"
  })
}

# Cyber Range Security Group
resource "aws_security_group" "cyber_range" {
  name        = "${local.cluster_name}-cyber-range-sg"
  description = "Security group for cyber range services"
  vpc_id      = aws_vpc.main.id
  
  dynamic "ingress" {
    for_each = local.cyber_range_ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
      description = ingress.value.description
    }
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-cyber-range-sg"
  })
}

# IAM Role for EKS Cluster
resource "aws_iam_role" "eks_cluster" {
  name = "${local.cluster_name}-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# IAM Role for EKS Node Group
resource "aws_iam_role" "eks_nodes" {
  name = "${local.cluster_name}-nodes-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodes.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodes.name
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = local.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
    security_group_ids      = [aws_security_group.eks_cluster.id]
  }
  
  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }
  
  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_cloudwatch_log_group.eks
  ]
  
  tags = local.common_tags
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/${local.cluster_name}/cluster"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.eks.arn
  
  tags = local.common_tags
}

# KMS Key for EKS Encryption
resource "aws_kms_key" "eks" {
  description             = "KMS key for EKS cluster encryption"
  deletion_window_in_days = 10
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-kms-key"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.cluster_name}-kms-key"
  target_key_id = aws_kms_key.eks.key_id
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${local.cluster_name}-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  
  capacity_type  = var.node_capacity_type
  instance_types = var.node_instance_types
  
  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }
  
  update_config {
    max_unavailable = 1
  }
  
  ami_type       = var.node_ami_type
  disk_size      = var.node_disk_size
  
  remote_access {
    ec2_ssh_key               = var.node_ssh_key
    source_security_group_ids = [aws_security_group.eks_nodes.id]
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
  
  tags = local.common_tags
}

# ECR Repository for container images
resource "aws_ecr_repository" "gan_cyber_range" {
  name                 = "gan-cyber-range"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
  
  tags = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "gan_cyber_range" {
  repository = aws_ecr_repository.gan_cyber_range.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 production images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["prod", "production"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Keep last 5 development images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["dev", "development"]
          countType     = "imageCountMoreThan"
          countNumber   = 5
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 3
        description  = "Expire untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# RDS Instance for application database
resource "aws_db_subnet_group" "main" {
  name       = "${local.cluster_name}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name        = "${local.cluster_name}-rds-sg"
  description = "Security group for RDS database"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-sg"
  })
}

resource "aws_db_instance" "main" {
  count = var.enable_rds ? 1 : 0
  
  identifier = "${local.cluster_name}-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.rds_instance_class
  
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.eks.arn
  
  db_name  = var.database_name
  username = var.database_username
  password = var.database_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = var.rds_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "production"
  deletion_protection = var.environment == "production"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring[0].arn
  
  tags = local.common_tags
}

# IAM Role for RDS Monitoring
resource "aws_iam_role" "rds_monitoring" {
  count = var.enable_rds ? 1 : 0
  
  name = "${local.cluster_name}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  count = var.enable_rds ? 1 : 0
  
  role       = aws_iam_role.rds_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.cluster_name}-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
  
  tags = local.common_tags
}

resource "aws_security_group" "elasticache" {
  name        = "${local.cluster_name}-cache-sg"
  description = "Security group for ElastiCache"
  vpc_id      = aws_vpc.main.id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-cache-sg"
  })
}

resource "aws_elasticache_replication_group" "main" {
  count = var.enable_redis ? 1 : 0
  
  replication_group_id       = "${local.cluster_name}-redis"
  description                = "Redis cluster for GAN Cyber Range"
  
  port                = 6379
  parameter_group_name = "default.redis7"
  node_type           = var.redis_node_type
  num_cache_clusters  = var.redis_num_cache_nodes
  
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.elasticache.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = var.redis_auth_token
  
  maintenance_window = "sun:03:00-sun:04:00"
  snapshot_window    = "02:00-03:00"
  
  tags = local.common_tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.cyber_range.id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = var.environment == "production"
  
  access_logs {
    bucket  = aws_s3_bucket.alb_logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = local.common_tags
}

# S3 Bucket for ALB Logs
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "${local.cluster_name}-alb-logs-${random_id.bucket_suffix.hex}"
  force_destroy = var.environment != "production"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_lifecycle_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  rule {
    id     = "log_lifecycle"
    status = "Enabled"
    
    expiration {
      days = var.alb_log_retention_days
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}