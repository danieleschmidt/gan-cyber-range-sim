# Outputs for GAN Cyber Range Simulator Terraform Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

# EKS Cluster Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.eks_cluster.name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_iam_role.eks_cluster.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = aws_eks_cluster.main.version
}

# EKS Node Group Outputs
output "node_group_arn" {
  description = "Amazon Resource Name (ARN) of the EKS Node Group"
  value       = aws_eks_node_group.main.arn
}

output "node_group_status" {
  description = "Status of the EKS Node Group"
  value       = aws_eks_node_group.main.status
}

output "node_group_capacity_type" {
  description = "Type of capacity associated with the EKS Node Group"
  value       = aws_eks_node_group.main.capacity_type
}

output "node_group_instance_types" {
  description = "List of instance types associated with the EKS Node Group"
  value       = aws_eks_node_group.main.instance_types
}

output "node_group_iam_role_name" {
  description = "IAM role name associated with EKS node group"
  value       = aws_iam_role.eks_nodes.name
}

output "node_group_iam_role_arn" {
  description = "IAM role ARN associated with EKS node group"
  value       = aws_iam_role.eks_nodes.arn
}

# Database Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.main[0].endpoint : null
}

output "rds_port" {
  description = "RDS instance port"
  value       = var.enable_rds ? aws_db_instance.main[0].port : null
}

output "rds_db_name" {
  description = "RDS database name"
  value       = var.enable_rds ? aws_db_instance.main[0].db_name : null
}

output "rds_username" {
  description = "RDS database username"
  value       = var.enable_rds ? aws_db_instance.main[0].username : null
  sensitive   = true
}

output "rds_security_group_id" {
  description = "Security group ID for RDS"
  value       = aws_security_group.rds.id
}

# Redis Outputs
output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].configuration_endpoint_address : null
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].port : null
}

output "redis_security_group_id" {
  description = "Security group ID for Redis"
  value       = aws_security_group.elasticache.id
}

# ECR Outputs
output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.gan_cyber_range.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"  
  value       = aws_ecr_repository.gan_cyber_range.arn
}

output "ecr_registry_id" {
  description = "Registry ID where the repository was created"
  value       = aws_ecr_repository.gan_cyber_range.registry_id
}

# Load Balancer Outputs
output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.main.arn
}

# S3 Outputs
output "alb_logs_bucket_name" {
  description = "Name of the S3 bucket for ALB logs"
  value       = aws_s3_bucket.alb_logs.bucket
}

output "alb_logs_bucket_arn" {
  description = "ARN of the S3 bucket for ALB logs"
  value       = aws_s3_bucket.alb_logs.arn
}

# Security Group Outputs
output "eks_cluster_security_group_id" {
  description = "Security group ID for EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "eks_nodes_security_group_id" {
  description = "Security group ID for EKS nodes"
  value       = aws_security_group.eks_nodes.id
}

output "cyber_range_security_group_id" {
  description = "Security group ID for cyber range services"
  value       = aws_security_group.cyber_range.id
}

# KMS Outputs
output "kms_key_id" {
  description = "ID of the KMS key used for encryption"
  value       = aws_kms_key.eks.key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key used for encryption"
  value       = aws_kms_key.eks.arn
}

output "kms_alias_name" {
  description = "Name of the KMS key alias"
  value       = aws_kms_alias.eks.name
}

# CloudWatch Outputs
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks.arn
}

# Connection Information
output "kubectl_config" {
  description = "kubectl config command to connect to the cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${aws_eks_cluster.main.name}"
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.main.name
}

# Application Configuration
output "application_config" {
  description = "Configuration values for the application"
  value = {
    cluster_name         = aws_eks_cluster.main.name
    cluster_endpoint     = aws_eks_cluster.main.endpoint
    database_endpoint    = var.enable_rds ? aws_db_instance.main[0].endpoint : null
    redis_endpoint      = var.enable_redis ? aws_elasticache_replication_group.main[0].configuration_endpoint_address : null
    ecr_repository_url  = aws_ecr_repository.gan_cyber_range.repository_url
    load_balancer_url   = "https://${aws_lb.main.dns_name}"
    vpc_id              = aws_vpc.main.id
    private_subnet_ids  = aws_subnet.private[*].id
    security_group_ids  = [
      aws_security_group.eks_nodes.id,
      aws_security_group.cyber_range.id
    ]
  }
  sensitive = true
}

# Environment Information
output "environment_info" {
  description = "Environment information"
  value = {
    environment     = var.environment
    aws_region      = var.aws_region
    project_name    = var.project_name
    vpc_cidr        = var.vpc_cidr
    availability_zones = data.aws_availability_zones.available.names
  }
}

# Cost Optimization Information
output "cost_optimization_info" {
  description = "Information for cost optimization"
  value = {
    node_capacity_type     = var.node_capacity_type
    node_instance_types    = var.node_instance_types
    enable_spot_instances  = var.enable_spot_instances
    node_desired_size      = var.node_desired_size
    node_min_size         = var.node_min_size
    node_max_size         = var.node_max_size
  }
}

# Monitoring and Logging Information
output "monitoring_endpoints" {
  description = "Monitoring and logging endpoints"
  value = {
    cloudwatch_log_group = aws_cloudwatch_log_group.eks.name
    alb_logs_bucket     = aws_s3_bucket.alb_logs.bucket
    cluster_logging     = aws_eks_cluster.main.enabled_cluster_log_types
  }
}

# Security Information
output "security_configuration" {
  description = "Security configuration information"
  value = {
    encryption_enabled        = true
    kms_key_id               = aws_kms_key.eks.key_id
    vpc_endpoint_private     = aws_eks_cluster.main.vpc_config[0].endpoint_private_access
    vpc_endpoint_public      = aws_eks_cluster.main.vpc_config[0].endpoint_public_access
    secrets_encryption       = var.enable_secrets_encryption
    pod_security_policy      = var.enable_pod_security_policy
    network_policy_enabled   = var.enable_network_policy
  }
}

# DNS and SSL Information
output "dns_configuration" {
  description = "DNS and SSL configuration"
  value = {
    domain_name      = var.domain_name
    certificate_arn  = var.certificate_arn
    load_balancer_dns = aws_lb.main.dns_name
    zone_id         = aws_lb.main.zone_id
  }
}