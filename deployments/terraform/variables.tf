# Variables for GAN Cyber Range Simulator Terraform Infrastructure

# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "gan-cyber-range-sim"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

# Networking Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3
  validation {
    condition     = var.az_count >= 2 && var.az_count <= 6
    error_message = "Availability zone count must be between 2 and 6."
  }
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the EKS cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# EKS Node Group Configuration
variable "node_capacity_type" {
  description = "Type of capacity associated with the EKS Node Group (ON_DEMAND or SPOT)"
  type        = string
  default     = "ON_DEMAND"
  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.node_capacity_type)
    error_message = "Node capacity type must be either ON_DEMAND or SPOT."
  }
}

variable "node_instance_types" {
  description = "List of instance types for EKS worker nodes"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "node_ami_type" {
  description = "Type of Amazon Machine Image (AMI) associated with the EKS Node Group"
  type        = string
  default     = "AL2_x86_64"
}

variable "node_disk_size" {
  description = "Disk size in GiB for worker nodes"
  type        = number
  default     = 50
  validation {
    condition     = var.node_disk_size >= 20 && var.node_disk_size <= 1000
    error_message = "Node disk size must be between 20 and 1000 GiB."
  }
}

variable "node_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "node_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 1
}

variable "node_ssh_key" {
  description = "EC2 Key Pair name for SSH access to worker nodes"
  type        = string
  default     = null
}

# RDS Configuration
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "Initial amount of storage (in gigabytes)"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "Maximum amount of storage (in gigabytes) for autoscaling"
  type        = number
  default     = 100
}

variable "rds_backup_retention_period" {
  description = "Days to retain backups"
  type        = number
  default     = 7
  validation {
    condition     = var.rds_backup_retention_period >= 0 && var.rds_backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "gancyberrange"
}

variable "database_username" {
  description = "Username for the database"
  type        = string
  default     = "dbadmin"
}

variable "database_password" {
  description = "Password for the database"
  type        = string
  sensitive   = true
  validation {
    condition     = length(var.database_password) >= 8
    error_message = "Database password must be at least 8 characters long."
  }
}

# Redis Configuration
variable "enable_redis" {
  description = "Enable ElastiCache Redis cluster"
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the Redis cluster"
  type        = number
  default     = 2
  validation {
    condition     = var.redis_num_cache_nodes >= 1 && var.redis_num_cache_nodes <= 6
    error_message = "Number of Redis cache nodes must be between 1 and 6."
  }
}

variable "redis_auth_token" {
  description = "Auth token for Redis cluster"
  type        = string
  sensitive   = true
  default     = null
}

# Logging Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 14
  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention days must be one of the valid CloudWatch retention periods."
  }
}

variable "alb_log_retention_days" {
  description = "ALB access log retention in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "enable_container_insights" {
  description = "Enable Container Insights for EKS cluster"
  type        = bool
  default     = true
}

variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

# Monitoring and Alerting
variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_fluentd" {
  description = "Enable Fluentd for log aggregation"
  type        = bool
  default     = true
}

# Backup Configuration
variable "enable_backup" {
  description = "Enable AWS Backup for resources"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

# Compliance and Security
variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Kubernetes Network Policy"
  type        = bool
  default     = true
}

variable "enable_secrets_encryption" {
  description = "Enable encryption of Kubernetes secrets"
  type        = bool
  default     = true
}

# Application Configuration
variable "cyber_range_replicas" {
  description = "Number of cyber range application replicas"
  type        = number
  default     = 2
  validation {
    condition     = var.cyber_range_replicas >= 1 && var.cyber_range_replicas <= 20
    error_message = "Cyber range replicas must be between 1 and 20."
  }
}

variable "cyber_range_cpu_request" {
  description = "CPU request for cyber range pods"
  type        = string
  default     = "500m"
}

variable "cyber_range_memory_request" {
  description = "Memory request for cyber range pods"
  type        = string
  default     = "1Gi"
}

variable "cyber_range_cpu_limit" {
  description = "CPU limit for cyber range pods"
  type        = string
  default     = "2"
}

variable "cyber_range_memory_limit" {
  description = "Memory limit for cyber range pods"
  type        = string
  default     = "4Gi"
}

# Domain and SSL Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = null
}

variable "certificate_arn" {
  description = "ARN of the SSL certificate"
  type        = string
  default     = null
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Secondary region for backup replication"
  type        = string
  default     = "us-east-1"
}

# Development and Testing
variable "enable_debug_mode" {
  description = "Enable debug mode for development"
  type        = bool
  default     = false
}

variable "enable_test_data" {
  description = "Enable test data population"
  type        = bool
  default     = false
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for the application"
  type        = map(bool)
  default = {
    advanced_analytics = true
    machine_learning   = true
    real_time_alerts   = true
    api_rate_limiting  = true
    audit_logging      = true
  }
}

# Resource Tagging
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}