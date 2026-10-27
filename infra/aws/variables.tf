variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "enterprise-assistant-poc"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.small"
}

variable "api_desired_count" {
  type    = number
  default = 1
}

variable "llm_provider" {
  description = "offline | anthropic | openai"
  type        = string
  default     = "offline"
}

variable "anthropic_model" {
  type    = string
  default = "claude-sonnet-5-5"
}

variable "anthropic_api_key" {
  description = "Leave empty to run the offline extractive generator."
  type        = string
  default     = ""
  sensitive   = true
}

variable "allowed_cidrs" {
  description = "Who may reach the demo. Restrict to the buyer's office/VPN ranges."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
