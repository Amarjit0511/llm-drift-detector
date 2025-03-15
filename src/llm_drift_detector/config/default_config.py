"""
Default configuration values for LLM Drift Detector.

This module defines the default configuration structure and values
that are used when no explicit configuration is provided.
"""

DEFAULT_CONFIG = {
    # Provider configurations
    "providers": {
        "openai": {
            "enabled": False,
            "model_name": "gpt-3.5-turbo",
            "api_key_env_var": "OPENAI_API_KEY",
            "timeout_seconds": 30,
            "max_retries": 3,
            "batch_size": 10
        },
        "azure": {
            "enabled": False,
            "model_name": "gpt-4",
            "api_key_env_var": "AZURE_OPENAI_API_KEY",
            "endpoint_env_var": "AZURE_OPENAI_ENDPOINT",
            "api_version": "2023-05-15",
            "deployment_name": "gpt-4",
            "timeout_seconds": 30,
            "max_retries": 3,
            "batch_size": 10
        },
        "anthropic": {
            "enabled": False,
            "model_name": "claude-2.1",
            "api_key_env_var": "ANTHROPIC_API_KEY",
            "timeout_seconds": 60,
            "max_retries": 3,
            "batch_size": 5
        },
        "vllm": {
            "enabled": False,
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "model_path": None,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_num_batched_tokens": 4096,
            "host": "localhost",
            "port": 8000,
            "use_existing_server": False
        },
        "huggingface": {
            "enabled": False,
            "model_name": "gpt2",
            "token_env_var": "HF_API_TOKEN",
            "use_auth_token": True,
            "device": "cuda",
            "batch_size": 1
        }
    },
    
    # Data collection settings
    "data": {
        "collection": {
            "frequency_minutes": 60,
            "batch_size": 100,
            "storage_path": "./data/collected/",
            "sampling_strategy": "random",  # Options: random, sequential, time-based
            "min_samples_per_day": 50
        },
        "reference": {
            "path": "./data/reference_distribution.pkl",
            "update_frequency_days": 7,
            "min_samples": 1000,
            "bootstrap_from_current": True
        },
        "storage": {
            "type": "local",  # Options: local, redis, sql
            "retention_days": 90,
            "compression": True,
            
            # Local file storage settings
            "local": {
                "directory": "./data/storage/",
                "format": "parquet"  # Options: json, csv, parquet
            },
            
            # Redis storage settings
            "redis": {
                "host": "localhost",
                "port": 6379,
                "password": None,
                "db": 0,
                "prefix": "llm_drift:"
            },
            
            # SQL storage settings
            "sql": {
                "connection_string": "sqlite:///data/storage/drift.db",
                "table_prefix": "llm_drift_"
            }
        }
    },
    
    # Drift detection settings
    "drift_detection": {
        "window_size": 100,  # Number of samples to use for current distribution
        "min_samples_for_detection": 30,
        "detection_frequency_minutes": 60,
        "p_value_threshold": 0.05,  # Statistical significance threshold
        
        # Embedding-based methods
        "embedding": {
            "enabled": True,
            "model": "all-MiniLM-L6-v2",
            "device": "cuda" if True else "cpu",  # Will use CUDA if available
            "batch_size": 32,
            "distance_metric": "cosine",  # Options: cosine, euclidean
            "pca_components": 50,  # Number of PCA components to use
            "threshold": 0.15,
            "weight": 1.0  # Relative weight in combined score
        },
        
        # Distribution-based methods
        "distribution": {
            "enabled": True,
            "features": ["response_length", "token_count", "response_time"],
            "methods": [
                {
                    "name": "ks_test",  # Kolmogorov-Smirnov test
                    "threshold": 0.05
                },
                {
                    "name": "js_divergence",  # Jensen-Shannon divergence
                    "threshold": 0.2
                }
            ],
            "weight": 1.0  # Relative weight in combined score
        },
        
        # Semantic coherence methods
        "semantic": {
            "enabled": True,
            "method": "perplexity",  # Options: perplexity, contradiction, topic
            "coherence_threshold": 0.7,
            "contradiction_threshold": 0.3,
            "topic_drift_threshold": 0.4,
            "model": "distilgpt2",  # Model for perplexity calculation
            "weight": 1.0  # Relative weight in combined score
        },
        
        # Performance metrics
        "performance": {
            "enabled": True,
            "metrics": [
                {
                    "name": "response_time",
                    "upper_threshold": 5.0,  # seconds
                    "lower_threshold": 0.1   # seconds
                },
                {
                    "name": "token_count",
                    "upper_threshold": 500,
                    "lower_threshold": 10
                },
                {
                    "name": "error_rate",
                    "upper_threshold": 0.05,  # 5% error rate
                    "lower_threshold": 0.0
                }
            ],
            "weight": 0.5  # Relative weight in combined score
        }
    },
    
    # Monitoring settings
    "monitoring": {
        "log_level": "INFO",
        "metrics_collection_interval": 60,  # seconds
        "storage_backend": "local",  # Options: local, prometheus, cloudwatch
        
        "local": {
            "metrics_file": "./data/metrics/metrics.csv",
            "rotation": "daily"  # Options: hourly, daily, weekly, monthly
        },
        
        "prometheus": {
            "enabled": False,
            "port": 8000,
            "endpoint": "/metrics"
        },
        
        "dashboard": {
            "enabled": True,
            "port": 8050,
            "host": "0.0.0.0",
            "update_interval_seconds": 60,
            "retention_days": 30
        },
        
        "alerts": {
            "enabled": True,
            "cooldown_minutes": 60,  # Minimum time between alerts
            
            "thresholds": {
                "warning": 0.7,  # Percentage of threshold for warning
                "critical": 1.0   # Percentage of threshold for critical alert
            },
            
            "channels": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password_env_var": "EMAIL_PASSWORD",
                    "from_address": "",
                    "to_addresses": []
                },
                "slack": {
                    "enabled": False,
                    "webhook_url_env_var": "SLACK_WEBHOOK_URL",
                    "channel": "#llm-drift-alerts",
                    "username": "LLM Drift Monitor"
                },
                "http": {
                    "enabled": False,
                    "endpoint": "",
                    "method": "POST",
                    "headers": {}
                }
            }
        }
    },
    
    # API settings
    "api": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
        "cors_origins": ["*"],
        "auth": {
            "enabled": False,
            "api_key_header": "X-API-Key",
            "api_keys": []
        }
    },
    
    # Logging settings
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,
        "rotate": True,
        "max_size": 10485760,  # 10 MB
        "backup_count": 5
    },
    
    # Plugin settings
    "plugins": {
        "enabled": True,
        "directory": "./plugins/",
        "allowlist": [],
        "blocklist": []
    }
}





