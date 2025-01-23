# OpenAI Structured CLI Use Cases

This directory contains 20 practical use cases demonstrating the power and flexibility of the OpenAI Structured CLI. Each use case is designed to showcase specific features that set this tool apart from general-purpose chatbots or single-file AI tools.

## Structure

Each use case follows a consistent structure:

```bash
use_case_name/
├── README.md            # Description, setup, and usage instructions
├── system_prompt.yaml   # System prompt configuration
├── task.j2             # Jinja2 template for processing
├── schema.json         # JSON schema for structured output
├── run.sh             # Shell script to run the example
└── sample_input/      # Sample files/data for demonstration
```

## Use Cases

1. [Code Review](01_code_review/) - Multi-file automated code review with structured output
2. [Test Generation](02_test_generation/) - Automated test case generation and optimization
3. [Schema Validation](03_schema_validation/) - Multi-file JSON/YAML config validation
4. [Text Analysis](04_text_analysis/) - Streaming text analysis for large files
5. [Release Notes](05_release_notes/) - Automated release notes from git history
6. [Security Analysis](06_security_analysis/) - Security vulnerability scanning
7. [Log Analysis](07_log_analysis/) - Large-scale log file processing
8. [PII Scanner](08_pii_scanner/) - GDPR/PII data leak detection
9. [CI/CD Validation](09_cicd_validation/) - Pipeline configuration validation
10. [IaC Validation](10_iac_validation/) - Infrastructure-as-Code template validation
11. [License Audit](11_license_audit/) - Dependency license compliance checking
12. [Multi-Repo Security](12_multi_repo_security/) - Cross-repository security scanning
13. [Test Analysis](13_test_analysis/) - Test failure root cause analysis
14. [API Testing](14_api_testing/) - OpenAPI-based API testing
15. [Proto Validation](15_proto_validation/) - Protocol Buffer validation
16. [Clone Detection](16_clone_detection/) - Code duplication analysis
17. [SAST Processing](17_sast_processing/) - Security scan result processing
18. [TODO Extraction](18_todo_extraction/) - Project-wide TODO comment extraction
19. [Table Extraction](19_table_extraction/) - Multi-format table data processing
20. [Pipeline Config](20_pipeline_config/) - Data pipeline configuration management

## Key Features Demonstrated

- **Multi-File Processing**: Handle multiple files in a single pass
- **Streaming**: Process large files chunk by chunk
- **Schema Validation**: Enforce structured output
- **Security Constraints**: Handle sensitive data appropriately
- **CI/CD Integration**: Automation-friendly design
- **Custom Templates**: Flexible Jinja2 templating
- **System Prompts**: Customizable AI behavior

## Getting Started

1. Each use case directory contains its own README with specific setup instructions
2. Use the provided `run.sh` script in each directory to execute the example
3. Examine the sample input and output to understand the transformation
4. Modify the templates and schemas to adapt to your needs

## Prerequisites

- OpenAI Structured CLI installed (`pip install openai-structured`)
- OpenAI API key configured
- Bash shell for running the examples
- Git for version control examples
