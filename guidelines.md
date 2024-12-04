# Project Guidelines
YOU MUST FOLLOW THESE GUIDELINES WHEN WRITING CODE FOR THIS PROJECT.

## Overview
This document outlines the coding guidelines and project structure for this machine learning project. All contributors must follow these guidelines.

## Project Architecture
The project follows a modular ML pipeline architecture with the following key components:

### Core Components
- **DataLoaders**: Handle data loading from different sources
  - Base class defines interface
  - Source-specific implementations (e.g., CaliforniaLoader)
  - Factory pattern for instantiation

- **DataProcessors**: Handle data cleaning and preprocessing
  - Base class for common functionality
  - Source-specific processors
  - Standardized column naming
  - Missing value handling
  - Duplicate removal

- **FeatureCreators**: Handle feature engineering
  - Base class with common feature types
  - Source-specific custom features
  - sklearn-style naming (double underscore)
  - Supports datetime, categorical, numeric, and text features

### API Components (Future)
- **Models**: API data models/schemas
- **Routers**: API endpoint definitions

### Configuration
- Uses YAML files for configuration
- Separate configs per data source
- Hierarchical structure:
  - Data loading config
  - Processing config
  - Feature creation config

## Coding Standards

### Documentation
- Use NumPy style docstrings for all functions
- Include type hints for function parameters
- Document class attributes and methods

### Dependencies
- Add new dependencies via Poetry in terminal: `poetry add <package_name>`
- Do not modify pyproject.toml directly

### Best Practices

#### Configuration Management
- Store configs in artifacts/<source_name>/config.yaml
- Use typed configuration loading
- Validate config structure

#### Logging
- Use app.utils.get_logger for consistent logging
- Log at appropriate levels (INFO, WARNING, ERROR)
- Include context in log messages

#### Error Handling
- Implement proper exception handling
- Log errors with context
- Create output directories as needed

#### Code Organization
- Follow modular design
- Use inheritance and factory patterns
- Separate concerns (loading, processing, feature creation)

## Project Structure
```
project_root/
├── app/
│   ├── api/
│   │   ├── models/
│   │   └── routers/
│   ├── core/
│   │   ├── data_loaders/
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   └── [source]_loader.py
│   │   ├── data_processors/
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   └── [source]_processor.py
│   │   ├── feature_creators/
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   └── [source]_creator.py
│   │   └── models/
│   ├── utils/
│   │   └── config.py
│   └── main.py
├── artifacts/
│   └── [source_name]/
│       ├── config.yaml
│       ├── models/
│       └── transformers/
├── data/
│   └── [source_name]/
│       ├── raw/
│       ├── interim/
│       ├── processed/
│       └── features/
├── scripts/
│   ├── test_components.py
│   └── test_data_loading.py
├── poetry.lock
└── pyproject.toml
```

### Data Directory Structure
- **raw/**: Original, immutable data
- **interim/**: Intermediate data after basic processing
- **processed/**: Final data with features
- **features/**: Extracted features and transformations

### Artifacts Directory Structure
- **config.yaml**: Source-specific configuration
- **models/**: Trained model artifacts
- **transformers/**: Feature transformation artifacts