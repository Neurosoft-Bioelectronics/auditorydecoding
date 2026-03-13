# auditorydecoding

Auditory decoding of brain activity in different animals.

## Overview

This project analyzes neural recordings from various animal models to decode auditory information from brain activity. The pipeline processes electrophysiology data (iEEG recordings) following the BIDS standard and prepares datasets for machine learning models.

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd auditorydecoding
```

2. Install dependencies using uv:
```bash
uv sync
```

## Data Preparation

### Setting Up Brainsets Config

Before processing data, you need to configure the brainsets module. The brainsets configuration file specifies settings for data processing.

1. Initialize and configure brainsets using the interactive command:

```bash
uv run brainsets config
```

This command will prompt you to set the `raw_dir` (location of your BIDS/raw data) and `processed_dir` (location where processed outputs will be stored).


### Processing the Data

Once brainsets is configured, prepare the Neurosoft minipigs 2026 dataset using the following command:

```bash
uv run brainsets prepare --local datasets/neurosoft_minipigs_2026
```

This command will:
- Validate the raw BIDS dataset
- Process iEEG recordings
- Extract relevant features and metadata
- Generate AI-ready datasets for model training

## Dataset Structure

The project uses the BIDS (Brain Imaging Data Structure) format for organizing neurophysiology data:

```
datasets/
├── neurosoft_minipigs_2026/
│   ├── sub-*/
│   │   └── ses-*/
│   │       └── ieeg/
│   │           ├── *_ieeg.edf
│   │           ├── *_ieeg.json
│   │           └── *_channels.tsv
│   └── dataset_description.json
```

## Usage

For detailed information on using the pipeline and analyzing the data, refer to the documentation in the `datasets/neurosoft_minipigs_2026/` directory.

## Development

### Running Tests

```bash
uv run pytest
```

### Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. To check code style:

```bash
uv run ruff check .
uv run ruff format --check .
```

To automatically fix formatting issues:

```bash
uv run ruff format .
```

## Dependencies

- **brainsets**: BIDS processing and neural data utilities
- **mne**: MEG/EEG analysis
- **mne-bids**: BIDS I/O for MNE
- **scikit-learn**: Machine learning utilities

See `pyproject.toml` for the complete dependency list.

## Contributing

Contributions are welcome. Please ensure code follows the project's style guidelines and passes all checks.

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue in the repository.
