# NAIT

## Introduction
This repository shares the code and data of our latest work "Towards Robust Process Reward Modeling via Noise-aware Learning".

<img src='./framework.png' alt="sym" width="100%" title="Framework">


## Installation

To set up the project locally, please follow the instructions below:

1. Clone the repository:

   ```sh
   git clone https://github.com/xiebin23/NAIT.git
   cd NAIT
   ```

2. Create and activate a virtual environment:

   ```sh
   conda create -n nait python=3.10
   conda activate nait
   ```

3. Install the required dependencies:
   ```sh
   pip install transformers deepspeed openrlhf==0.9.1
   ```

## Example

After setting up the environment, you can run the experiments and analysis scripts as follows:

1. **Training:**

   ```sh
   python partial_data_construct/hh_data_process_partial.py
   ```

2. **Evaluate:**

