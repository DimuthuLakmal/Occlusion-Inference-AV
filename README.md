# Vision-enhanced Spatio-Temporal Sparse Bipartite Graph Neural Network for Occlusion Inference

## Installation and Quick Start

1. Create a new Python environment or select a pre-existing one. 
   This code is tested with Python 3.8, but is very probably compatible with newer releases of Python.

   If you use Anaconda3, this can be done as follows:
   ```shell 
   conda create --name vstsbgat python=3.8
   conda activate vstsbgat
   ```

2. Install required packages by navigating to the root directory and using
    ```shell 
    pip3 install -r requirements.txt
    ```
   
3. Follow the instructions in README files placed in the build_dataset folder for preprocessing inD dataset. Please contact inD authors to access the original inD dataset. We have only        provided the annotation extension in this repo.
   Once you have the observation data prepared, please edit the `configs/config.yaml` file to set the correct paths to your data.

4. Run the training script:
    ```shell 
    python3 src/train.py
    ```
   
5. Run the evaluation script:
    ```shell 
    python3 src/test.py
    ```

## Citation

If you use one of our datasets or these scripts in your work, please cite inD datasets as follows:
### inD Paper
```
@INPROCEEDINGS{inDdataset,
               title={The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections},
               author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
               booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
               pages={1929-1934},
               year={2019},
               doi={10.1109/IV47402.2020.9304839}}
```
