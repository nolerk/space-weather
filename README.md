# Space Weather

Various Machine Learning Approaches to Predict Solar Flares

## Get Started

1. Clone the repo and `cd` to it

2. Create and activate a virtual environment

3. Install dependencies

    ```bash
    python3 -m pip install .
    ```

4. Download and extract the dataset

    ```bash
    mkdir data
    cd data

    wget -O partition1_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EBCFKM/BMXYCB"
    tar xf partition1_instances.tar.gz
    rm partition1_instances.tar.gz

    wget -O partition2_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EBCFKM/TCRPUD"
    tar xf partition2_instances.tar.gz
    rm partition2_instances.tar.gz

    wget -O partition3_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EBCFKM/PTPGQT"
    tar xf partition3_instances.tar.gz
    rm partition3_instances.tar.gz

    wget -O partition4_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EBCFKM/FIFLFU"
    tar xf partition4_instances.tar.gz
    rm partition4_instances.tar.gz

    wget -O partition5_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/EBCFKM/QC2C3X"
    tar xf partition5_instances.tar.gz
    rm partition5_instances.tar.gz

    cd ..
    ```

5. To train models and save their state, run the command

    ```bash
    python3 main.py
    ```
