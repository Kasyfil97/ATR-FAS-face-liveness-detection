# ATR-FAS: Flash Light Anti-Spoofing
This is an unofficial implementation of the paper [Enhancing Mobile Face Anti-Spoofing: A Robust Framework for Diverse Attack Types under Screen Flash](https://arxiv.org/abs/2308.15346). There are modifications based on the original [code](https://github.com/Chaochao-Lin/ATR-FAS). Instead of using 2D convolution, we use 3D convolution to extract more features from the sequence of frames. The architecture is as follows:
![fig2](https://github.com/Chaochao-Lin/ATR-FAS/blob/main/imgs/fig2.png)

## Installing packages
```bash
git clone https://github.com/Kasyfil97/ATR-FAS-flash-light-anti-spoofing.git
cd ATR-FAS-flash-light-anti-spoofing
pip install -r requirements.txt
```

## Create Dataset
To create dataset, you should create raw dataset folder where it consist of two folder (fake and real). Each folder 
```
raw_dataset/
├── fake/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── real/
    ├── video1.mp4
    ├── video2.mp4
    └── ...

```

After organizing your raw dataset, you need to create a training dataset by extracting frames from the videos. You can use the provided script `create_dataset.py` to do this. The script will create a new folder structure as follows:

```
raw_dataset/
├── fake/
│   ├── video1/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
│   ├── video2/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
│   └── ...
└── real/
    ├── video1/
    │   ├── frame1.jpg
    │   ├── frame2.jpg
    │   └── ...
    ├── video2/
    │   ├── frame1.jpg
    │   ├── frame2.jpg
    │   └── ...
    └── ...
```

To run the script, use the following command:

```bash
python extract_frames.py --input_folder raw_dataset --output_folder train_dataset
```
The code is a work in progress.
