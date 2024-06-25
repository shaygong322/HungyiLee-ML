### Homework 2: Classification

+ 任务：Multiclass Classification ---- Framewise phoneme prediction from speech.
  + phoneme: a unit of speech sound in a language that can serve to distinguish one word from the other
  + Task Introduction:
    + Data Preprocessing: Extract MFCC features from raw waveform (already done by TAs!) 
    + Classification: Perform framewise phoneme classification using pre-extracted MFCC features

+ Data Preprocessing
  + 25ms 为一个frame，相隔 10ms 取一个 frame，1s 可以取 100 个 frame
  + frame: 39-dim MFCC
  + Since each frame only contains 25ms of speech, a single frame is unlikely to represent a complete phoneme
    + Usually, a phoneme will span several frames
    + Concatenate the neighboring phonemes for training 对想找的 frame 加上前后各五个 flatten之后是 11 * 39 = 429 dim

+ Dataset
  + Training: 4268 preprocessed audio features with labels (total 2644158 frames)
  + Testing: 1078 preprocessed audio features (total 646268 frames) 
  + Label: 41 classes, each class represents a phoneme

+ Data Format (The TAs have already extracted the features)

  libriphone/

  - train_split.txt (train metadata): 其中每一行对应一个训练数据，其所对应的文件在feat/train/中
  - train_labels.txt (train labels): 由训练数据和labels组成，格式为: filename labels。其中，label 为 frame 对应的 phoneme
  - test_split.txt (test metadata): 其中每一行对应一个训练数据，其所对应的文件在feat/test/中
  - feat/: {id}.pt 音频对应的 MFCC，维度为39，这些文件可以通过 torch.load() 直接导入，导入后的shape为(T, 39)
    - train/
    - test/