import numpy as np
import torch
from tqdm import tqdm


def pred(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))
