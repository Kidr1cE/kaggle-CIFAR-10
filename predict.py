import pandas
import torch
from model import resnet
from data_loader import get_test_data
from utils import utils

id_to_class = {0:'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8:'ship', 9: 'truck'}

save_path = "./data/resnet18.pth"

def predict(net, batch_size, device):
    net.eval()
    
    data_iter = get_test_data(batch_size)
    y = []

    with torch.no_grad():
        for X in data_iter:
            X = X.to(device)  # 确保数据在正确的设备上 不然会报错

            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            y.extend(predicted.cpu().tolist())  # 将结果转回CPU并转换为列表 ？

    write_submission(y)


def write_submission(y):
    submission = pandas.DataFrame({
        'id': range(1, len(y) + 1),
        'label': [id_to_class[idx] for idx in y]
    })

    submission.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    net = resnet.resnet18(10, 3)
    net = net.to(utils.try_gpu())

    net.load_state_dict(torch.load(save_path))

    predict.predict(net, 128, utils.try_gpu())
