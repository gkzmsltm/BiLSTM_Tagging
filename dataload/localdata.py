from torch.utils.data import Dataset
from dataload import lang

filenames = {'dev': 'data/pos_data/dev.dat',
            'test': 'data/pos_data/test.dat',
            'train': 'data/pos_data/train.dat'}

class CustomDataset(Dataset):
    def __init__(self, filename, engdict, posdict, device, strmode=False, charmode=False):
        #   데이터셋의 전처리를 해주는 부분
        data=[[],[]]
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            for i in range(len(lines)):
                pair = [p.strip() for p in lines[i].split('\t')]
                if strmode:
                    input_tensor = pair[0]
                    target_tensor = posdict.tensorFromSentence(pair[1], device=device)
                elif charmode:
                    input_tensor = engdict.charTensorFromSentence(pair[0], device=device)
                    target_tensor = posdict.tensorFromSentence(pair[1], device=device)
                else:
                    input_tensor = engdict.tensorFromSentence(pair[0], device=device)
                    target_tensor = posdict.tensorFromSentence(pair[1], device=device)
                data[0].append(input_tensor)
                data[1].append(target_tensor)
        self.x_data = data[0]
        self.y_data = data[1]

    def __len__(self):
        #   데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return len(self.x_data)

    def __getitem__(self, idx): 
        #   데이터셋에서 특정 1개의 샘플을 가져오는 함수
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
    
    
def load_eng_pos(device, strmode=False, charmode=False):
    engdict = lang.Lang('eng')
    posdict = lang.Lang('pos')
    MAX_LENGTH = 0
    
    for filename in filenames.values():
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            for i in range(len(lines)):
                pair = [p.strip() for p in lines[i].split('\t')]
#                 LENGTH = len(pair[0].split(' '))
#                 if MAX_LENGTH < LENGTH:
#                     MAX_LENGTH = LENGTH
                engdict.addSentence(pair[0])
                posdict.addSentence(pair[1])

    datasets = {'dev': CustomDataset(filenames['dev'], engdict, posdict, device, strmode, charmode),
                'test': CustomDataset(filenames['test'], engdict, posdict, device, strmode, charmode),
                'train': CustomDataset(filenames['train'], engdict, posdict, device, strmode, charmode)}            
                
    return datasets, engdict, posdict

# def load_eng_pos():
#     engdict = lang.Lang('eng')
#     posdict = lang.Lang('pos')
#     MAX_LENGTH = 0
    
#     for filename in filenames.values():
#         with open(filename, 'r') as fp:
#             lines = fp.readlines()
#             for i in range(len(lines)):
#                 pair = [p.strip() for p in lines[i].split('\t')]
#                 LENGTH = len(pair[0].split(' '))
#                 if MAX_LENGTH < LENGTH:
#                     MAX_LENGTH = LENGTH
#                 engdict.addSentence(pair[0])
#                 posdict.addSentence(pair[1])

#     return engdict, posdict, MAX_LENGTH
