def score1_acc_sample(model, dataset, batch_size):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total = len(dataset)
    cnt_corr = 0
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        
        o = model(x_train)
        t = y_train.view(-1)

        __tagged, tagged = torch.max(o, dim=-1)
        
        for i in range(x_train.size(0)):
            d = tagged[i]!=y_train[i]
            if tagged[i][d].size()[0] == 0:
                cnt_corr +=1

    # print(f'{cnt_corr / total * 100:.4}%')
    return cnt_corr / total
    
    
    
def score2_acc_word(model, dataset, batch_size):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    total_words = 0
    cnt_corr = 0
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        
        o = model(x_train)
        t = y_train.view(-1)

        __tagged, tagged = torch.max(o, dim=-1)
        
        for i in range(x_train.size(0)):
            l = tagged[i].tolist()
            if tag_PAD in l:
                seq_len = l.index(tag_PAD)
            else:
                seq_len = len(l)
            total_words += seq_len
            d = tagged[i,:seq_len]==y_train[i,:seq_len]
            cnt_corr += tagged[i,:seq_len][d].size()[0]

    # print(f'{cnt_corr} / {total_words}\t{cnt_corr / total_words * 100:.4}%')
    return cnt_corr / total_words




def score3_f1(model, dataset, batch_size):
    c = posdict.n_words
    table = torch.zeros(c,c)
    # print(table.size())
    # return
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # total_words = 0
    # cnt_corr = 0
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        
        o = model(x_train)
        t = y_train.view(-1)

        __tagged, tagged = torch.max(o, dim=-1)
        
        for i in range(x_train.size(0)):
            l = tagged[i].tolist()
            if tag_PAD in l:
                seq_len = l.index(tag_PAD)
            else:
                seq_len = len(l)
            for j in range(seq_len):
                table[y_train[i,j],tagged[i,j]] += 1

    tp = torch.tensor([table[i,i] for i in range(c)])[2:]
    d0sum = table[2:,2:].sum(dim=0)
    d1sum = table[2:,2:].sum(dim=1)
    allsum = d1sum.sum()
    
    pr = tp / d0sum
    temp = pr != pr
    pr[temp] = 0
    
    re = tp / d1sum
    temp = re != re
    re[temp] = 0
    
    f1 = 2 * pr * re / (pr + re)
    temp = f1 != f1
    f1[temp] = 0
    
    avg_f1 = (f1 * d1sum).sum() / allsum

    # print(ttttf1)
    return avg_f1.item()