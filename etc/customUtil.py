import matplotlib.pyplot as plt

def showGraph(dict_losses, dict_scores):
    num_epoch = len(dict_losses['dev'])
    plt.figure(figsize=(13,13))

    plt.subplot(421)
    plt.title('Loss')
    plt.plot(range(len(dict_losses['train'])), dict_losses['train'], label='train')
    plt.xlabel('Iteration')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.subplot(423)
    plt.plot(range(1,num_epoch+1), dict_losses['dev'], label='dev')
    plt.plot(range(1,num_epoch+1), dict_losses['test'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Dev, Test Loss')
    plt.legend()

    plt.subplot(222)
    plt.plot(range(1,num_epoch+1), dict_scores['train'][0], label='train')
    plt.plot(range(1,num_epoch+1), dict_scores['dev'][0], label='dev')
    plt.plot(range(1,num_epoch+1), dict_scores['test'][0], label='test')
    plt.legend()
    plt.title('score1_acc_sample')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(223)
    plt.plot(range(1,num_epoch+1), dict_scores['train'][1], label='train')
    plt.plot(range(1,num_epoch+1), dict_scores['dev'][1], label='dev')
    plt.plot(range(1,num_epoch+1), dict_scores['test'][1], label='test')
    plt.legend()
    plt.title('score2_acc_word')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(224)
    plt.plot(range(1,num_epoch+1), dict_scores['train'][2], label='train')
    plt.plot(range(1,num_epoch+1), dict_scores['dev'][2], label='dev')
    plt.plot(range(1,num_epoch+1), dict_scores['test'][2], label='test')
    plt.legend()
    plt.title('score3_f1')
    plt.xlabel('Epoch')
    plt.ylabel('Avg F1 Score')

    plt.show()
    
    
def showParallel(words, tags, targets=None):
    if '<PAD>' in words:
        length = words.index('<PAD>')
    else:
        lengths = [len(words),len(tags)]        
        if targets is not None:
            lengths.append(len(targets))
        length = min(lengths)
    
    if targets is not None:
        print(f'{"words":20}{"tags":8}targets')
        for i in range(length):
            color = 31 if tags[i] != targets[i] else 0
            print(f'\033[{color}m{words[i]:20}{tags[i]:8}{targets[i]}\033[0m')
    else:
        print(f'{"words":20}tags')
        for i in range(length):
            print(f'{words[i]:20}{tags[i]}')