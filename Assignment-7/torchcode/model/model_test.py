import torch
import os
from torch.functional import F


# os.chdir('d:\Python Projects\EVA')
cwd = os.getcwd()
# model_dir = os.path.join(cwd, 'Assignment-6/saved_models/model.pth')


def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = []):
    
    # model.load_state_dict(torch.load(model_dir)) # Loading the saved model
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():

        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            for d,i,j in zip(data, pred, target):
                if i != j:
                    misclassified.append([d.cpu(),i[0].cpu(),j.cpu()])

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))
    
    test_acc.append(100. * correct / len(test_dataloader.dataset))
    return misclassified