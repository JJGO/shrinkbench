import torch


def correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k) #.mul_(100.0 / batch_size))
        return res


def accuracy(model, dataloader, topk=(1,)):
    # Copy device from model
    device = next(model.parameters()).device

    accs = [0]*len(topk)

    with torch.no_grad():

        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            for i, c in enumerate(correct(output, target, topk)):
                accs[i] += c.item()

    accs = [acc / len(dataloader) for acc in accs]

    return accs
