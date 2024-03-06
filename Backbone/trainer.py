import os
import time
import torch
from util import dice_score, CE_VE

def train_epoch(dataloader, model, optimizer, epoch, args):
    model.train()
    loss_f = 0
    dice = 0
    img_count = 0
    batch_count = 0
    for i, batch in enumerate(dataloader):
        batch_count += 1
        pet_tensor = torch.autograd.Variable(batch[0]).cuda(0)
        ct_tensor = torch.autograd.Variable(batch[1]).cuda(0)
        label_tensor = torch.autograd.Variable(batch[2])
        img_count += pet_tensor.size()[0]
        label_tensor = label_tensor.cuda(0).long()

        # ### optimize backbone
        output_tensor, prob = model(pet_tensor, ct_tensor, dropout=False)
        pred_tensor = torch.argmax(output_tensor, 1)
        seg_loss = model.get_loss(label_tensor)
        loss = seg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_f += loss.float().item()
        dice_value = dice_score(pred_tensor, label_tensor)
        dice += dice_value
        if i % 100 == 0:
            print("\tSeg Loss: %.3f, Dice: %.3f" % (seg_loss.float(), dice_value))

    loss_f /= len(dataloader)
    dice = dice / batch_count
    return loss_f, dice

def test_epoch(dataloader, model):
    model.eval()
    img_count = 0
    batch_count = 0
    dice = 0
    CE = 0
    VE = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_count += 1
            pet_tensor = torch.autograd.Variable(batch[0]).cuda(0)
            ct_tensor = torch.autograd.Variable(batch[1]).cuda(0)
            label_tensor = torch.autograd.Variable(batch[2])
            img_count += pet_tensor.size()[0]
            label_tensor = label_tensor.cuda(0).long()

            output_tensor, prob = model(pet_tensor, ct_tensor, dropout=False)
            pred_tensor = torch.argmax(output_tensor, 1)

            dice += dice_score(pred_tensor, label_tensor)
            ce, ve = CE_VE(pred_tensor, label_tensor)
            CE += ce
            VE += ve
    return dice/batch_count, CE/batch_count, VE/batch_count


def train(args, dataloader, num_epochs, model, optimizer):
    prev_dice = 0.0
    print("Start training...")
    best_epoch = 0
    for epoch in range(num_epochs):
        t_start = time.time()
        #### training
        train_loss, train_dice = train_epoch(dataloader['train'], model, optimizer, epoch, args)

        #### validation
        val_dice, val_ce, val_ve = test_epoch(dataloader['val'], model)

        #### testing
        test_dice, test_ce, test_ve = test_epoch(dataloader['test'], model)
        delta = time.time() - t_start

        str1 = "Epoch: {}\tTrain Loss: {:.4f}\tTrain Dice: {:.3f}\tVal Dice: {:.3f}\tVal CE: {:.3f}\tVal VE: {:.3f}\tlr: {:.6f}\n".format(
            epoch+1, train_loss, train_dice, val_dice, val_ce, val_ve, optimizer.state_dict()['param_groups'][0]['lr'])

        str = "\t\tTest Dice: {:.3f}\tTest CE: {:.3f}\tTest VE: {:.3f}\n".format(
             test_dice, test_ce, test_ve)
        print(str1)
        print(str)

        if val_dice > prev_dice:
            prev_dice = val_dice
            best_epoch = epoch
            torch.save(model.cpu().state_dict(), os.path.join(args.save_dir, "%s_val_%.3f_test_%.3f_epoch_%d.pth"%(args.model_name, val_dice, test_dice, epoch)))
            print("Best model saved")
        model.cuda(0)

        if epoch - best_epoch > 30:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            new_lr = cur_lr * 0.5
            for p in optimizer.param_groups:
                p['lr'] = new_lr
