#!/home/hzili1/tools/anaconda3/envs/py36/bin/python

import sys
import torch
import numpy as np
import time
import shutil
import kaldi_io
import random

def prepare_utt2feat(data_dir):
    utt2feat = {}
    with open("{}/feats.scp".format(data_dir), 'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        utt2feat[line.split()[0]] = line.split()[1]
    return utt2feat

def compute_mean_std(utt2feat, num_utt):
    utt_list = list(utt2feat.keys())
    random.shuffle(utt_list)
    utt_list = utt_list[:num_utt]
    feat_list = []
    cnt = 0
    for utt in utt_list:
        feat = kaldi_io.read_mat(utt2feat[utt])
        feat_list.append(feat)
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
    feat_array = np.concatenate(feat_list, 0)
    print(feat_array.shape)
    mean, std = np.mean(feat_array, 0), np.std(feat_array, 0)
    return mean, std

def record_info(train_info, dev_info, iteration, logger):
    loss_info = {"train_loss": train_info['loss'], "dev_loss": dev_info['loss']}
    logger.add_scalars("losses", loss_info, iteration)
    acc_info = {"train_top1": train_info['top1'], "train_top5": train_info['top5'], "dev_top1": dev_info['top1'], "dev_top5": dev_info['top5']}
    logger.add_scalars("acc", acc_info, iteration)
    return 0

def train(train_loader, model, device, criterion, metric_fc, optimizer, args):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader, 1):
        input, target = input.to(device), target.to(device)
        #input = input.transpose(0, 1) # (T, N, F)

        # compute output
        embedding_a, embedding_b = model(input)
        output, cosine = metric_fc(embedding_b, target)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = compute_accuracy(cosine, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
        optimizer.step()

    info = {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg}
    return info

def validate(valset_list, model, device, criterion, metric_fc, args):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for valset in valset_list:
            val_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers,
                    batch_size=args.batch_size, pin_memory=True, shuffle=False)
            for i, (input, target) in enumerate(val_loader, 1):
                input, target = input.to(device), target.to(device)
                #input = input.transpose(0, 1) # (T, N, F)

                # compute output
                embedding_a, embedding_b = model(input)
                output, cosine = metric_fc(embedding_b, target)

                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = compute_accuracy(cosine, target, topk=(1, 5))
                losses.update(loss.item(), target.size(0))
                top1.update(acc1[0], target.size(0))
                top5.update(acc5[0], target.size(0))

    info = {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg}
    return info

def extract(extractloader, model, device, args):
    # switch to evaluate mode
    model.eval()

    mean, std = np.load("{}/mean.npy".format(args.train_egs_dir)), np.load("{}/std.npy".format(args.train_egs_dir))
    num_fail, num_success = 0, 0
    utt2embedding = {}

    print("{} BATCHES IN TOTAL".format(len(extractloader)))
    with torch.no_grad():
        for i, (feats_list, vad_list) in enumerate(extractloader, 1):
            print("BATCH {}/{}".format(i, len(extractloader)))
            sys.stdout.flush()

            tmp_feats_file = "{}/tmp/feats.scp".format(args.output_dir)
            with open(tmp_feats_file, 'w') as fh:
                fh.write("".join(feats_list))
            tmp_vad_file = "{}/tmp/vad.scp".format(args.output_dir)
            with open(tmp_vad_file, 'w') as fh:
                fh.write("".join(vad_list))

            feats_ark = "ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{} ark:- | select-voiced-frames ark:- scp,s,cs:{} ark:- |".format(tmp_feats_file, tmp_vad_file)
            for uttname, mat in kaldi_io.read_mat_ark(feats_ark):
                assert mean is not None and std is not None
                mat = (mat - mean) / std
                utt_len = mat.shape[0]

                # Discard the utterance if utterance length is smaller than min_chunk_size
                if utt_len < args.min_chunk_size:
                    num_fail += 1
                    print("Warning: utterance {} too short ({} frames), skipping it".format(uttname, utt_len))
                    continue
                else:
                    x_vector = AverageMeter()
                    num_chunk = int(utt_len / args.max_chunk_size)
                    
                    if num_chunk > 0:
                        chunk_list = []
                        for j in range(num_chunk):
                            chunk_list.append(mat[j * args.max_chunk_size:(j + 1) * args.max_chunk_size])
                        chunk = np.array(chunk_list)
                        chunk = (torch.from_numpy(chunk)).to(device)
                        embedding_a, embedding_b = model(chunk)
                        if args.embedding_type == "a":
                            embedding = (embedding_a.cpu()).numpy()
                        elif args.embedding_type == "b":
                            embedding = (embedding_b.cpu()).numpy()
                        else:
                            raise ValueError("Embedding type not defined.")
                        embedding = np.mean(embedding, axis=0, keepdims=True)
                        x_vector.update(embedding, num_chunk * args.max_chunk_size)
                    
                    if not args.drop_last:
                        chunk = mat[num_chunk * args.max_chunk_size : utt_len]
                        if len(chunk) >= args.min_chunk_size:
                            chunk = np.expand_dims(chunk, axis=0)
                            chunk = (torch.from_numpy(chunk)).to(device)
                            embedding_a, embedding_b = model(chunk)
                            if args.embedding_type == "a":
                                embedding = (embedding_a.cpu()).numpy()
                            elif args.embedding_type == "b":
                                embedding = (embedding_b.cpu()).numpy()
                            else:
                                raise ValueError("Embedding type not defined.")
                            x_vector.update(embedding, utt_len - num_chunk * args.max_chunk_size)

                    if x_vector.count != 0:
                        x_vector_mean = x_vector.avg 
                        x_vector_mean = x_vector_mean.reshape((-1))
                        utt2embedding[uttname] = x_vector_mean
                        num_success += 1
                    else:
                        print("Warning: cannot extract embeddings for utterance {} ({} frames), skipping it".format(uttname, utt_len))
                        num_fail += 1

    utt_list = list(utt2embedding.keys())
    utt_list.sort()
    with open("{}/xvector_tmp.ark".format(args.output_dir), 'wb') as fh:
        for utt in utt_list:
            kaldi_io.write_vec_flt(fh, utt2embedding[utt], utt)

    print("{} utterances fail, {} utterances success".format(num_fail, num_success))
    return 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_accuracy(output, target, topk=(1,)):
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
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def extract_sliding_window(extract_dataset, model, device, args):
    # switch to evaluate mode
    model.eval()

    utt2embedding = {}

    with torch.no_grad():
        for uttname, feat in extract_dataset:
            num_frames = len(feat)
            num_batch = int(np.ceil(1.0 * num_frames / args.batch_size))
            pad_zero = np.zeros((args.half_window, feat.shape[1]))
            feat_pad = np.concatenate([pad_zero, feat, pad_zero], axis=0) 
            embedding_list = []
            for i in range(num_batch):
                start_idx = args.batch_size * i
                end_idx = args.batch_size * (i + 1) if i != num_batch - 1 else num_frames
                feat_batch = np.array([feat_pad[idx - args.half_window + args.half_window:idx + args.half_window + args.half_window, :] for idx in range(start_idx, end_idx)])
                feat_batch = torch.from_numpy(feat_batch).to(device).float()
                embedding_a, embedding_b = model(feat_batch)
                if args.embedding_type == "a":
                    embedding = (embedding_a.cpu()).numpy()
                elif args.embedding_type == "b":
                    embedding = (embedding_b.cpu()).numpy()
                else:
                    raise ValueError("Embedding type not defined.")
                embedding_list.append(embedding)
            utt_embedding = np.concatenate(embedding_list, axis=0)
            utt2embedding[uttname] = utt_embedding
            assert len(utt_embedding) == num_frames
            np.save("{}/{}.npy".format(args.output_dir, uttname), utt_embedding)
    return utt2embedding

# Ongoing work
#def extract_diarization(extractloader, model, device, args):
#    # switch to evaluate mode
#    model.eval()
#
#    utt2embedding = {}
#
#    print("{} BATCHES IN TOTAL".format(len(extractloader)))
#    with torch.no_grad():
#        for i, (feats_list, vad_list) in enumerate(extractloader, 1):
#
#            tmp_feats_file = "{}/tmp/feats.scp".format(args.output_dir)
#            with open(tmp_feats_file, 'w') as fh:
#                fh.write("".join(feats_list))
#            tmp_vad_file = "{}/tmp/vad.scp".format(args.output_dir)
#            with open(tmp_vad_file, 'w') as fh:
#                fh.write("".join(vad_list))
#
#            feats_ark = "ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{} ark:- | select-voiced-frames ark:- scp,s,cs:{} ark:- |".format(tmp_feats_file, tmp_vad_file)
#            for uttname, mat in kaldi_io.read_mat_ark(feats_ark):
#                assert mean is not None and std is not None
#                mat = (mat - mean) / std
#                utt_len = mat.shape[0]
#
#                # Discard the utterance if utterance length is smaller than min_chunk_size
#                if utt_len < args.min_chunk_size:
#                    num_fail += 1
#                    print("Warning: utterance {} too short ({} frames), skipping it".format(uttname, utt_len))
#                    continue
#                else:
#                    x_vector = AverageMeter()
#                    num_chunk = int(utt_len / args.max_chunk_size)
#                    
#                    if num_chunk > 0:
#                        chunk_list = []
#                        for j in range(num_chunk):
#                            chunk_list.append(mat[j * args.max_chunk_size:(j + 1) * args.max_chunk_size])
#                        chunk = np.array(chunk_list)
#                        chunk = (torch.from_numpy(chunk)).to(device)
#                        embedding_a, embedding_b = model(chunk)
#                        if args.embedding_type == "a":
#                            embedding = (embedding_a.cpu()).numpy()
#                        elif args.embedding_type == "b":
#                            embedding = (embedding_b.cpu()).numpy()
#                        else:
#                            raise ValueError("Embedding type not defined.")
#                        embedding = np.mean(embedding, axis=0, keepdims=True)
#                        x_vector.update(embedding, num_chunk * args.max_chunk_size)
#                    
#                    if not args.drop_last:
#                        chunk = mat[num_chunk * args.max_chunk_size : utt_len]
#                        if len(chunk) >= args.min_chunk_size:
#                            chunk = np.expand_dims(chunk, axis=0)
#                            chunk = (torch.from_numpy(chunk)).to(device)
#                            embedding_a, embedding_b = model(chunk)
#                            if args.embedding_type == "a":
#                                embedding = (embedding_a.cpu()).numpy()
#                            elif args.embedding_type == "b":
#                                embedding = (embedding_b.cpu()).numpy()
#                            else:
#                                raise ValueError("Embedding type not defined.")
#                            x_vector.update(embedding, utt_len - num_chunk * args.max_chunk_size)
#
#                    if x_vector.count != 0:
#                        x_vector_mean = x_vector.avg 
#                        x_vector_mean = x_vector_mean.reshape((-1))
#                        utt2embedding[uttname] = x_vector_mean
#                        num_success += 1
#                    else:
#                        print("Warning: cannot extract embeddings for utterance {} ({} frames), skipping it".format(uttname, utt_len))
#                        num_fail += 1
#
#    utt_list = list(utt2embedding.keys())
#    utt_list.sort()
#    with open("{}/xvector_tmp.ark".format(args.output_dir), 'wb') as fh:
#        for utt in utt_list:
#            kaldi_io.write_vec_flt(fh, utt2embedding[utt], utt)
#
#    print("{} utterances fail, {} utterances success".format(num_fail, num_success))
#    return 0

def save_checkpoint(state, model_filename):
    torch.save(state, model_filename)
    return 0

if __name__ == "__main__":
    output = torch.rand(10, 5)
    print("output", output)
    target = torch.from_numpy(np.array([2, 1, 0, 2, 4, 2, 3, 3, 1, 0]))
    print("target", target)
    compute_accuracy(output, target, topk=(1, 3))
