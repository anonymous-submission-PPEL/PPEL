import torch
import os
from utils.core_utils import _get_splits,_init_model, _init_loaders, _extract_survival_metadata, _init_loss_function, _summary
from utils.file_utils import _save_pkl


def _get_val_results(args,model,train_loader,val_loader,log_file,loss_fn):
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.omics_format, val_loader, loss_fn, all_survival)

    print('Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}'.format(
        val_cindex,
        val_cindex_ipcw,
        val_IBS,
        val_iauc
    ))
    log_file.write(
        'Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}\n'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))

    return results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss



def _val(datasets,cur,args,log_file):
    '''

        :param datasets: tuple
        :param cur: Int
        :param args: argspace.Namespace
        :param log_file: file
        :return:
        '''

    # ----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)

    # ----> initialize model
    model = _init_model(args)

    # ----> load params of model

    path = os.path.join(args.results_dir, "model_best_s{}.pth".format(cur))
    model.load_state_dict(torch.load(path), strict=True)
    print("Loaded model from {}".format(path))
    log_file.write("Loaded model from {}\n".format(path))
    
    # ----> init loss function
    loss_fn = _init_loss_function(args)

    # ----> initialize loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # ----> val
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _get_val_results(args, model, train_loader,
                                                                                                        val_loader, log_file, loss_fn)
    filename = os.path.join(args.results_dir, "split_{}_results_missing.pkl".format(cur))
    _save_pkl(filename, results_dict)
    print("Saved missing-genomics results to {}".format(filename))
    log_file.write("Saved missing-genomics results to {}\n".format(filename))

    return (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)
