import yaml
import random
from dataset import LabelEmbeddingData
from utils.load_data import *
from utils.metrics import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.tppred import TPMLC, TPMLC_single
from torch.optim import AdamW
from utils.sampling import Sampler
from utils.visualization import *

class Model ():

    def __init__(self, args):
        """
        initialize the hyper-parameters
        """

        self.args = args

        # Load constants
        with open(args.cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # self.model = cfg['model']
        self.d_fea = cfg['d_fea']
        self.max_len = cfg['max_len']
        self.pts = cfg['pts']

        # network parameters
        self.seed = args.seed
        self.d_model = args.dm
        self.n_heads = args.nh
        self.n_layers_enc = args.nle
        self.n_layers_dec = args.nld
        self.drop = args.drop

        # shared training parameters
        self.batch_size = args.b

        # jointly training parameters
        self.epochs = args.e
        self.lr = args.lr
        self.w = args.w
        self.model_path = args.pth

        # retraining parameters
        self.re_method = args.s
        self.re_epochs = args.e2
        self.re_lr = args.lr2
        self.re_w = args.w2
        self.re_model_path = args.pth2

        # other parameters
        self.dataset_dir = args.src
        self.task_tag = ""
        self.result_folder = args.result_folder

        # If training all layers, the trained model will saved to self.model_path.
        # If retraining the classifiers, method will load the model self.model_path,
        # and save the retrained model to self.re_model_path

        self.names = [pt[:-4] for pt in self.pts]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_class = len(self.pts)

        self.pt2idx = {}
        for i, pt in enumerate(self.names):
            self.pt2idx[pt] = i

        self.set_seed(seed=self.seed)

    def set_task(self, task=None):

        self.task_tag = task + "_" if task is not None else ""


    def train_epoch(self, model, optimizer, criterion, train_dataloder, val_dataloder, target = None):

        model.train()
        train_losses = []

        for i, data in enumerate(train_dataloder):
            optimizer.zero_grad()

            X, y, masks, label_input = data

            out, _, _, _ = model(X, masks, label_input)
            # out = model(X, masks)

            if target == None:
                loss = criterion(out, y.float())
            else:
                loss = criterion(out[:, target], y.float()[:, target])

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validating the model after each step
        model.eval()
        val_losses = []
        y_pred = []
        y_true = []

        with torch.no_grad():

            for i, data in enumerate(val_dataloder):
                X, y, masks, label_input = data
                out, _, _, _ = model(X, masks, label_input)
                # out = model(X, masks)

                if target == None:
                    loss = criterion(out, y.float())
                else:
                    loss = criterion(out[:, target], y.float()[:, target])

                val_losses.append(loss.item())
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

        # print("Epoch {}, train loss = {}, validation loss = {}".
        #       format(epoch, np.mean(train_losses), np.mean(val_losses)))

        # optimized by validation loss

        return float(np.mean(train_losses)), float(np.mean(val_losses)), y_true, y_pred

    def retrain_classifiers(self):
        """
        Retraining each specific classifier layer
        """
        print(f"Retraining classifier layers, task: {self.task_tag}")

        checkpoint = torch.load(self.model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load training and validation datasets
        train_feas, train_labels, train_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'train'), True, *self.pts)

        val_feas, val_labels, val_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'val'), True, *self.pts)
        val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device),
                                   batch_size=self.batch_size, shuffle=False)

        print('dataset',os.path.join(self.dataset_dir, 'train'))

        criterion = torch.nn.BCELoss()

        # Reinitialize classifiers
        self.reset_classifiers(model)

        best_model = None

        for i, fn in enumerate(self.pts):
            name = fn.split('.')[0]
            print("Retrain classifier", name)

            # Freeze the model layers except the i-th classifier
            self.freeze_layers(model, i)

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), self.re_lr, weight_decay=self.re_w)

            min_loss = 10000
            max_f1 = 0

            for epoch in range(self.re_epochs):

                sampler = Sampler(train_labels, method=self.re_method, lam=epoch / (self.re_epochs))
                sampler.set_target(i)

                train_dataloader = DataLoader(
                    dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device),
                    batch_size=self.batch_size, sampler=sampler)

                train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloader, val_dataloder, target=i)

                print("Epoch {}, train loss = {}, validation loss = {}".
                      format(epoch, train_loss, val_loss))

                if val_loss <= min_loss:
                
                    print('update loss', val_loss)
                    best_model = model
                    min_loss = val_loss
                
                    self.evaluation(np.array(y_true), np.array(y_pred), 'val')

        if self.re_model_path is not None:
            self.save_model(best_model, self.re_model_path)


    def train_all(self):

        print("Training all layers2, task name: ", self.task_tag)

        # Load training and validation features
        train_feas, train_labels, train_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'train'), True, *self.pts)

        val_feas, val_labels, val_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'val'), True, *self.pts)

        train_dataloder = DataLoader(dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device),
                                     batch_size=self.batch_size, shuffle=True)
        val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device),
                                   batch_size=self.batch_size, shuffle=False)
        
        # phase 1
        model = TPMLC_single(self.d_fea, self.n_class, self.max_len, self.d_model, device=self.device, nhead=self.n_heads,
                      n_enc_layers=self.n_layers_enc, n_dec_layers=self.n_layers_dec, dropout=self.drop).to(self.device)

        criterion = torch.nn.BCELoss()
        optimizer = AdamW(model.parameters(), self.lr, weight_decay=self.w)

        # optimized values
        min_loss = 1000
        best_model = None

        for epoch in range(self.epochs):

            train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloder,
                                                                    val_dataloder)

            print("Epoch {}, train loss = {}, validation loss = {}".
                  format(epoch, train_loss, val_loss))

            # optimized by validation loss
            if val_loss <= min_loss:
                best_model = model
                min_loss = val_loss
                self.evaluation(np.array(y_true), np.array(y_pred), 'val')

        # save the model with min validation loss
        sv = self.model_path[:-4] + '_single.pth'
        if self.model_path is not None:
            self.save_model(best_model, sv)

        # phase 2

        checkpoint = torch.load(sv)
        rp_model = checkpoint['model']
        rp_model.load_state_dict(checkpoint['model_state_dict'])

        model = TPMLC(self.d_fea, self.n_class, self.max_len, self.d_model, device=self.device, nhead=self.n_heads,
                             n_enc_layers=self.n_layers_enc, n_dec_layers=self.n_layers_dec, dropout=self.drop).to(
            self.device)
        model_dict = model.state_dict()

        st = {} 
        for k, v in rp_model.named_parameters():
            if k.startswith('rp') and k in model_dict.keys():
                st[k] = v

        model_dict.update(st)
        model.load_state_dict(model_dict)


        # optimized values
        min_loss = 1000
        best_model = None

        self.freeze_layers_dec(model)

        criterion = torch.nn.BCELoss()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), self.re_lr, weight_decay=self.w)

        for epoch in range(self.re_epochs):

            train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloder,
                                                                    val_dataloder)

            print("Epoch {}, train loss = {}, validation loss = {}".
                  format(epoch, train_loss, val_loss))

            # optimized by validation loss
            if val_loss <= min_loss:
                best_model = model
                min_loss = val_loss
                self.evaluation(np.array(y_true), np.array(y_pred), 'val')

        # save the model with min validation loss
        if self.model_path is not None:
            self.save_model(best_model, self.model_path)

            
    def independent_test(self, pth=None):
        """
        Independent test
        """
        model_path = pth if pth is not None else self.model_path

        print(f"Independent test{self.task_tag}, model path: {model_path}")

        # Load model
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load independent test dataset
        test_feas, test_labels, test_pad_masks, test_seqs = load_features(os.path.join(self.dataset_dir, 'test'), True, *self.pts)
        test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device),
                                    batch_size=self.batch_size, shuffle=True)

        # Predict
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for i, data in enumerate(test_dataloder):
                X, y, masks, label_input = data
                out, atts_x, atts_tgt, atts_cross = model(X, masks, label_input)
                # out = model(X, masks)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

        self.evaluation(np.array(y_true), np.array(y_pred), 'test')


    def evaluation(self, y_true, y_pred, tag='val'):
        """
        Evaluate the predictive performance
        """
        binary_metrics(y_pred, y_true, self.names, 0.5,
                       f'{self.result_folder}/{self.task_tag}{tag}_binary.csv', show=False)
        instances_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                                  f'{self.result_folder}/{self.task_tag}{tag}_sample.csv', show=False)
        label_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                              f'{self.result_folder}/{self.task_tag}{tag}_label.csv', show=False)

    def freeze_layers(self, model, i):
        """
        Freeze the specific classifier layer i
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                if name.split('.')[1] == str(i):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    def freeze_layers_dec(self, model):
        """
        Freeze the decoder classifier layers
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                param.requires_grad = True
            else:
                if name.startswith('rp.decoder_layers') or name.startswith('rp.label'):
                    print("freeze", name)
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def reset_classifiers(self, model):
        """
        Reinitialize the classifier layers
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def save_model(self, model, path):

        torch.save({
            'model': model,
            'model_state_dict': model.state_dict(),
            'pt_order': self.names,
            'args': self.args
        }, f'{path}')


    def set_seed(self, seed=123):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


    def visualization(self, idx=0, pt='AMP', pth=None, title="TPpred-MLC"):

        model_path = pth if pth is not None else self.model_path

        print(f"Independent test{self.task_tag}, model path: {model_path}")

        # Load model
        checkpoint = torch.load(model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load independent test dataset
        test_feas, test_labels, test_pad_masks, test_seqs = load_features(os.path.join(self.dataset_dir, 'test'), True,
                                                                          *self.pts)
        test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device),
                                    batch_size=self.batch_size, shuffle=False)

        hooks_x = Hooks()
        hooks_y = Hooks()
        hooks_cls = Hooks()
        classifiers = []
        lem = None

        for name, module in model.named_children():
            if name == 'rp':
                for child_name, child_module in module.named_children():
                    if child_name == 'encoder_layers':
                        child_module[-1].register_forward_hook(hook=hooks_x.hook)
                    if child_name == 'decoder_layers':
                        child_module[-1].register_forward_hook(hook=hooks_y.hook)
                    if child_name == 'label_embedding':
                        lem = child_module.weight.cpu().detach().numpy()

            elif name == 'fcs':
                for i in range(len(module)):
                    classifiers.append(module[i][0])    # Linear, Sigmoid
                    module[i].register_forward_hook(hook=hooks_cls.hook_cls)

        # Predict
        model.eval()
        y_pred = []
        y_true = []

        feature_x = []
        feature_y = []
        atts_x = []
        atts_y = []
        atts_cross = []

        with torch.no_grad():
            for i, data in enumerate(test_dataloder):
                X, y, masks, label_input = data
                out, att_x, att_y, att_cross = model(X, masks, label_input)
                # out = model(X, masks)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

                _, embed_x = hooks_x.get_data()
                _, embed_y = hooks_y.get_data()

                feature_x.append(embed_x)
                feature_y.append(embed_y)

                att_nx = np.array([ax.cpu().detach().numpy() for ax in att_x])
                atts_x.append(att_nx)
                att_ny = np.array([ay.cpu().detach().numpy() for ay in att_y])
                atts_y.append(att_ny)
                att_cross = np.array([ac.cpu().numpy() for ac in att_cross])
                atts_cross.append(att_cross)

        df = binary_metrics(np.array(y_pred), np.array(y_true), self.names)

        cls_in, cls_out = hooks_cls.get_data()

        feature_x = np.concatenate(feature_x, axis=0)
        feature_y = np.concatenate(feature_y, axis=0)
        atts_x = np.concatenate(atts_x, axis=1)
        atts_y = np.concatenate(atts_y, axis=1)
        atts_cross = np.concatenate(atts_cross, axis=1)
        print(atts_x.shape)
        print(atts_y.shape)
        print(atts_cross.shape)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        all_true = []
        all_true_m = []
        y_pred_cls = np.zeros_like(y_pred, dtype=np.int)
        y_pred_cls[y_pred >= 0.5] = 1  # 预测类别
        for i in range(len(y_true)):
            if np.all(y_true[i] == y_pred_cls[i]):
                all_true.append(i)
                if np.sum(y_true[i]) > 1 and y_true[i][self.pt2idx['ABP']] == 1 and y_true[i][self.pt2idx['AMP']] == 1:
                    all_true_m.append(i)

        print("all true", all_true_m)

        print('label', y_true[idx])
        masks = [np.sum(m) for m in ~test_pad_masks]
        print("pred", y_pred[idx].round(3))
        # attention : 层数, 样本数 ...
        
        # visualize_attention(atts_y[5][idx], xlabel=self.names, ylabel=self.names)
        visualize_attention(atts_x[-1][idx][:masks[idx],:masks[idx]], xlabel=[r for r in test_seqs[idx]], ylabel=[r for r in test_seqs[idx]], save="xx.png")
        visualize_attention(atts_cross[-1][idx][:,:masks[idx]], xlabel=[r for r in test_seqs[idx]], ylabel=self.names, save="xy.png")
        visualize_attention(atts_y[-1][idx][:3,:], xlabel=self.names, ylabel=self.names[:3], save="yy.png")

        visualize_attention_avg(atts_y[-1], xlabel=self.names, ylabel=self.names, save="yy_all.png")     
        
        visualize_func_residue_attention(atts_cross[-1], funcs=self.names, seqs=test_seqs, save="xy_all.png")


