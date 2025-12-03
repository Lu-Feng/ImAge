import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import network
from loss import loss_function
from dataloaders.GSVCities import get_GSVCities
from torch.cuda.amp import GradScaler,autocast

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

val_ds0 = datasets_ws.BaseDataset(args, args.eval_datasets_folder, "pitts30k", "val")
logging.info(f"Val set0: {val_ds0}")
val_ds1 = datasets_ws.BaseDataset(args, args.eval_datasets_folder, "msls", "val")
logging.info(f"Val set1: {val_ds1}")

#### Initialize model
model = network.VPRmodel(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

#### Print the number of model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
aggregator_params = sum(p.numel() for p in model.module.aggregator.parameters()) if model.module.aggregator else 0

print(f"The entire parameters: {total_params / 1e6:.2f}M")
print(f"The trainable parameters: {trainable_params / 1e6:.2f}M")
print(f"The aggregator parameters: {aggregator_params / 1e6:.2f}M")

#### Initialize agg tokens
if not args.aggregator:
    args.features_dim = 768
    if not args.resume:
        pretrained_model = network.get_backbone(args)
        if args.initialization_dataset == "msls_train":
            from initialize_agg_tokens import initialize_learnable_aggregation_tokens_centroids_msls_train, initialize_learnable_aggregation_tokens_centroids_L2N
            triplets_ds = datasets_ws.TripletsDataset(args, args.eval_datasets_folder, "msls", "train", args.negs_num_per_query)
            logging.info(f"Train query set: {triplets_ds}")
            triplets_ds.is_inference = True
            initial_centroids, initial_descriptors = initialize_learnable_aggregation_tokens_centroids_msls_train(args, triplets_ds, pretrained_model.to(args.device))
            centroids_L2N = initialize_learnable_aggregation_tokens_centroids_L2N(initial_centroids, initial_descriptors)
            model.module.learnable_aggregation_tokens = torch.nn.Parameter(torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0))

        elif args.initialization_dataset == "gsv_cities":
            from initialize_agg_tokens import initialize_learnable_aggregation_tokens_centroids_gsv, initialize_learnable_aggregation_tokens_centroids_L2N
            TRAIN_CITIES = [
            'Bangkok',
            'BuenosAires',
            'LosAngeles',
            'MexicoCity',
            'OSL', # refers to Oslo
            'Rome',
            'Barcelona',
            'Chicago',
            'Madrid',
            'Miami',
            'Phoenix',
            'TRT', # refers to Toronto
            'Boston',
            'Lisbon',
            'Medellin',
            'Minneapolis',
            'PRG', # refers to Prague
            'WashingtonDC',
            'Brussels',
            'London',
            'Melbourne',
            'Osaka',
            'PRS', # refers to Paris
            ]
            initial_dataset = get_GSVCities(image_size=(224, 224), cities=TRAIN_CITIES)
            initial_centroids, initial_descriptors = initialize_learnable_aggregation_tokens_centroids_gsv(args, initial_dataset, pretrained_model.to(args.device))
            centroids_L2N = initialize_learnable_aggregation_tokens_centroids_L2N(initial_centroids, initial_descriptors)
            model.module.learnable_aggregation_tokens = torch.nn.Parameter(torch.from_numpy(centroids_L2N).to(args.device).unsqueeze(0))
    args.features_dim = args.features_dim * args.num_learnable_aggregation_tokens

if args.aggregator in ["netvlad"]:  # If using NetVLAD layer, initialize it
    args.features_dim = 768
    if not args.resume:
        triplets_ds = datasets_ws.TripletsDataset(args, args.eval_datasets_folder, "msls", "train", args.negs_num_per_query)
        logging.info(f"Train query set: {triplets_ds}")
        triplets_ds.is_inference = True
        pretrained_model = network.get_backbone(args)
        model.module.agg.initialize_netvlad_layer(args, triplets_ds, pretrained_model.to(args.device)) 
    args.features_dim = args.features_dim * 8

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r1_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
else:
    best_r1_r5 = start_epoch_num = not_improved_num = 0

if args.training_dataset == "gsv_cities":
    TRAIN_CITIES = [
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]
else:
    TRAIN_CITIES = [
        "SFXL",
        'Bangkok',
        'BuenosAires',
        'LosAngeles',
        'MexicoCity',
        'OSL', # refers to Oslo
        'Rome',
        'Barcelona',
        'Chicago',
        'Madrid',
        'Miami',
        'Phoenix',
        'TRT', # refers to Toronto
        'Boston',
        'Lisbon',
        'Medellin',
        'Minneapolis',
        'PRG', # refers to Prague
        'WashingtonDC',
        'Brussels',
        'London',
        'Melbourne',
        'Osaka',
        'PRS', # refers to Paris
    ]

    citylist = [
        "Trondheim",
        "Amsterdam",
        "Helsinki",
        "Tokyo",
        "Toronto",
        "Saopaulo",
        "Moscow",
        "Zurich",
        "Paris",
        "Budapest",
        "Austin",
        "Berlin",
        "Ottawa",
        "Goa",
        "Amman",
        "Nairobi",
        "Manila",
        "bangkok",
        "boston",
        "london",
        "melbourne",
        "phoenix",
        "Pitts30k"
    ]
    newcitylist = []
    for i in range(18):
        for cityname in citylist:
            if i==17 and (cityname == "Amman" or cityname == "Nairobi"):
                continue
            else:
                newcitylist.append(cityname+str(i))
    TRAIN_CITIES = TRAIN_CITIES + newcitylist

train_dataset = get_GSVCities(image_size=(224, 224), cities=TRAIN_CITIES)
train_loader_config = {
    'batch_size': args.train_batch_size,
    'num_workers': args.num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': False}

#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
scaler = GradScaler()
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
          
    model = model.train()
    epoch_losses=[]
    for images, place_id in tqdm(ds):       
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)

        optimizer.zero_grad()
        with autocast():
            descriptors = model(images.to(args.device)).cuda()
            loss = loss_function(descriptors, labels) # Call the loss_function we defined above
            del descriptors

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        del loss

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls0, recalls_str0 = test.test(args, val_ds0, model)
    logging.info(f"Recalls on val set0 {val_ds0}: {recalls_str0}")
    recalls1, recalls_str1 = test.test(args, val_ds1, model)
    logging.info(f"Recalls on val set1 {val_ds1}: {recalls_str1}")
    is_best = recalls1[0] + recalls1[1] > best_r1_r5

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls1, "best_r1_r5": best_r1_r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")
    
    # If recall@1 + recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best (R@1 + R@5) = {best_r1_r5:.1f}, current (R@1 + R@5) = {(recalls1[0]+recalls1[1]):.1f}")
        best_r1_r5 = (recalls1[0]+recalls1[1])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best (R@1 + R@5) = {best_r1_r5:.1f}, current (R@1 + R@5) = {(recalls1[0]+recalls1[1]):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break

logging.info(f"Best (R@1 + R@5): {best_r1_r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

# load the best model for testing
logging.info("Test *best* model on test set")
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"), weights_only=False)["model_state_dict"]
model.load_state_dict(best_model_state_dict)
recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")