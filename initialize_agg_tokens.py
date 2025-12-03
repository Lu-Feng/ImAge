from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import torch
import math 
import numpy as np
from tqdm import tqdm
import faiss
import torch.nn.functional as F
import logging

def initialize_learnable_aggregation_tokens_centroids_gsv(args, cluster_ds, backbone):
    descriptors_num = 2000000
    descs_num_per_image = 100
    images_num = math.ceil(descriptors_num / descs_num_per_image)
    random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
    random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                           batch_size=args.infer_batch_size, sampler=random_sampler)
    with torch.no_grad():
        backbone = backbone.eval()
        logging.debug("Extracting features to initialize aggregation tokens!")
        descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
        for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
            inputs = inputs.to(args.device)
            BS, N, ch, h, w = inputs.shape
            inputs = inputs.view(BS*N, ch, h, w)
            if args.backbone.startswith("dinov2"):
                outputs = backbone(inputs)
                B, P, D = outputs["x_prenorm"].shape
                W = H = int(math.sqrt(P))
                outputs = outputs["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
            # elif args.backbone.startswith("vit"):
            #     x = backbone.embeddings(pixel_values=inputs, interpolate_pos_encoding=True)
            #     x = backbone.encoder(x)
            #     outputs = backbone.layernorm(x[0])
            #     B, P, D = outputs.shape
            #     W = H = int(math.sqrt(P))
            #     outputs = outputs[:, 1:, :].view(B, W, H, D).permute(0, 3, 1, 2)
            # elif args.backbone.startswith("clip"):
            #     x = backbone._embeds(inputs)
            #     outputs = backbone.transformer(x)
            #     B, P, D = outputs.shape
            #     W = H = int(math.sqrt(P-1))
            #     outputs = outputs[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2) 
            norm_outputs = F.normalize(outputs, p=2, dim=1)
            image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
            image_descriptors = image_descriptors[::4]
            image_descriptors = image_descriptors.cpu().numpy()
            batchix = iteration * args.infer_batch_size * descs_num_per_image
            for ix in range(image_descriptors.shape[0]):
                sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                startix = batchix + ix * descs_num_per_image
                descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
    kmeans = faiss.Kmeans(args.features_dim, args.num_learnable_aggregation_tokens, niter=100, verbose=False)
    kmeans.train(descriptors)
    logging.debug(f"The shape of cluster centers: {kmeans.centroids.shape}")
    return kmeans.centroids, descriptors

def initialize_learnable_aggregation_tokens_centroids_msls_train(args, cluster_ds, backbone):
    descriptors_num = 500000
    descs_num_per_image = 100
    images_num = math.ceil(descriptors_num / descs_num_per_image)
    random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
    random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                           batch_size=args.infer_batch_size, sampler=random_sampler)
    with torch.no_grad():
        backbone = backbone.eval()
        logging.debug("Extracting features to initialize aggregation tokens!")
        descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
        for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
            inputs = inputs.to(args.device)
            if args.backbone.startswith("dinov2"):
                outputs = backbone(inputs)
                B, P, D = outputs["x_prenorm"].shape
                W = H = int(math.sqrt(P))
                outputs = outputs["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
            # elif args.backbone.startswith("vit"):
            #     x = backbone.embeddings(pixel_values=inputs, interpolate_pos_encoding=True)
            #     x = backbone.encoder(x)
            #     outputs = backbone.layernorm(x[0])
            #     B, P, D = outputs.shape
            #     W = H = int(math.sqrt(P))
            #     outputs = outputs[:, 1:, :].view(B, W, H, D).permute(0, 3, 1, 2)
            # elif args.backbone.startswith("clip"):
            #     x = backbone._embeds(inputs)
            #     outputs = backbone.transformer(x)
            #     B, P, D = outputs.shape
            #     W = H = int(math.sqrt(P-1))
            #     outputs = outputs[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2) 
            norm_outputs = F.normalize(outputs, p=2, dim=1)
            image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
            image_descriptors = image_descriptors.cpu().numpy()
            batchix = iteration * args.infer_batch_size * descs_num_per_image
            for ix in range(image_descriptors.shape[0]):
                sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                startix = batchix + ix * descs_num_per_image
                descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
    kmeans = faiss.Kmeans(args.features_dim, args.num_learnable_aggregation_tokens, niter=100, verbose=False)
    kmeans.train(descriptors)
    logging.debug(f"The shapes of cluster centers: {kmeans.centroids.shape}")
    return kmeans.centroids, descriptors

def initialize_learnable_aggregation_tokens_centroids_L2N(centroids, descriptors):
    centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    dots = np.dot(centroids_assign, descriptors.T)
    dots.sort(0)
    dots = dots[::-1, :]  # sort, descending
    if dots.shape[0] > 1:
        alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
    else:
        alpha = (-np.log(0.01) / np.mean(dots[0, :])).item()
    centroids_L2N = alpha * centroids_assign
    return centroids_L2N