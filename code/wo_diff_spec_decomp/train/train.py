import torch.optim as optim
from torch.optim import lr_scheduler
from prefetch_dataloader import *
from model import *
from loss import *
from gen_hdf5 import *
from dataset import *
from util import *
from metric import *
import time
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inDir", "-i", required=True)
parser.add_argument("--datasetDir", "-d", required=True)
parser.add_argument("--outDir", "-o", required=True)
# training and dataset parameters
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--numSavedImgs", type=int, default=6)
parser.add_argument("--lrG", type=float, default=1e-4)
parser.add_argument("--lrD", type=float, default=1e-4)
parser.add_argument("--lrGamma", type=float, default=0.5)
parser.add_argument("--lrMilestone", type=float, default=3)
parser.add_argument("--seed", type=int, default=990819)
parser.add_argument("--saveInterval", type=int, default=1)
parser.add_argument("--GUpdateIters", type=int, default=1)
parser.add_argument("--patchSize", type=int, default=128)
parser.add_argument("--numPatches", type=int, default=400)
parser.add_argument("--l1LossW", type=float, default=1.0)
parser.add_argument("--ganLossW", type=float, default=5e-3)
parser.add_argument("--gpLossW", type=float, default=10)

parser.add_argument("--isLoad", dest="loadModel", action="store_true")
parser.set_defaults(loadModel=False)
parser.add_argument("--modelPath", type=str, default=None)
# model parameters
parser.add_argument("--blockSize", type=int, default=8)
parser.add_argument("--haloSize", type=int, default=3)
parser.add_argument("--numHeads", type=int, default=4)
parser.add_argument("--numSA", type=int, default=5)
parser.add_argument("--inCh", type=int, default=3)
parser.add_argument("--auxInCh", type=int, default=7)
parser.add_argument("--baseCh", type=int, default=256)
parser.add_argument("--numGradientCheckpoint", type=int, default=0)  # how many Trans blocks with gradient checkpoint
args, unknown = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permutation = [0, 3, 1, 2]
data_ratio = (0.95, 0.05)


def train_SANet(args, train_dataloader, train_num_samples, val_dataloader, val_num_samples, root_save_path):
    print("\t-Creating AFGSANet")
    G = AFGSANet(args.inCh, args.auxInCh, args.baseCh, num_sa=args.numSA, block_size=args.blockSize,
                 halo_size=args.haloSize, num_heads=args.numHeads, num_gcp=args.numGradientCheckpoint).to(device)
    D = DiscriminatorVGG128(3, 64).to(device)
    if args.loadModel:
        G.load_state_dict(torch.load(os.path.join(args.modelPath, 'G.pt')))
        D.load_state_dict(torch.load(os.path.join(args.modelPath, 'D.pt')))
    print_model_structure(G)
    print_model_structure(D)

    l1_loss = L1ReconstructionLoss().to(device)
    gan_loss = GANLoss('wgan').to(device)
    gp_loss = GradientPenaltyLoss(device).to(device)

    milestones = [i * args.lrMilestone - 1 for i in range(1, args.epochs//args.lrMilestone)]
    optimizer_generator = optim.Adam(G.parameters(), lr=args.lrG, betas=(0.9, 0.999), eps=1e-8)
    scheduler_generator = lr_scheduler.MultiStepLR(optimizer_generator, milestones=milestones, gamma=0.5)
    optimizer_discriminator = optim.Adam(D.parameters(), lr=args.lrD, betas=(0.9, 0.999), eps=1e-8)
    scheduler_discriminator = lr_scheduler.MultiStepLR(optimizer_discriminator, milestones=milestones, gamma=0.5)

    accumulated_generator_loss = 0
    accumulated_discriminator_loss = 0
    total_iteraions = math.ceil(train_num_samples / args.batchSize)
    save_img_interval = val_num_samples // args.numSavedImgs

    print("\t-Start training")
    for epoch in range(args.epochs):
        start = time.time()
        for i_batch, batch_sample in enumerate(train_dataloader):
            aux_features = batch_sample['aux']
            aux_features[:, :, :, :3] = torch.FloatTensor(preprocess_normal(aux_features[:, :, :, :3]))  # normal is not yet preprocessed
            aux_features = aux_features.permute(permutation).to(device)
            noisy = batch_sample['noisy']
            noisy = preprocess_specular(noisy)
            noisy = noisy.permute(permutation).to(device)
            gt = batch_sample['gt']
            gt = preprocess_specular(gt)
            gt = gt.permute(permutation).to(device)

            end_io = time.time()
            if i_batch != 0:
                io_took = end_io - end
            else:
                io_took = end_io - start

            output = G(noisy, aux_features)

            # train discriminator
            optimizer_discriminator.zero_grad()
            pred_d_fake = D(output.detach())
            pred_d_real = D(gt)
            try:
                loss_d_real = gan_loss(pred_d_real, True)
                loss_d_fake = gan_loss(pred_d_fake, False)
                loss_gp = gp_loss(D, gt, output.detach())
            except:
                break
            discriminator_loss = (loss_d_fake + loss_d_real) / 2 + args.gpLossW * loss_gp
            discriminator_loss.backward()
            optimizer_discriminator.step()
            accumulated_discriminator_loss += discriminator_loss.item() / args.batchSize

            # train generator
            optimizer_generator.zero_grad()
            pred_g_fake = D(output)
            try:
                loss_g_fake = gan_loss(pred_g_fake, True)
                loss_l1 = l1_loss(output, gt)
            except:
                break
            generator_loss = args.ganLossW * loss_g_fake + args.l1LossW * loss_l1
            generator_loss.backward()
            optimizer_generator.step()
            accumulated_generator_loss += generator_loss.item() / args.batchSize

            if i_batch == 0:
                iter_took = time.time() - start
            else:
                iter_took = time.time() - end
            end = time.time()
            print("\r\t-Epoch: %d \tTook: %f sec \tIteration: %d/%d \tIter Took: %f sec \tI/O Took: %f sec \tG Loss: %f \tD Loss: %f" %
                  (epoch + 1, end - start, i_batch + 1, total_iteraions, iter_took, io_took,
                   accumulated_generator_loss/(i_batch+1), accumulated_discriminator_loss/(i_batch+1)), end='')

        end = time.time()
        print("\r\t-Epoch: %d \tG loss: %f \tD Loss: %f \tTook: %d seconds" %
              (epoch + 1, accumulated_generator_loss/(i_batch+1), accumulated_discriminator_loss/(i_batch+1),
               end - start))
        # save loss values
        with open(os.path.join(root_save_path, "train_loss.txt"), 'a') as f:
            f.write("Epoch: %d \tG loss: %f \tD Loss: %f\n" % (epoch + 1, accumulated_generator_loss/(i_batch+1),
                                                               accumulated_discriminator_loss/(i_batch+1)))

        scheduler_discriminator.step()
        scheduler_generator.step()
        accumulated_generator_loss = 0
        accumulated_discriminator_loss = 0

        # validate and save model, example images
        if epoch % args.saveInterval == 0:
            current_save_path = create_folder(os.path.join(root_save_path, 'model_epoch%d' % (epoch + 1)))
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_mrse = 0.0
            start = time.time()
            with torch.no_grad():
                G.eval()
                # save model
                torch.save(G.state_dict(), os.path.join(current_save_path, "G.pt"))
                torch.save(D.state_dict(), os.path.join(current_save_path, "D.pt"))

                for i_batch, batch_sample in enumerate(val_dataloader):
                    aux_features = batch_sample['aux']
                    aux_features[:, :, :, :3] = torch.FloatTensor(preprocess_normal(aux_features[:, :, :, :3]))  # normal is not yet preprocessed
                    aux_features = aux_features.permute(permutation).to(device)
                    noisy = batch_sample['noisy']
                    noisy = preprocess_specular(noisy)
                    noisy = noisy.permute(permutation).to(device)
                    gt = batch_sample['gt']
                    gt = gt.permute(permutation)

                    output = G(noisy, aux_features)

                    # transfer to image
                    output_c_n = postprocess_specular(output.cpu().numpy())
                    gt_c_n = gt.numpy()
                    noisy_c_n_255 = tensor2img(noisy.cpu().numpy(), post_spec=True)
                    output_c_n_255 = tensor2img(output.cpu().numpy(), post_spec=True)
                    gt_c_n_255 = tensor2img(gt.cpu().numpy())

                    # save image
                    if i_batch % save_img_interval == 0:
                        save_img_group(current_save_path, i_batch, noisy_c_n_255.copy(), output_c_n_255.copy(),
                                       gt_c_n_255.copy())

                    # calculate mrse
                    avg_mrse += calculate_rmse(output_c_n.copy(), gt_c_n.copy())
                    # calculate psnr
                    avg_psnr += calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
                    # calculate ssim
                    avg_ssim += calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())

                    end = time.time()
                    print("\r\t-Validation: %d \tTook: %f seconds \tIteration: %d/%d" %
                          (epoch + 1, end - start, i_batch + 1, val_num_samples), end='')
                G.train()

                avg_mrse /= val_num_samples
                avg_psnr /= val_num_samples
                avg_ssim /= val_num_samples
                print("\r\t-Validation: %d \tTook: %d seconds \tAvg MRSE: %f \tAvg PSNR: %f \tAvg 1-SSIM: %f" %
                      (epoch + 1, end - start, avg_mrse, avg_psnr, 1-avg_ssim))
                # save evaluation results
                with open(os.path.join(root_save_path, "evaluation.txt"), 'a') as f:
                    f.write("Validation: %d \tAvg MRSE: %f \tAvg PSNR: %f \tAvg 1-SSIM: %f\n" %
                            (epoch + 1, avg_mrse, avg_psnr, 1-avg_ssim))


def train(args, data_ratio):
    train_save_path = os.path.join(args.datasetDir, "train.h5")
    val_save_path = os.path.join(args.datasetDir, "val.h5")
    exist = True
    for path in [train_save_path, val_save_path]:
        if not os.path.exists(path):
            exist = False
    if not exist:
        constructor = Hdf5Constructor(args.inDir, args.datasetDir, args.patchSize, args.numPatches, args.seed,
                                      data_ratio)
        constructor.construct_hdf5()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    train_dataset = Dataset(train_save_path)
    train_num_samples = len(train_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=7,
    #                               pin_memory=True)  # original
    train_dataloader = DataLoaderX(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=7,
                                   pin_memory=True)  # prefetch

    val_dataset = Dataset(val_save_path)
    val_num_samples = len(val_dataset)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=shuffle, num_workers=1, pin_memory=True)
    val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=False, num_workers=7, pin_memory=True)

    root_save_path = create_folder(os.path.join(args.outDir, 'AFGSA'), still_create=True)  # path to save model, imgs

    train_SANet(args, train_dataloader, train_num_samples, val_dataloader, val_num_samples, root_save_path)
    print("Finish training!")


if __name__ == "__main__":
    # create folders
    create_folder(args.outDir)
    create_folder(args.datasetDir)

    train(args, data_ratio)
