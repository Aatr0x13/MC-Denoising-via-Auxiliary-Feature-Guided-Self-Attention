from prefetch_dataloader import *
from model import *
from dataset import *
from util import *
from metric import *
from preprocessing import *
import time
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inDir", "-i", required=True)
parser.add_argument("--outDir", "-o", required=True)
# testing parameters
parser.add_argument("--numSavedImgs", type=int, default=6)
# 0 for all/ 4, 8, 16, 32, 128, 256, 512, 1024
parser.add_argument("--datasetSPP", type=int, choices=[0, 4, 8, 16, 32, 128, 256, 512, 1024], default=0)
# model parameters (fixed)
parser.add_argument("--blockSize", type=int, default=8)
parser.add_argument("--haloSize", type=int, default=3)
parser.add_argument("--numHeads", type=int, default=4)
parser.add_argument("--numSA", type=int, default=5)
parser.add_argument("--inCh", type=int, default=3)
parser.add_argument("--auxInCh", type=int, default=7)
parser.add_argument("--baseCh", type=int, default=256)
parser.add_argument("--batchSize", type=int, default=1)
parser.add_argument("--numGradientCheckpoint", type=int, default=0)  # how many Trans blocks with gradient checkpoint
args, unknown = parser.parse_known_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permutation = [0, 3, 1, 2]
spp = [4, 8, 16, 32, 128, 256, 512, 1024]


def test_AFGSA(args, test_dataloader, test_num_samples, save_path):
    G = AFGSANet(args.inCh, args.auxInCh, args.baseCh, num_sa=args.numSA, block_size=args.blockSize,
                 halo_size=args.haloSize, num_heads=args.numHeads, num_gcp=args.numGradientCheckpoint).to(device)
    G.load_state_dict(torch.load(r'..\..\..\models\wo_diff_spec_decomp\G_ours.pt'))
    G.eval()

    save_img_interval = test_num_samples // args.numSavedImgs
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_rmse = 0.0
    nan_count = 0
    with torch.no_grad():
        start = time.time()
        for i_batch, batch_sample in enumerate(test_dataloader):
            aux_features = batch_sample['aux']
            aux_features[:, :, :, :3] = torch.FloatTensor(preprocess_normal(aux_features[:, :, :, :3]))  # normal is not yet preprocessed
            aux_features = aux_features.permute(permutation).to(device)

            noisy = batch_sample['noisy']
            noisy = preprocess_specular(noisy)
            noisy = noisy.permute(permutation).to(device)

            gt = batch_sample['gt']
            gt = gt.permute(permutation)

            output = G(noisy, aux_features)
            output_c_n = postprocess_specular(output.cpu().numpy()[0])
            gt_c_n = gt.numpy()[0]

            noisy_c_n_255 = tensor2img(noisy.cpu().numpy()[0], post_spec=True)
            output_c_n_255 = tensor2img(output.cpu().numpy()[0], post_spec=True)
            gt_c_n_255 = tensor2img(gt.cpu().numpy()[0])

            # save image
            if i_batch % save_img_interval == 0:
                save_img_group(save_path, i_batch, noisy_c_n_255.copy(), output_c_n_255.copy(), gt_c_n_255.copy())

            # rmse: output after post-processing, without tone mapping and * 255 (use postprocess_specular/ postprocess_diffuse)
            # psnr, ssim: output after post-processing, tone mapping and * 255 (use tensor2img)
            rmse = calculate_rmse(np.transpose(output_c_n.copy(), (1, 2, 0)), np.transpose(gt_c_n.copy(), (1, 2, 0)))
            avg_psnr += calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
            avg_ssim += calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())
            if np.isnan(rmse):
                nan_count += 1
            else:
                avg_rmse += rmse

            end = time.time()
            print("\r\t-Test: Took: %f seconds \tIteration: %d/%d" % (end - start, i_batch + 1, test_num_samples),
                  end='')

        avg_rmse /= (test_num_samples - nan_count)
        avg_psnr /= test_num_samples
        avg_ssim /= test_num_samples
        print("\r\t-Test: Took: %d seconds \tAvg RMSE: %f \tAvg PSNR: %f \tAvg SSIM: %f" %
              (end - start, avg_rmse, avg_psnr, avg_ssim))
        # save evaluation results
        with open(os.path.join(save_path, "evaluation.txt"), 'a') as f:
            f.write("Test: Avg RMSE: %f \tAvg PSNR: %f \tAvg SSIM: %f\n" % (avg_rmse, avg_psnr, avg_ssim))
    return avg_rmse, avg_psnr, 1-avg_ssim


def test(args):
    if args.datasetSPP == 0:
        spp = [4, 8, 16, 32, 128, 256, 512, 1024]
    elif args.datasetSPP in [4, 8, 16, 32, 128, 256, 512, 1024]:
        spp = [args.datasetSPP]

    for s in spp:
        print('SPP:', s)
        test_dataset_path = os.path.join(args.inDir, "test_%d.h5" % s)

        test_dataset = Dataset(test_dataset_path)
        test_num_samples = len(test_dataset)
        test_dataloader = DataLoaderX(test_dataset, batch_size=1, num_workers=7, pin_memory=True)  # prefetch

        save_path = create_folder(os.path.join(args.outDir, "%dspp" % s), still_create=True)  # path to save model, imgs

        rmse, psnr, ssim = test_AFGSA(args, test_dataloader, test_num_samples, save_path)
        with open(os.path.join(args.outDir, "all_test.txt"), 'a') as f:
            f.write("%d \tRMSE: %f \tPSNR: %f \t1-SSIM: %f\n" % (s, rmse, psnr, ssim))


if __name__ == "__main__":
    create_folder(args.outDir)
    test(args)
