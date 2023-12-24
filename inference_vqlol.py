import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
# from yaml import load

from utils import img2tensor, tensor2img, imwrite 
from models.vqlol_arch import VQLOL
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' CUDA_VISIBLE_DEVICES=1 python basicsr/test.py -opt options/test_LOLBlur_LQ_stage_AIEM.yml

def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/input_dir', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default='./model_weights/VQCNIR_LOLBlur_G.pth', help='path for model weights') 
    parser.add_argument('-o', '--output', type=str, default='./results_test_LOLBlur/enhanced', help='Output folder')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=6000, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # set up the model
    sr_model = VQLOL(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=1).to(device)
    sr_model.load_state_dict(torch.load(args.weight)['params'], strict=False)
    sr_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = args.max_size ** 2 
        h, w = img_tensor.shape[2:]
        if h * w < max_size: 
            output = sr_model.test(img_tensor)
        else:
            output = sr_model.test_tile(img_tensor)
        output_img = tensor2img(output)

        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
