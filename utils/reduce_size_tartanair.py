
import argparse
import os
import cv2


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Reduce size of TARTAN dataset")

    parser.add_argument('--path_root', type=str, required=True,
                        help="Path to TARTANAIR dataset")
    parser.add_argument('--path_out', type=str, required=False,
                        help="Path to reduce TARTAN dataset")
    parser.add_argument('--scale', type=float, required=False,
                        default=2, help='The scale for reducing the size')
    args = parser.parse_args()

    for split in ['train','test','val']:
        for env in os.listdir(os.path.join(args.path_root, split)):
            for seq in os.listdir(os.path.join(args.path_root, split, env, 'Easy')):

                src = os.path.join(args.path_root, split, env, 'Easy', seq)
                dst = os.path.join(args.path_out, split, env, 'Easy', seq)

                for cam in ['right','left']:

                    os.makedirs(os.path.join(dst, 'image_' + cam))

                    for im_idx in os.listdir(os.path.join(src, 'image_' + cam)):
                        name = os.path.join('image_' + cam, im_idx + "_" + cam + ".png")
                        
                        im = cv2.imread(os.path.join(src, name))
                        H,W,C = im.shape
                        cv2.resize(im, dsize=(W,H), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(os.path.join(dst, name))

                for cam in ['right','left']:

                    os.makedirs(os.path.join(dst, 'seg_' + cam))

                    for im_idx in os.listdir(os.path.join(src, 'seg_' + cam)):
                        name = os.path.join('seg_' + cam, im_idx + "_" + cam + ".png")
                        
                        im = cv2.imread(os.path.join(src, name))
                        H,W,C = im.shape
                        cv2.resize(im, dsize=(W,H), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(os.path.join(dst, name))


                for cam in ['right','left']:

                    os.makedirs(os.path.join(dst, 'depth_' + cam))

                    for im_idx in os.listdir(os.path.join(src, 'depth_' + cam)):
                        name = os.path.join('depth_' + cam, im_idx + "_" + cam + ".png")
                        
                        im = cv2.imread(os.path.join(src, name))
                        H,W,C = im.shape
                        cv2.resize(im, dsize=(W,H), interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(os.path.join(dst, name))
