"""
    This script generates a json file for the TARTANAIR Depth Completion dataset.
"""

import os
import argparse
import random
import json

parser = argparse.ArgumentParser(
    description="TartanAir Depth Completion jason generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the Tartan Depth Completion dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='tartanair.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False,
                    default=int(1e10), help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,
                    default=int(1e10), help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,
                    default=int(1e10), help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,
                    default=7240, help='Random seed')

args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), \
        "File does not exist : {}".format(path_file)


def generate_json():
    check_dir_existence(args.path_out)

    # For train/val splits
    dict_json = {}
    for split in ['train', 'val', 'test']:
        path_base = args.path_root + '/' + split

        list_seq = os.listdir(path_base)
        list_seq.sort()

        list_pairs = []
        for seq in list_seq:
            cnt_seq = 0

            for cam in ['cam0']:
                list_depth = os.listdir(os.path.join(path_base,seq,'depth_sparse0/data'))
                list_depth.sort()
                print(list_depth)
                for name in list_depth:
                    path_rgb = os.path.join(split,seq, cam, 'data', name.replace('.csv','.png'))
                    path_depth_features = os.path.join(split, seq, 'depth_sparse0', 'data', name)
                    path_depth_sgbm = os.path.join(split, seq, 'depth_SGBM0', 'data', name.replace('.csv', '_depth.npy'))
                    path_confidence_sgbm = os.path.join(split, seq, 'depth_SGBM0', 'data', name.replace('.csv', '_uncertainty.npy'))
                    path_gt = os.path.join(split, seq, 'ground_truth/depth0', 'data', name.replace('.csv','.npy'))
                    path_seg = os.path.join(split, seq, 'ground_truth/seg0', 'data', name.replace('.csv','.npy'))
                    #path_calib = split + '/' + seq + '/calib_cam_to_cam.txt'

                    dict_sample = {
                        'rgb': path_rgb,
                        'depth_features': path_depth_features,
                        'depth_sgbm': path_depth_sgbm,
                        'confidence_sgbm' : path_confidence_sgbm,
                        'gt': path_gt,
                        'seq': path_seg,
                        #'K': path_calib
                    }

                    flag_valid = True
                    for val in dict_sample.values():
                        flag_valid &= os.path.exists(args.path_root + '/' + val)

                        if not flag_valid:
                            print(args.path_root + '/' + val)
                            break

                    if not flag_valid:
                        continue
                    list_pairs.append(dict_sample)
                    cnt_seq += 1

            print("{} : {} samples".format(seq, cnt_seq))

        dict_json[split] = list_pairs
        print("{} split : Total {} samples".format(split, len(list_pairs)))

    # Cut if maximum is set
    for s in [('train', args.num_train), ('val', args.num_val),
              ('test', args.num_test)]:
        if len(dict_json[s[0]]) > s[1]:
            # Do shuffle
            random.shuffle(dict_json[s[0]])

            num_orig = len(dict_json[s[0]])
            dict_json[s[0]] = dict_json[s[0]][0:s[1]]
            print("{} split : {} -> {}".format(s[0], num_orig,
                                               len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    generate_json()
