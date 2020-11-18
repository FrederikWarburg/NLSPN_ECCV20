"""
    This script generates a json file for the TARTANAIR Depth Completion dataset.
"""

import os
import argparse
import random
import json
import numpy as np

parser = argparse.ArgumentParser(
    description="Realsense Depth Completion json generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the Realsene Depth Completion dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='realsense.json', help="Output file name")
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


def align_with_rgb(list_rgb, list_depth):

    list_rgb = np.asarray([int(i.replace('.png','')) for i in list_rgb])
    list_depth = np.asarray([int(i.replace('.png','')) for i in list_depth])

    aligned_depth = []
    for rgb in list_rgb:
        #print(list_depth - rgb)
        closest = np.argmin(list_depth - rgb)
        #print(closest)
        aligned_depth.append(str(list_depth[closest]) + '.png')
        #import pdb; pdb.set_trace()
    return aligned_depth

def generate_json():
    check_dir_existence(args.path_out)

    # For train/val splits
    dict_json = {}
    for split in ['train','val','test']:
        path_split = os.path.join(args.path_root, split)

        list_pairs = []

        list_seq = os.listdir(path_split)
        list_seq.sort() 

        for seq in list_seq:
            cnt_seq = 0

            for cam in ['cam0']:
                list_rgb = os.listdir(os.path.join(path_split,seq,'cam0/data'))
                list_rgb.sort()

                list_depth = os.listdir(os.path.join(path_split,seq,'depth0/data'))
                list_depth.sort()
                list_depth = align_with_rgb(list_rgb, list_depth)
                
                for rgb_name, depth_name in zip(list_rgb, list_depth):
                    path_rgb = os.path.join(split, seq, cam, 'data', rgb_name)
                    path_depth = os.path.join(split, seq, 'depth0', 'data', depth_name)

                    dict_sample = {
                        'rgb': path_rgb,
                        'depth': path_depth
                        #'K': path_calib
                    }

                    flag_valid = True
                    for val in dict_sample.values():
                        flag_valid &= os.path.exists(args.path_root + '/' + val)

                        if not flag_valid:
                            print("not valid", args.path_root + '/' + val)
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
